use super::compressor::{CompressorDecodeState, CompressorWeights};
use super::rope_cache::V4RopeTables;
#[cfg(feature = "nccl")]
use crate::models::layers::distributed::AllReduce;
use crate::models::layers::distributed::Comm;
use crate::models::layers::VarBuilderX;
use candle_core::{DType, Device, Result, Tensor, D};

/// Indexer weights for ratio=4 layers: learned top-k selection over compressed KV.
///
/// The indexer follows the reference tensor-parallel layout: query heads and
/// per-head weights are local to each rank, then the score vector is summed
/// across ranks before top-k selection.
pub struct IndexerWeights {
    pub wq_b: Tensor,
    pub wq_b_scale: Option<Tensor>,
    pub weights_proj: Tensor,
    pub compressor: CompressorWeights,
    pub index_head_dim: usize,
    pub global_index_n_heads: usize,
    pub index_n_heads: usize,
    pub index_topk: usize,
    #[cfg(feature = "nccl")]
    score_all_reduce: Option<AllReduce>,
}

impl IndexerWeights {
    pub fn load(
        vb: &VarBuilderX,
        prefix: &str,
        hidden_dim: usize,
        index_head_dim: usize,
        index_n_heads: usize,
        index_topk: usize,
        q_lora_rank: usize,
        _rope_head_dim: usize,
        _n_shards: usize,
        comm: std::rc::Rc<Comm>,
        _layer_idx: usize,
        _rope_base: f64,
    ) -> Result<Option<Self>> {
        if index_n_heads % comm.world_size() != 0 {
            candle_core::bail!(
                "DeepSeek V4 index heads {} not divisible by TP world size {}",
                index_n_heads,
                comm.world_size()
            );
        }
        let wq_b_out = index_n_heads * index_head_dim;

        let indexer_prefix = if prefix.is_empty() {
            "indexer".to_string()
        } else {
            format!("{prefix}.indexer")
        };

        let has_scale = vb.has_key(&format!("{indexer_prefix}.wq_b.scale"));

        let wq_b = if has_scale {
            vb.get_with_hints_dtype(
                (wq_b_out, q_lora_rank),
                &format!("{indexer_prefix}.wq_b.weight"),
                crate::models::layers::distributed::shard(0, comm.rank(), comm.world_size()),
                DType::U8,
            )?
        } else {
            vb.get_with_hints_dtype(
                (wq_b_out, q_lora_rank),
                &format!("{indexer_prefix}.wq_b.weight"),
                crate::models::layers::distributed::shard(0, comm.rank(), comm.world_size()),
                DType::BF16,
            )?
        };

        let wq_b_scale = if has_scale {
            Some(vb.get_with_hints_dtype(
                (wq_b_out / 128, q_lora_rank / 128),
                &format!("{indexer_prefix}.wq_b.scale"),
                crate::models::layers::distributed::shard(0, comm.rank(), comm.world_size()),
                DType::F32,
            )?)
        } else {
            None
        };

        let weights_proj = vb.get_with_hints_dtype(
            (index_n_heads, hidden_dim),
            &format!("{indexer_prefix}.weights_proj.weight"),
            crate::models::layers::distributed::shard(0, comm.rank(), comm.world_size()),
            DType::BF16,
        )?;

        let compressor = CompressorWeights::load(
            vb,
            &format!("{indexer_prefix}.compressor"),
            4,
            index_head_dim,
            hidden_dim,
        )?
        .expect("indexer compressor should exist for ratio=4");

        Ok(Some(Self {
            wq_b,
            wq_b_scale,
            weights_proj,
            compressor,
            index_head_dim,
            global_index_n_heads: index_n_heads,
            index_n_heads: index_n_heads / comm.world_size(),
            index_topk,
            #[cfg(feature = "nccl")]
            score_all_reduce: (comm.world_size() > 1).then(|| AllReduce::new(comm.clone())),
        }))
    }

    fn project_q(&self, qr: &Tensor, _is_prefill: bool) -> Result<Tensor> {
        if let Some(scale) = &self.wq_b_scale {
            attention_rs::fp8_linear::fp8_matmul_ue8m0(qr, &self.wq_b, scale, &[128, 128])
        } else {
            qr.matmul(&self.wq_b.t()?)
        }
    }

    fn prepare_q(
        &self,
        qr: &Tensor,
        seq_len: usize,
        _rope: &V4RopeTables,
        _start_pos: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let q = self
            .project_q(qr, is_prefill)?
            .reshape((seq_len, self.index_n_heads, self.index_head_dim))?
            .contiguous()?;
        // The indexer query follows the same QAT transform as OpenInfer: RoPE
        // is applied per index head, then the 128-wide head is Hadamard
        // rotated and FP4 round-tripped before score computation.
        let q = q.contiguous()?;
        self.rope_query(&q, _rope, _start_pos)
    }

    fn rope_query(&self, q: &Tensor, rope: &V4RopeTables, start_pos: usize) -> Result<Tensor> {
        let seq_len = q.dim(0)?;
        rope.apply_inplace(q, start_pos, false)?;
        attention_rs::deepseek_v4::hadamard_fp4_quant_bf16_inplace(
            q,
            self.index_n_heads,
            self.index_head_dim,
        )?;
        q.reshape((seq_len, self.index_n_heads, self.index_head_dim))?
            .contiguous()
    }

    fn rope_query_from_positions(
        &self,
        q: &Tensor,
        rope: &V4RopeTables,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = q.dim(0)?;
        rope.apply_from_positions(q, positions, 0, false)?;
        attention_rs::deepseek_v4::hadamard_fp4_quant_bf16_inplace(
            q,
            self.index_n_heads,
            self.index_head_dim,
        )?;
        q.reshape((seq_len, self.index_n_heads, self.index_head_dim))?
            .contiguous()
    }

    pub fn scores_prefill(
        &self,
        input: &Tensor,
        qr: &Tensor,
        seq_len: usize,
        compressed_len: usize,
        rope: &V4RopeTables,
    ) -> Result<Tensor> {
        let score_scale =
            1.0 / (self.index_head_dim as f32).sqrt() / (self.global_index_n_heads as f32).sqrt();

        let q = self.prepare_q(qr, seq_len, rope, 0, true)?;
        let q = q.reshape((seq_len, self.index_n_heads * self.index_head_dim))?;

        let compressed_kv = self
            .compressor
            .prefill(input, seq_len, Some(rope), 0, true)?;

        let weights = input.matmul(&self.weights_proj.t()?)?;

        let scores = attention_rs::deepseek_v4::indexer_scores_prefill(
            &q,
            &compressed_kv,
            &weights,
            seq_len,
            self.index_n_heads,
            self.index_head_dim,
            compressed_len,
            score_scale,
        )?;
        #[cfg(feature = "nccl")]
        let scores = if let Some(all_reduce) = &self.score_all_reduce {
            scores.apply_op1_no_bwd(all_reduce)?
        } else {
            scores
        };
        Ok(scores)
    }

    pub fn scores_decode(
        &self,
        input: &Tensor,
        qr: &Tensor,
        indexer_kv_cache: &Tensor,
        compressed_len: usize,
        rope: &V4RopeTables,
        start_pos: usize,
    ) -> Result<Tensor> {
        let score_scale =
            1.0 / (self.index_head_dim as f32).sqrt() / (self.global_index_n_heads as f32).sqrt();

        let q = self.prepare_q(qr, 1, rope, start_pos, false)?;
        let q = q.reshape((1, self.index_n_heads * self.index_head_dim))?;
        let weights = input.matmul(&self.weights_proj.t()?)?;

        let scores = attention_rs::deepseek_v4::indexer_scores_decode(
            &q,
            indexer_kv_cache,
            &weights,
            self.index_n_heads,
            self.index_head_dim,
            compressed_len,
            score_scale,
        )?;
        #[cfg(feature = "nccl")]
        let scores = if let Some(all_reduce) = &self.score_all_reduce {
            scores.apply_op1_no_bwd(all_reduce)?
        } else {
            scores
        };
        Ok(scores)
    }

    pub fn scores_decode_from_positions_into(
        &self,
        input: &Tensor,
        qr: &Tensor,
        indexer_kv_cache: &Tensor,
        score_len: usize,
        rope: &V4RopeTables,
        positions: &Tensor,
        scores: &Tensor,
    ) -> Result<()> {
        let score_scale =
            1.0 / (self.index_head_dim as f32).sqrt() / (self.global_index_n_heads as f32).sqrt();

        let q = self
            .project_q(qr, false)?
            .reshape((1, self.index_n_heads, self.index_head_dim))?
            .contiguous()?;
        let q = self.rope_query_from_positions(&q, rope, positions)?;
        let q = q.reshape((1, self.index_n_heads * self.index_head_dim))?;
        let weights = input.matmul(&self.weights_proj.t()?)?;

        attention_rs::deepseek_v4::indexer_scores_decode_into(
            &q,
            indexer_kv_cache,
            &weights,
            self.index_n_heads,
            self.index_head_dim,
            score_len,
            score_scale,
            scores,
        )?;
        #[cfg(feature = "nccl")]
        if let Some(all_reduce) = &self.score_all_reduce {
            let reduced = scores.apply_op1_no_bwd(all_reduce)?;
            scores.copy_(&reduced, 0)?;
        }
        Ok(())
    }

    pub fn scores_decode_from_positions(
        &self,
        input: &Tensor,
        qr: &Tensor,
        indexer_kv_cache: &Tensor,
        score_len: usize,
        rope: &V4RopeTables,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let score_scale =
            1.0 / (self.index_head_dim as f32).sqrt() / (self.global_index_n_heads as f32).sqrt();

        let q = self
            .project_q(qr, false)?
            .reshape((1, self.index_n_heads, self.index_head_dim))?
            .contiguous()?;
        let q = self.rope_query_from_positions(&q, rope, positions)?;
        let q = q.reshape((1, self.index_n_heads * self.index_head_dim))?;
        let weights = input.matmul(&self.weights_proj.t()?)?;

        let scores = attention_rs::deepseek_v4::indexer_scores_decode(
            &q,
            indexer_kv_cache,
            &weights,
            self.index_n_heads,
            self.index_head_dim,
            score_len,
            score_scale,
        )?;
        #[cfg(feature = "nccl")]
        let scores = if let Some(all_reduce) = &self.score_all_reduce {
            scores.apply_op1_no_bwd(all_reduce)?
        } else {
            scores
        };
        Ok(scores)
    }

    /// Score queries against an existing indexer compressed cache (continued prefill).
    pub fn scores_prefill_against_cache(
        &self,
        input: &Tensor,
        qr: &Tensor,
        indexer_kv: &Tensor,
        compressed_len: usize,
        rope: &V4RopeTables,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = input.dim(0)?;
        let score_scale =
            1.0 / (self.index_head_dim as f32).sqrt() / (self.global_index_n_heads as f32).sqrt();

        let q = self
            .project_q(qr, true)?
            .reshape((seq_len, self.index_n_heads, self.index_head_dim))?
            .contiguous()?;
        let q = self.rope_query_from_positions(&q, rope, positions)?;
        let q = q.reshape((seq_len, self.index_n_heads * self.index_head_dim))?;
        let weights = input.matmul(&self.weights_proj.t()?)?;
        let kv = indexer_kv
            .narrow(0, 0, compressed_len.max(1))?
            .contiguous()?;

        let scores = attention_rs::deepseek_v4::indexer_scores_prefill(
            &q,
            &kv,
            &weights,
            seq_len,
            self.index_n_heads,
            self.index_head_dim,
            compressed_len.max(1),
            score_scale,
        )?;
        #[cfg(feature = "nccl")]
        let scores = if let Some(all_reduce) = &self.score_all_reduce {
            scores.apply_op1_no_bwd(all_reduce)?
        } else {
            scores
        };
        Ok(scores)
    }

    pub fn topk_prefill(
        &self,
        scores: &Tensor,
        seq_len: usize,
        compressed_len: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let topk = self.index_topk.min(compressed_len);
        attention_rs::deepseek_v4::indexer_topk_prefill(
            scores,
            seq_len,
            compressed_len,
            topk,
            4,
            offset,
        )
    }

    /// Top-k with per-query absolute positions (continued / fresh prefill).
    pub fn topk_prefill_from_pos(
        &self,
        scores: &Tensor,
        positions: &Tensor,
        compressed_len: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let seq_len = scores.dim(0)?;
        let topk = self.index_topk.min(compressed_len.max(1));
        attention_rs::deepseek_v4::indexer_topk_prefill_from_pos(
            scores,
            positions,
            seq_len,
            compressed_len.max(1),
            topk,
            self.compressor.ratio,
            offset,
        )
    }

    pub fn topk_decode_into(
        &self,
        scores: &Tensor,
        compressed_len: usize,
        offset: usize,
        topk_idxs: &Tensor,
    ) -> Result<()> {
        let topk = self.index_topk.min(compressed_len);
        attention_rs::deepseek_v4::indexer_topk_decode_into(
            scores,
            compressed_len,
            topk,
            offset,
            topk_idxs,
        )
    }

    pub fn topk_decode(
        &self,
        scores: &Tensor,
        compressed_len: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let topk = self.index_topk.min(compressed_len);
        attention_rs::deepseek_v4::indexer_topk_decode(scores, compressed_len, topk, offset)
    }
}

pub struct IndexerDecodeState {
    pub compressor_state: CompressorDecodeState,
    pub kv_cache: Tensor,
    pub compressed_len: usize,
    pub max_compressed_len: usize,
}

impl IndexerDecodeState {
    pub fn new(index_head_dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        let max_compressed_len = max_seq_len.div_ceil(4).max(1);
        let compressor_state = CompressorDecodeState::new(4, index_head_dim, device)?;
        let kv_cache = Tensor::zeros((max_compressed_len, index_head_dim), DType::BF16, device)?;

        Ok(Self {
            compressor_state,
            kv_cache,
            compressed_len: 0,
            max_compressed_len,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.compressor_state.reset()?;
        self.kv_cache.zero_()?;
        self.compressed_len = 0;
        Ok(())
    }

    pub fn seed_from_prefill(&mut self, compressed: &Tensor) -> Result<()> {
        let n = compressed.dim(0)?;
        if n > self.max_compressed_len {
            candle_core::bail!(
                "indexer prefill len {n} exceeds capacity {}",
                self.max_compressed_len
            );
        }
        if n > 0 {
            let compressed = compressed.contiguous()?;
            attention_rs::deepseek_v4::copy_contiguous_into(&self.kv_cache, &compressed, 0)?;
        }
        self.compressed_len = n;
        Ok(())
    }

    pub fn append_compressed(&mut self, row: &Tensor) -> Result<()> {
        let i = self.compressed_len;
        self.write_compressed_at(row, i)
    }

    /// Official decode writes at `start_pos // ratio` (assignment, not append).
    pub fn write_compressed_at(&mut self, row: &Tensor, index: usize) -> Result<()> {
        if index >= self.max_compressed_len {
            candle_core::bail!(
                "indexer compressed index {index} exceeds capacity {}",
                self.max_compressed_len
            );
        }
        let dim = row.dim(D::Minus1)?;
        let row = row.reshape((1, dim))?.contiguous()?;
        attention_rs::deepseek_v4::copy_contiguous_into(&self.kv_cache, &row, index * dim)?;
        self.compressed_len = self.compressed_len.max(index + 1);
        Ok(())
    }

    pub fn append_compressed_rows_at(&mut self, rows: &Tensor, start_row: usize) -> Result<()> {
        let n = rows.dim(0)?;
        if start_row + n > self.max_compressed_len {
            candle_core::bail!(
                "indexer rows [{start_row}, {}) exceed capacity {}",
                start_row + n,
                self.max_compressed_len
            );
        }
        if n > 0 {
            let dim = rows.dim(D::Minus1)?;
            let rows = rows.contiguous()?;
            attention_rs::deepseek_v4::copy_contiguous_into(
                &self.kv_cache,
                &rows,
                start_row * dim,
            )?;
        }
        self.compressed_len = self.compressed_len.max(start_row + n);
        Ok(())
    }

    pub fn write_compressed_from_pos(
        &mut self,
        row: &Tensor,
        positions: &Tensor,
        ratio: usize,
    ) -> Result<()> {
        let dim = row.dim(D::Minus1)?;
        let row = row.reshape((1, dim))?.contiguous()?;
        attention_rs::deepseek_v4::write_indexer_row_from_pos(
            &self.kv_cache,
            &row,
            positions,
            dim,
            ratio,
        )?;
        Ok(())
    }
}
