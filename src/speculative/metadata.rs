use crate::core::runner::ModelRunner;
use attention_rs::InputMetadata;
use candle_core::{Result, Tensor};

/// Per-sequence metadata used when building speculative verify batches.
#[derive(Debug, Clone)]
pub struct SpecSeqInfo {
    pub id: usize,
    pub len: usize,
    pub block_table: Vec<u32>,
}

impl ModelRunner {
    pub(crate) fn compute_slot_mappings(
        &self,
        seq_info: &SpecSeqInfo,
        num_tokens: usize,
        block_size: usize,
        ctx: &str,
    ) -> Result<Vec<i64>> {
        let mut slots = Vec::with_capacity(num_tokens);
        for i in 0..num_tokens {
            let pos = seq_info.len + i;
            let block_idx = pos / block_size;
            let block_offset = pos % block_size;
            if block_idx < seq_info.block_table.len() {
                let physical_block = seq_info.block_table[block_idx] as i64;
                slots.push(physical_block * block_size as i64 + block_offset as i64);
            } else {
                candle_core::bail!(
                    "Speculative {} missing KV block: block_idx {} >= block_table.len() {}. \
                     Blocks must be pre-allocated before verify.",
                    ctx,
                    block_idx,
                    seq_info.block_table.len()
                );
            }
        }
        Ok(slots)
    }

    pub(crate) fn build_verify_metadata(
        &self,
        seq_info: &SpecSeqInfo,
        slot_mappings: &[i64],
        q_len: usize,
    ) -> Result<InputMetadata> {
        let total_kv_len = (seq_info.len + q_len) as u32;
        let mamba_slot_mapping = self.prepare_mamba_slot_mapping(&[seq_info.id], false)?;

        #[cfg(feature = "flashinfer")]
        let flashinfer_metadata = if let Some(params) = self.flashinfer_kv_params() {
            let num_pages = (total_kv_len as usize).div_ceil(params.page_size);
            if num_pages > seq_info.block_table.len() {
                candle_core::bail!(
                    "Speculative verify needs {} KV pages for {} tokens, but only {} pages are allocated",
                    num_pages,
                    total_kv_len,
                    seq_info.block_table.len()
                );
            }
            let indptr_host = vec![0u32, num_pages as u32];
            let indices_vec = seq_info.block_table[..num_pages].to_vec();
            let last_page_tokens = if total_kv_len == 0 {
                0
            } else {
                (total_kv_len as usize - 1) % params.page_size + 1
            };
            let last_len_host = vec![last_page_tokens as u32];
            let kv_len_arr_host = vec![total_kv_len];
            let q_cu_seqlens_host = vec![0u32, q_len as u32];
            let batch_indices = Tensor::zeros((q_len,), candle_core::DType::U32, self.device())?;
            let append_positions = Tensor::from_vec(
                (seq_info.len as u32..total_kv_len).collect::<Vec<_>>(),
                (q_len,),
                self.device(),
            )?;

            #[cfg(all(feature = "cuda", feature = "graph"))]
            let use_graph = self
                .mtp_capturer
                .as_ref()
                .map_or(false, |c| c.is_mtp_captured(q_len));
            #[cfg(not(all(feature = "cuda", feature = "graph")))]
            let use_graph = false;

            let prefill_plan_info = if use_graph {
                None
            } else {
                Some(attention_rs::flashinfer::prefill_plan(
                    self.device(),
                    &q_cu_seqlens_host,
                    &indptr_host,
                    &kv_len_arr_host,
                    q_len as u32,
                    1,
                    params.num_qo_heads,
                    params.num_kv_heads,
                    params.head_dim,
                    params.page_size,
                    params.out_dtype,
                    None,
                    Some(params.kv_dtype),
                    false,
                )?)
            };

            Some(attention_rs::FlashInferMetadata {
                indptr: Tensor::from_vec(indptr_host.clone(), (2,), self.device())?,
                indptr_host,
                indices: Tensor::from_vec(indices_vec, (num_pages,), self.device())?,
                last_len: Tensor::from_vec(last_len_host.clone(), (1,), self.device())?,
                last_len_host: Some(last_len_host),
                kv_len_arr_host: Some(kv_len_arr_host),
                total_num_rows: Some(q_len as u32),
                batch_indices: Some(batch_indices),
                positions: Some(append_positions),
                use_cuda_graph: use_graph,
                decode_plan_info: None,
                prefill_plan_info,
                mla_decode_plan_info: None,
                mla_prefill_plan_info: None,
            })
        } else {
            None
        };
        #[cfg(not(feature = "flashinfer"))]
        let flashinfer_metadata = None;

        Ok(InputMetadata {
            is_prefill: true,
            is_mla: self.is_mla_model(),
            sequence_ids: Some(vec![seq_info.id]),
            mamba_slot_mapping,
            slot_mapping: Tensor::from_vec(slot_mappings.to_vec(), (q_len,), self.device())?,
            context_lens: Some(Tensor::from_vec(vec![total_kv_len], (1,), self.device())?),
            block_tables: Some(Tensor::from_vec(
                seq_info.block_table.clone(),
                (1, seq_info.block_table.len()),
                self.device(),
            )?),
            block_tables_host: Some(vec![seq_info.block_table.clone()]),
            context_lens_host: Some(vec![total_kv_len]),
            seqlens: None,
            cu_seqlens_q: Some(Tensor::from_vec(
                vec![0u32, q_len as u32],
                (2,),
                self.device(),
            )?),
            cu_seqlens_k: Some(Tensor::from_vec(
                vec![0u32, total_kv_len],
                (2,),
                self.device(),
            )?),
            max_seqlen_q: q_len,
            max_seqlen_k: seq_info.len + q_len,
            max_context_len: seq_info.len + q_len,
            flashinfer_metadata,
            is_mtp_verify: true,
        })
    }

    pub(crate) fn build_verify_metadata_batch(
        &self,
        seq_infos: &[SpecSeqInfo],
        slot_mappings: &[Vec<i64>],
        q_lens: &[usize],
    ) -> Result<InputMetadata> {
        if seq_infos.is_empty()
            || seq_infos.len() != slot_mappings.len()
            || seq_infos.len() != q_lens.len()
        {
            candle_core::bail!("Speculative verify batch metadata has inconsistent dimensions");
        }
        let batch_size = seq_infos.len();
        let total_q_len = q_lens.iter().sum::<usize>();
        let sequence_ids = seq_infos.iter().map(|seq| seq.id).collect::<Vec<_>>();
        let total_kv_lens = seq_infos
            .iter()
            .zip(q_lens)
            .map(|(seq, &q_len)| (seq.len + q_len) as u32)
            .collect::<Vec<_>>();
        let slot_mapping = slot_mappings.iter().flatten().copied().collect::<Vec<_>>();
        if slot_mapping.len() != total_q_len {
            candle_core::bail!("Speculative verify batch slot/query count mismatch");
        }
        let mamba_slot_mapping = self.prepare_mamba_slot_mapping(&sequence_ids, false)?;

        #[cfg(feature = "flashinfer")]
        let flashinfer_metadata = if let Some(params) = self.flashinfer_kv_params() {
            let mut indptr_host = vec![0u32];
            let mut indices_host = Vec::new();
            let mut last_len_host = Vec::with_capacity(batch_size);
            for (seq, &total_kv_len) in seq_infos.iter().zip(&total_kv_lens) {
                let num_pages = (total_kv_len as usize).div_ceil(params.page_size);
                if num_pages > seq.block_table.len() {
                    candle_core::bail!(
                        "Speculative verify needs {} pages for sequence {}, but only {} are allocated",
                        num_pages,
                        seq.id,
                        seq.block_table.len()
                    );
                }
                indices_host.extend_from_slice(&seq.block_table[..num_pages]);
                indptr_host.push(indices_host.len() as u32);
                last_len_host.push(((total_kv_len as usize - 1) % params.page_size + 1) as u32);
            }
            let mut q_cu_seqlens_host = vec![0u32];
            let mut batch_indices_host = Vec::with_capacity(total_q_len);
            let mut append_positions_host = Vec::with_capacity(total_q_len);
            for (batch_idx, (seq, &q_len)) in seq_infos.iter().zip(q_lens).enumerate() {
                q_cu_seqlens_host.push(q_cu_seqlens_host.last().copied().unwrap() + q_len as u32);
                batch_indices_host.extend(std::iter::repeat_n(batch_idx as u32, q_len));
                append_positions_host.extend(seq.len as u32..seq.len as u32 + q_len as u32);
            }
            let kv_len_arr_host = total_kv_lens.clone();
            let prefill_plan_info = Some(attention_rs::flashinfer::prefill_plan(
                self.device(),
                &q_cu_seqlens_host,
                &indptr_host,
                &kv_len_arr_host,
                total_q_len as u32,
                batch_size,
                params.num_qo_heads,
                params.num_kv_heads,
                params.head_dim,
                params.page_size,
                params.out_dtype,
                None,
                Some(params.kv_dtype),
                false,
            )?);
            Some(attention_rs::FlashInferMetadata {
                indptr: Tensor::from_vec(indptr_host.clone(), (indptr_host.len(),), self.device())?,
                indptr_host,
                indices: Tensor::from_vec(
                    indices_host.clone(),
                    (indices_host.len(),),
                    self.device(),
                )?,
                last_len: Tensor::from_vec(
                    last_len_host.clone(),
                    (last_len_host.len(),),
                    self.device(),
                )?,
                last_len_host: Some(last_len_host),
                kv_len_arr_host: Some(kv_len_arr_host),
                total_num_rows: Some(total_q_len as u32),
                batch_indices: Some(Tensor::from_vec(
                    batch_indices_host,
                    (total_q_len,),
                    self.device(),
                )?),
                positions: Some(Tensor::from_vec(
                    append_positions_host,
                    (total_q_len,),
                    self.device(),
                )?),
                use_cuda_graph: false,
                decode_plan_info: None,
                prefill_plan_info,
                mla_decode_plan_info: None,
                mla_prefill_plan_info: None,
            })
        } else {
            None
        };
        #[cfg(not(feature = "flashinfer"))]
        let flashinfer_metadata = None;

        let mut block_tables_host = Vec::with_capacity(batch_size);
        let max_blocks = seq_infos
            .iter()
            .map(|seq| seq.block_table.len())
            .max()
            .unwrap_or(0);
        let mut block_tables_flat = Vec::with_capacity(batch_size * max_blocks);
        for seq in seq_infos {
            block_tables_host.push(seq.block_table.clone());
            block_tables_flat.extend_from_slice(&seq.block_table);
            block_tables_flat.resize(
                block_tables_flat.len() + max_blocks - seq.block_table.len(),
                0,
            );
        }
        let mut cu_seqlens_q = vec![0u32];
        let mut cu_seqlens_k = vec![0u32];
        for (&q_len, &kv_len) in q_lens.iter().zip(&total_kv_lens) {
            cu_seqlens_q.push(cu_seqlens_q.last().copied().unwrap() + q_len as u32);
            cu_seqlens_k.push(cu_seqlens_k.last().copied().unwrap() + kv_len);
        }
        let max_seqlen_q = q_lens.iter().copied().max().unwrap_or(0);
        let max_seqlen_k = total_kv_lens.iter().copied().max().unwrap_or(0) as usize;
        Ok(InputMetadata {
            is_prefill: true,
            is_mla: self.is_mla_model(),
            sequence_ids: Some(sequence_ids),
            mamba_slot_mapping,
            slot_mapping: Tensor::from_vec(slot_mapping, (total_q_len,), self.device())?,
            block_tables: Some(Tensor::from_vec(
                block_tables_flat,
                (batch_size, max_blocks),
                self.device(),
            )?),
            block_tables_host: Some(block_tables_host),
            context_lens_host: Some(total_kv_lens.clone()),
            context_lens: Some(Tensor::from_vec(
                total_kv_lens,
                (batch_size,),
                self.device(),
            )?),
            cu_seqlens_q: Some(Tensor::from_vec(
                cu_seqlens_q,
                (batch_size + 1,),
                self.device(),
            )?),
            cu_seqlens_k: Some(Tensor::from_vec(
                cu_seqlens_k,
                (batch_size + 1,),
                self.device(),
            )?),
            max_seqlen_q,
            max_seqlen_k,
            max_context_len: max_seqlen_k,
            seqlens: None,
            flashinfer_metadata,
            is_mtp_verify: true,
        })
    }
}
