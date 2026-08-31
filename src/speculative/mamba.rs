use crate::core::runner::{Model, ModelRunner};

impl ModelRunner {
    pub(crate) fn rollback_mamba_at(
        &self,
        seq_id: usize,
        keep_tokens: usize,
        snapshot_offset: usize,
    ) -> Result<bool, candle_core::Error> {
        match self.model() {
            Model::Qwen3_5(m) => m.mtp_rollback_mamba_at(seq_id, keep_tokens, snapshot_offset),
            Model::Qwen3_5MoE(m) => m.mtp_rollback_mamba_at(seq_id, keep_tokens, snapshot_offset),
            Model::Qwen3VL(m) => m.mtp_rollback_mamba_at(seq_id, keep_tokens, snapshot_offset),
            _ => Ok(false),
        }
    }

    pub(crate) fn rollback_mamba(
        &self,
        seq_id: usize,
        keep_tokens: usize,
    ) -> Result<bool, candle_core::Error> {
        self.rollback_mamba_at(seq_id, keep_tokens, 0)
    }
}
