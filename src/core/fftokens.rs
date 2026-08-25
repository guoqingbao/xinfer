// FF-tokens drafter -- INERT.
//
// The pure-FF drafter is not wired into the engine dispatch and must be realigned before use.
// The shared speculative core is anchor-first: it samples the anchor (a model forward), then the
// drafter speculates on top of it. A true FF drafter has no model forward at all -- the grammar's
// forced run *is* the output, and the "anchor" must be created by the FF step itself rather than
// consumed from one. Re-enable once the core supports anchor-less (ff-only) proposals.
//
// use candle_core::{Result, Tensor};
//
// use crate::core::mtp::MtpSeqInfo;
// use crate::core::runner::{ModelRunner, Seqs};
// use crate::core::speculative::Drafter;
//
// /// A drafter that emits the grammar's forced token run (all infallible).
// pub struct FfDrafter;
//
// impl Drafter for FfDrafter {
//     fn name(&self) -> &'static str {
//         "ff"
//     }
//
//     fn anchor(&self, runner: &ModelRunner, seqs: Seqs, _seq: &MtpSeqInfo) -> Result<(u32, Option<Tensor>)> {
//         // Plain (grammar-aware) decode step: the anchor token.
//         let anchor = runner.run(seqs, false)?[0];
//         Ok((anchor, None))
//     }
//
//     fn draft(
//         &self,
//         _runner: &ModelRunner,
//         _seq: &MtpSeqInfo,
//         _anchor: u32,
//         _hidden: &Option<Tensor>,
//     ) -> Result<Vec<u32>> {
//         // Pure FF: no model speculation. The core emits the grammar-forced run.
//         Ok(vec![])
//     }
// }