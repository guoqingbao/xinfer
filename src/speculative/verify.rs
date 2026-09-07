use candle_core::{Result, Tensor, D};

/// Rolling window size for speculative acceptance stats (recent decode steps).

/// Outcome of speculative verification for a single sequence.
#[derive(Debug, Clone)]
pub struct MtpVerifyResult {
    pub accepted_tokens: Vec<u32>,
    pub continuation_token: u32,
    pub num_accepted: usize,
    pub num_proposed: usize,
}

/// Verify draft tokens against target model logits (greedy / argmax).
pub fn verify_draft_greedy(
    verify_logits: &Tensor,
    draft_tokens: &[u32],
) -> Result<MtpVerifyResult> {
    let num_positions = verify_logits.dim(0)?;
    let num_proposed = draft_tokens.len();

    if num_positions == 0 || num_proposed == 0 {
        let first_token = if num_positions > 0 {
            verify_logits
                .get(0)?
                .argmax(D::Minus1)?
                .to_scalar::<u32>()?
        } else {
            0
        };
        return Ok(MtpVerifyResult {
            accepted_tokens: vec![],
            continuation_token: first_token,
            num_accepted: 0,
            num_proposed,
        });
    }

    let verify_logits = verify_logits.to_dtype(candle_core::DType::F32)?;
    let all_target_tokens = verify_logits.argmax(D::Minus1)?;
    let target_vec: Vec<u32> = all_target_tokens.to_vec1()?;

    let compare_len = num_proposed.min(num_positions);
    let mut num_accepted = 0;
    for i in 0..compare_len {
        if target_vec[i] == draft_tokens[i] {
            num_accepted += 1;
        } else {
            break;
        }
    }

    let accepted_tokens = draft_tokens[..num_accepted].to_vec();
    let continuation_token = if num_accepted < num_positions {
        target_vec[num_accepted]
    } else {
        target_vec[num_positions - 1]
    };

    Ok(MtpVerifyResult {
        accepted_tokens,
        continuation_token,
        num_accepted,
        num_proposed,
    })
}

/// Validate draft tokens against a PDA. Returns the number of leading tokens
/// that are grammar-legal (stops at first illegal token).
/// If `pda` is None, returns `tokens.len()` (no constraint).
pub fn pda_validate_draft(
    pda: Option<&pushdown_rs::machine::PdaMachine>,
    start_ctrl: u32,
    tokens: &[u32],
) -> usize {
    let Some(pda) = pda else { return tokens.len() };
    let mut ctrl = start_ctrl;
    let mut stack = vec![pda.start_stack];
    let mut count = 0;
    for &token in tokens {
        let top = stack.last().copied().unwrap_or(pda.start_stack);
        match pda.lookup(ctrl, Some(token), top).as_slice() {
            [t] => {
                // Advance: pop top, push the new symbols.
                stack.pop();
                for &p in t.push.iter().rev() {
                    stack.push(p);
                }
                ctrl = t.next_q;
                count += 1;
            }
            _ => break, // No transition: token is illegal.
        }
    }
    count
}

