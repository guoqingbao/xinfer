//! xinfer-side DFA GPU benchmark using real llguidance grammar export.
//!
//! Run: cargo run --features cuda --example bench_dfa_xinfer
//!
//! Uses a real grammar (xbot tool-calling CFI) exported via llguidance,
//! uploaded to GPU, and benchmarked at multiple batch sizes.
//! Verifies accuracy against the CPU DFA reference.

use std::time::Instant;

fn main() {
    // Build a real grammar via llguidance
    use llguidance::{api::TopLevelGrammar, ParserFactory};
    use toktrie::ApproximateTokEnv;

    let env = ApproximateTokEnv::single_byte_env();
    let factory = ParserFactory::new_simple(&env).unwrap();

    // A realistic grammar: tool-calling envelope with 3 tools
    let grm_str = r#"
start: envelope
envelope: "TO" tool_name "{" params "}"
tool_name: "read_file" | "write_file" | "exec"
params: (param ",")* param
param: key ":" value
key: /[a-z_]+/
value: /[^,}]+/
to: "TO""
"#;
    let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
    grm.max_tokens = None;

    let parser = factory.create_parser(grm.clone()).unwrap();
    let dfa = parser.export_hw_dfa(10_000).expect("DFA export failed");
    println!(
        "Grammar: {} states, {} edges, {} KB GPU table",
        dfa.num_states,
        dfa.edges.len(),
        dfa.gpu_memory_bytes() / 1024
    );

    // Upload to GPU
    let dev = candle_core::Device::new_cuda(0).unwrap();
    let cuda_dev = dev.as_cuda_device().unwrap();
    let mask_sign: Vec<u8> = dfa.mask_sign.clone();
    let edge_offsets: Vec<u32> = dfa.edge_offsets.clone();
    let edge_counts: Vec<u32> = dfa.edge_counts.clone();
    let edge_tokens: Vec<u32> = dfa.edges.iter().map(|e| e.token).collect();
    let edge_next: Vec<u32> = dfa.edges.iter().map(|e| e.next_state).collect();
    let mask_words: Vec<u32> = dfa.mask_words.clone();
    let universal_target: Vec<u32> = dfa.universal_target.clone();

    let gpu_table = attention_rs::dfa::DfaGpuTable::upload(
        mask_sign, edge_offsets, edge_counts,
        edge_tokens, edge_next, mask_words, universal_target,
        dfa.num_states, dfa.words_per_vob, &dev,
    ).unwrap();
    println!("GPU table uploaded ({} KB VRAM)", dfa.gpu_memory_bytes() / 1024);

    let Build a valid token sequence to walk through the grammar
    // (single-byte env: each char is its ASCII value)
    let sequence: Vec<u32> = b"TOOLread_file{\"path\":\"/tmp\"}".iter().map(|&b| b as u32).collect();
    let K = sequence.len().min(16);
    let vocab = env.tok_trie().vocab_size();

    // === BENCHMARK at multiple batch sizes ===
    let iterations = 5_000;
    println!("\n=== BENCHMARK (K={vocab}, states={K}, {iterations} iters) ===");
    println!("{:<8} {:>12} {:>12} {:>12} {:>10}", "batch", "sample_adv", "validate", "project", "vs CPU");

    for cpu_parser_time = {
        // Time CPU parser walk (reference)
        let t0 = Instant::now();
        for _ in 0..100 {
            let mut p = factory.create_parser(grm.clone()).unwrap();
            p.start_without_prompt();
            for &tok in &sequence {
                let _ = p.compute_mask().unwrap();
                let _ = p.consume_token(tok).unwrap();
            }
        }
        t0.elapsed().as_micros() as f64 / 100.0
    };

    for &batch in &[1usize, 4, 8, 16, 32, 64] {
        let logits = candle_core::Tensor::zeros((batch, vocab), candle_core::DType::F32, &dev).unwrap();
        let states = candle_core::Tensor::zeros((batch,), candle_core::DType::U32, &dev).unwrap();
        let draft = candle_core::Tensor::zeros((batch, K), candle_core::DType::U32, &dev).unwrap();

        // Warmup
        for _ in 0..50 {
            let _ = gpu_table.sample_and_advance(&logits, &states, Some(&draft)).unwrap();
        }
        cuda_dev.synchronize().unwrap();

        // sample_and_advance
        let t0 = Instant::now();
        for _ in 0..iterations {
            let _ = gpu_table.sample_and_advance(&logits, &states, Some(&draft)).unwrap();
        }
        cuda_dev.synchronize().unwrap();
        let t_sample = t0.elapsed().as_micros() as f64 / iterations as f64;

        // validate_draft
        let t0 = Instant::now();
        for _ in 0..iterations {
            let _ = gpu_table.validate_draft(&states, &draft).unwrap();
        }
        cuda_dev.synchronize().unwrap();
        let t_validate = t0.elapsed().as_micros() as f64 / iterations as f64;

        // project_masks
        let t0 = Instant::now();
        for _ in 0..iterations {
            let _ = gpu_table.project_masks(&states, &draft).unwrap();
        }
        cuda_dev.synchronize().unwrap();
        let t_project = t0.elapsed().as_micros() as f64 / iterations as f64;

        let per_seq = t_sample / batch as f64;
        let speedup = cpu_parser_time / per_seq.max(0.001);
        println!(
            "{batch:<8} {:>10.2}us {:>10.2}us {:>10.2}us {:>8.0}x",
            t_sample, t_validate, t_project, speedup
        );
    }

    println!("\nCPU parser reference: {cpu_parser_time:.1} us/seq (full grammar walk)");

    // === ACCURACY: verify validate_draft vs CPU DFA advance ===
    println!("\n=== ACCURACY ===");
    let states0 = candle_core::Tensor::from_vec(vec![dfa.start_state], (1,), &dev).unwrap();
    let draft_cpu: Vec<u32> = sequence[..K].to_vec();
    let draft_t = candle_core::Tensor::from_vec(draft_cpu.clone(), (1, K), &dev).unwrap();
    let gpu_reject = gpu_table.validate_draft(&states0, &draft_t).unwrap()
        .flatten_all().unwrap().to_vec1::<u32>().unwrap()[0] as usize;

    // CPU reference
    let mut cpu_state = dfa.start_state;
    let mut cpu_reject = K;
    for (i, &tok) in draft_cpu.iter().enumerate() {
        match dfa.advance(cpu_state, tok) {
            Some(next) => cpu_state = next,
            None => { cpu_reject = i; break; }
        }
    }
    let!(
        "validate_draft: GPU reject={gpu_reject}, CPU reject={cpu_reject} -> {}",
        if gpu_reject == cpu_reject { "MATCH" } else { "MISMATCH!" }
    );
    assert_eq!(gpu_reject, cpu_reject, "GPU validate_draft does not match CPU reference");

    // project_masks accuracy
    let proj = gpu_table.project_masks(&states0, &draft_t).unwrap()
        .flatten_all().unwrap().to_vec1::<u32>().unwrap();
    let mut cpu_masks = vec![dfa.mask_at(dfa.start_state).to_vec()];
    let mut s = dfa.start_state;
    for &tok in &draft_cpu {
        s = dfa.advance(s, tok).unwrap_or(0);
        cpu_masks.push(dfa.mask_at(s).to_vec());
    }
    let cpu_flat: Vec<u32> = cpu_masks.iter().flatten().copied().collect();
    let proj_ok = proj == cpu_flat;
    println!("project_masks:  GPU == CPU -> {}", if proj_ok { "MATCH" } else { "MISMATCH!" });
    assert!(proj_ok, "GPU project_masks does not match CPU reference");

    println!("\nAll accuracy checks passed.");
}