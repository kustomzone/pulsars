//! pulsar-cli: greedy Hy3 decode for parity work.
//!
//!   pulsar-cli -m model.gguf -p "text" -n 32 [--ctx 2048] [--no-bos]
//!   pulsar-cli -m model.gguf --tokens 120000,16883,11 -n 32
//!
//! -p tokenizes raw text (BOS prepended unless --no-bos); --tokens feeds
//! exact ids, which is how A/B runs align with ds4 --dump-tokens output.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("pulsar-cli requires Linux + CUDA");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
fn main() {
    if let Err(e) = run() {
        eprintln!("pulsar-cli: {e}");
        std::process::exit(1);
    }
}

#[cfg(target_os = "linux")]
fn run() -> engine::Result {
    let mut model_path = None;
    let mut prompt = None;
    let mut tokens_arg = None;
    let mut n_predict = 16usize;
    let mut ctx = 2048u32;
    let mut bos = true;
    let mut dump_logits = None;
    let mut teacher_force = false;
    let mut decode_consistency = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut need = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "-m" => model_path = Some(need("-m")?),
            "-p" => prompt = Some(need("-p")?),
            "--tokens" => tokens_arg = Some(need("--tokens")?),
            "-n" => n_predict = need("-n")?.parse()?,
            "--ctx" => ctx = need("--ctx")?.parse()?,
            "--no-bos" => bos = false,
            "--dump-logits" => dump_logits = Some(need("--dump-logits")?),
            "--teacher-force" => teacher_force = true,
            "--decode-consistency" => decode_consistency = Some(need("--decode-consistency")?.parse::<usize>()?),
            other => return Err(format!("unknown arg {other}").into()),
        }
    }
    let model_path = model_path.ok_or("missing -m MODEL.gguf")?;

    eprintln!("pulsar: loading {model_path}");
    let t0 = std::time::Instant::now();
    let model = engine::Model::load(std::path::Path::new(&model_path))?;
    let tok = {
        let (_, g) = engine::parse_header(std::path::Path::new(&model_path))?;
        tokenizer::Tokenizer::from_gguf(&g)?
    };
    eprintln!(
        "pulsar: loaded in {:.1}s ({} layers, {} experts x top-{})",
        t0.elapsed().as_secs_f32(),
        model.shape.n_exec_layer,
        model.shape.n_expert,
        model.shape.n_expert_used
    );

    let prompt_ids: Vec<u32> = match (tokens_arg, prompt) {
        (Some(t), _) => t.split(',').map(|s| s.trim().parse()).collect::<std::result::Result<_, _>>()?,
        (None, Some(p)) => {
            let mut ids = Vec::new();
            if bos {
                ids.push(tok.bos_id.ok_or("model has no BOS id")?);
            }
            ids.extend(tok.encode(&p));
            ids
        }
        (None, None) => return Err("need -p TEXT or --tokens IDS".into()),
    };
    eprintln!("pulsar: prompt ids {prompt_ids:?}");

    let mut st = engine::State::new(&model, ctx)?;

    if teacher_force {
        // Per-position top-5 (id, logit) along the given token sequence,
        // one JSON line per position, for cross-engine agreement checks.
        for (i, &id) in prompt_ids.iter().enumerate() {
            let l = model.forward_token(&mut st, id, i as u32, true)?.unwrap();
            let mut top: Vec<u32> = (0..l.len() as u32).collect();
            top.sort_by(|&a, &b| l[b as usize].total_cmp(&l[a as usize]));
            let entries: Vec<String> = top[..5]
                .iter()
                .map(|&t| format!("[{},{}]", t, l[t as usize]))
                .collect();
            println!("{{\"pos\":{},\"after\":{},\"top\":[{}]}}", i, id, entries.join(","));
        }
        return Ok(());
    }

    if let Some(nsteps) = decode_consistency {
        // Greedy-decode nsteps tokens through the incremental (n_tok=1)
        // path, then fresh-prefill the identical sequence batched and
        // compare the logits at the same position. Divergence here is the
        // reduction-order drift between the batch and decode matmul
        // kernels - the ds4 --decode-consistency analogue.
        let mut logits = None;
        let mut pos0 = 0u32;
        for chunk in prompt_ids.chunks(st.max_batch() as usize) {
            logits = model.forward_batch(&mut st, chunk, pos0, true)?;
            pos0 += chunk.len() as u32;
        }
        let mut seq = prompt_ids.clone();
        for _ in 0..nsteps.saturating_sub(1) {
            let next = engine::argmax(logits.as_ref().ok_or("no logits")?);
            seq.push(next);
            logits = model.forward_batch(&mut st, &[next], seq.len() as u32 - 1, true)?;
        }
        let decode_logits = logits.ok_or("no logits")?;
        let decode_argmax = engine::argmax(&decode_logits);

        drop(st); // free VRAM before the fresh state
        let mut st2 = engine::State::new(&model, ctx)?;
        let mut fresh = None;
        let mut pos0 = 0u32;
        for chunk in seq.chunks(st2.max_batch() as usize) {
            fresh = model.forward_batch(&mut st2, chunk, pos0, true)?;
            pos0 += chunk.len() as u32;
        }
        let fresh_logits = fresh.ok_or("no logits")?;
        let fresh_argmax = engine::argmax(&fresh_logits);

        let mut maxd = 0f32;
        let mut sum = 0f64;
        for (a, b) in decode_logits.iter().zip(&fresh_logits) {
            let d = (a - b).abs();
            maxd = maxd.max(d);
            sum += d as f64;
        }
        let gap = {
            let mut top = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for &v in &decode_logits {
                if v > top {
                    second = top;
                    top = v;
                } else if v > second {
                    second = v;
                }
            }
            top - second
        };
        println!(
            "decode-consistency after {} steps ({} total tokens):\n  max |dlogit| {maxd:.4}, mean {:.5}\n  argmax decode={decode_argmax} fresh-prefill={fresh_argmax} ({}), decode top1-top2 gap {gap:.4}",
            nsteps,
            seq.len(),
            sum / decode_logits.len() as f64,
            if decode_argmax == fresh_argmax { "MATCH" } else { "FLIP" },
        );
        return Ok(());
    }

    let t1 = std::time::Instant::now();
    let mut logits = None;
    let mut pos0 = 0u32;
    for chunk in prompt_ids.chunks(st.max_batch() as usize) {
        let last = pos0 as usize + chunk.len() == prompt_ids.len();
        logits = model.forward_batch(&mut st, chunk, pos0, last)?;
        pos0 += chunk.len() as u32;
    }
    eprintln!(
        "pulsar: prefill {} tokens in {:.2}s",
        prompt_ids.len(),
        t1.elapsed().as_secs_f32()
    );

    if let Some(path) = dump_logits {
        let l = logits.as_ref().ok_or("no logits")?;
        let mut s = String::with_capacity(l.len() * 12);
        s.push('[');
        for (i, v) in l.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("{v}"));
        }
        s.push(']');
        std::fs::write(&path, s)?;
        eprintln!("pulsar: wrote {} logits to {path}", l.len());
        return Ok(());
    }

    let mut pos = prompt_ids.len() as u32;
    let mut generated = Vec::new();
    let t2 = std::time::Instant::now();
    for _ in 0..n_predict {
        let l = logits.as_ref().ok_or("no logits")?;
        let next = engine::argmax(l);
        if Some(next) == tok.eos_id {
            break;
        }
        generated.push(next);
        print!("{}", String::from_utf8_lossy(&tok.decode(&[next])));
        use std::io::Write;
        std::io::stdout().flush().ok();
        if pos >= ctx {
            break;
        }
        logits = model.forward_token(&mut st, next, pos, true)?;
        pos += 1;
    }
    println!();
    st.save_warm(&model)?;
    let dt = t2.elapsed().as_secs_f32();
    eprintln!(
        "pulsar: {} tokens in {:.2}s ({:.2} tok/s), vram cache {:.0}% hits, host cache {:.0}% of remainder\npulsar: ids {generated:?}",
        generated.len(),
        dt,
        generated.len() as f32 / dt.max(1e-6),
        100.0 * st.dev_cache.hits as f64 / (st.dev_cache.hits + st.dev_cache.misses).max(1) as f64,
        100.0 * st.store.hits as f64 / (st.store.hits + st.store.misses).max(1) as f64
    );
    Ok(())
}
