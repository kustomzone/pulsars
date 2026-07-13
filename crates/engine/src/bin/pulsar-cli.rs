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
    let t1 = std::time::Instant::now();
    let mut logits = None;
    for (i, &id) in prompt_ids.iter().enumerate() {
        let last = i + 1 == prompt_ids.len();
        logits = model.forward_token(&mut st, id, i as u32, last)?;
    }
    eprintln!(
        "pulsar: prefill {} tokens in {:.2}s",
        prompt_ids.len(),
        t1.elapsed().as_secs_f32()
    );

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
    let dt = t2.elapsed().as_secs_f32();
    eprintln!(
        "pulsar: {} tokens in {:.2}s ({:.2} tok/s)\npulsar: ids {generated:?}",
        generated.len(),
        dt,
        generated.len() as f32 / dt.max(1e-6)
    );
    Ok(())
}
