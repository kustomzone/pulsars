//! Qwen3.5/3.6 MoE hybrid (qwen35moe) forward path, task #21.
//!
//! Reference: llama.cpp qwen35moe.cpp + delta-net-base.cpp (see
//! docs/qwen36-port-notes.md). 3 of 4 layers run Gated DeltaNet linear
//! attention (depthwise conv window + per-head delta-rule state, O(1)
//! memory); every 4th layer is sigmoid-gated full attention with
//! partial neox rope (M-RoPE reduces to plain neox for text-only:
//! all three position ids are equal). MoE on every layer: softmax
//! top-8 of 256 plus a shared expert behind a scalar sigmoid gate.
//! Decode-only graph: prefill loops tokens (conv window + delta state
//! are sequential). ponytail: chunked GDN prefill + DFlash spec decode
//! (task #23) are the perf pass.

use super::{Attn, Ffn, LayerW, Model, Result, State};
use kernels::DeviceBuf;

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Per-GDN-layer device state.
struct GdnState {
    /// delta-rule state [ssm_v_heads][ssm_state][ssm_state]
    s: DeviceBuf,
    /// conv window [ssm_conv_k - 1][conv_dim]
    conv: DeviceBuf,
}

/// qwen35 runtime: GDN states + scratch (own buffers - the shared
/// State scratch is sized for the attention geometry, not conv_dim).
pub(super) struct Qwen35Rt {
    states: Vec<Option<GdnState>>,
    qkv: DeviceBuf,      // [conv_dim] raw projection
    conv_out: DeviceBuf, // [conv_dim] conv+silu, layout [q|k|v]
    z: DeviceBuf,        // [value_dim]
    gq: DeviceBuf,       // [key_dim] delta-rule inputs (conv_out slices)
    gk: DeviceBuf,       // [key_dim]
    gv: DeviceBuf,       // [value_dim]
    small: DeviceBuf,    // [ssm_v_heads] alpha/beta matvec scratch
    g: DeviceBuf,        // [ssm_v_heads] log-decay upload
    beta: DeviceBuf,     // [ssm_v_heads]
    gdn_o: DeviceBuf,    // [value_dim] delta output
    gdn_tmp: DeviceBuf,  // [value_dim] gated-norm result
    qfull: DeviceBuf,    // [2*n_head*head_dim] fused q+gate (attn layers)
    gate: DeviceBuf,     // [n_head*head_dim]
    shg: DeviceBuf,      // [1] shared-expert gate logit
}

impl Qwen35Rt {
    pub fn new(m: &Model) -> Result<Qwen35Rt> {
        let s = m.shape;
        let key_dim = (s.ssm_k_heads * s.ssm_state) as usize;
        let value_dim = (s.ssm_v_heads * s.ssm_state) as usize;
        let conv_dim = 2 * key_dim + value_dim;
        let mut states = Vec::with_capacity(s.n_exec_layer as usize);
        for il in 0..s.n_exec_layer {
            if (il + 1) % s.full_attn_interval == 0 {
                states.push(None);
            } else {
                let sbytes = s.ssm_v_heads as usize
                    * s.ssm_state as usize
                    * s.ssm_state as usize
                    * 4;
                let cbytes = (s.ssm_conv_k as usize - 1) * conv_dim * 4;
                let mut st = GdnState {
                    s: DeviceBuf::alloc(sbytes)?,
                    conv: DeviceBuf::alloc(cbytes)?,
                };
                kernels::zero(&mut st.s, sbytes)?;
                kernels::zero(&mut st.conv, cbytes)?;
                states.push(Some(st));
            }
        }
        let f32s = |n: usize| DeviceBuf::alloc(n * 4);
        Ok(Qwen35Rt {
            states,
            qkv: f32s(conv_dim)?,
            conv_out: f32s(conv_dim)?,
            z: f32s(value_dim)?,
            gq: f32s(key_dim)?,
            gk: f32s(key_dim)?,
            gv: f32s(value_dim)?,
            small: f32s(s.ssm_v_heads as usize)?,
            g: f32s(s.ssm_v_heads as usize)?,
            beta: f32s(s.ssm_v_heads as usize)?,
            gdn_o: f32s(value_dim)?,
            gdn_tmp: f32s(value_dim)?,
            qfull: f32s(2 * (s.n_head * s.head_dim) as usize)?,
            gate: f32s((s.n_head * s.head_dim) as usize)?,
            shg: f32s(1)?,
        })
    }

    fn reset(&mut self) -> Result {
        for st in self.states.iter_mut().flatten() {
            let (sb, cb) = (st.s.bytes(), st.conv.bytes());
            kernels::zero(&mut st.s, sb)?;
            kernels::zero(&mut st.conv, cb)?;
        }
        Ok(())
    }
}

impl Model {
    pub(super) fn forward_qwen35(&self, st: &mut State, tokens: &[u32], pos0: u32, rows: u32) -> Result<Option<Vec<f32>>> {
        if tokens.is_empty() {
            return Err("empty batch".into());
        }
        if rows > 1 {
            return Err("qwen35: multi-row logits not supported yet".into());
        }
        if pos0 + tokens.len() as u32 > st.ctx {
            return Err("position exceeds context".into());
        }
        let mut rt = st.qwen35.take().ok_or("qwen35 state missing")?;
        let r = self.forward_qwen35_inner(st, &mut rt, tokens, pos0, rows);
        st.qwen35 = Some(rt);
        r
    }

    fn forward_qwen35_inner(&self, st: &mut State, rt: &mut Qwen35Rt, tokens: &[u32], pos0: u32, rows: u32) -> Result<Option<Vec<f32>>> {
        let s = self.shape;
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = pos0 + i as u32;
            if pos == 0 {
                rt.reset()?;
            }
            st.tok.write(0, kernels::as_bytes(&[tok as i32]))?;
            kernels::embed_q8_0(&mut st.cur, &self.token_embd, &st.tok, s.n_embd, s.n_vocab, 1)?;
            for (il, l) in self.layers.iter().enumerate() {
                self.eval_qwen35_layer(st, rt, il, l, pos)?;
            }
        }
        if rows == 0 {
            return Ok(None);
        }
        kernels::rms_norm(&mut st.normed, &st.cur, &self.output_norm, s.n_embd, 1, s.rms_eps)?;
        self.head_logits(st, 1)?;
        kernels::sync()?;
        Ok(Some(st.logits.read_f32(s.n_vocab as usize)?))
    }

    fn eval_qwen35_layer(&self, st: &mut State, rt: &mut Qwen35Rt, il: usize, l: &LayerW, pos: u32) -> Result {
        let s = self.shape;
        let eps = s.rms_eps;
        let Attn::Qwen35(w) = &l.attn else {
            return Err("qwen35 layer without Qwen35 attn weights".into());
        };
        let key_dim = s.ssm_k_heads * s.ssm_state;
        let value_dim = s.ssm_v_heads * s.ssm_state;
        let conv_dim = 2 * key_dim + value_dim;

        kernels::rms_norm(&mut st.normed, &st.cur, &l.attn_norm, s.n_embd, 1, eps)?;

        if let Some(gdn) = &w.gdn {
            // ---- Gated DeltaNet
            kernels::matmul_q8_0(&mut rt.qkv, &gdn.wqkv, &st.normed, s.n_embd, conv_dim, 1)?;
            kernels::matmul_q8_0(&mut rt.z, &gdn.wz, &st.normed, s.n_embd, value_dim, 1)?;
            // decay + mixing coefficients (32-wide: host math is exact
            // and cheap; reference: g = a * softplus(alpha + dt_bias))
            kernels::matmul_f32(&mut rt.small, &gdn.alpha_w, &st.normed, s.n_embd, s.ssm_v_heads, 1)?;
            kernels::sync()?;
            let alpha = rt.small.read_f32(s.ssm_v_heads as usize)?;
            let g: Vec<f32> = alpha
                .iter()
                .zip(&gdn.a)
                .zip(&gdn.dt_bias)
                .map(|((&al, &a), &dt)| a * softplus(al + dt))
                .collect();
            rt.g.write(0, kernels::as_bytes(&g))?;
            kernels::matmul_f32(&mut rt.small, &gdn.beta_w, &st.normed, s.n_embd, s.ssm_v_heads, 1)?;
            kernels::sync()?;
            let beta: Vec<f32> =
                rt.small.read_f32(s.ssm_v_heads as usize)?.iter().map(|&b| sigmoid(b)).collect();
            rt.beta.write(0, kernels::as_bytes(&beta))?;

            let gs = rt.states[il].as_mut().ok_or("gdn state missing")?;
            kernels::qwen35_conv_step(&mut rt.conv_out, &rt.qkv, &gdn.conv, &mut gs.conv, conv_dim, s.ssm_conv_k)?;
            // conv_out = [q key_dim | k key_dim | v value_dim]; l2-norm
            // exactly the q and k head rows in place (2*key_dim/state
            // rows of ssm_state), leaving v untouched
            kernels::qwen35_l2_norm(&mut rt.conv_out, 2 * s.ssm_k_heads, s.ssm_state, eps)?;
            self.qwen35_gdn(rt, il, key_dim, value_dim)?;
            // gated rms norm per v-head (weight shared across heads),
            // then * silu(z) - the swiglu kernel IS silu(gate)*up
            kernels::gqa_head_rms_norm(&mut rt.gdn_o, Some(&gdn.ssm_norm), s.ssm_v_heads, s.ssm_state, eps)?;
            kernels::swiglu(&mut rt.gdn_tmp, &rt.z, &rt.gdn_o, value_dim, 0.0, 1.0, 0)?;
            kernels::matmul_q8_0(&mut st.attn_out, &gdn.ssm_out, &rt.gdn_tmp, value_dim, s.n_embd, 1)?;
        } else if let Some(attn) = &w.attn {
            // ---- sigmoid-gated full attention (partial neox rope)
            let hd = s.head_dim;
            kernels::matmul_q8_0(&mut rt.qfull, &attn.wq, &st.normed, s.n_embd, 2 * s.n_head * hd, 1)?;
            kernels::qwen35_split_gate(&mut st.q, &mut rt.gate, &rt.qfull, s.n_head, hd)?;
            kernels::matmul_q8_0(&mut st.k, &attn.wk, &st.normed, s.n_embd, s.n_head_kv * hd, 1)?;
            kernels::matmul_q8_0(&mut st.v, &attn.wv, &st.normed, s.n_embd, s.n_head_kv * hd, 1)?;
            kernels::gqa_head_rms_norm(&mut st.q, Some(&attn.q_norm), s.n_head, hd, eps)?;
            kernels::gqa_head_rms_norm(&mut st.k, Some(&attn.k_norm), s.n_head_kv, hd, eps)?;
            kernels::gqa_rope(&mut st.q, 1, s.n_head, hd, s.rot_dim, pos, s.rope_freq_base, None)?;
            kernels::gqa_rope(&mut st.k, 1, s.n_head_kv, hd, s.rot_dim, pos, s.rope_freq_base, None)?;
            kernels::gqa_kv_append(&mut st.kcache[il], &st.k, 1, s.n_head_kv, hd, st.ctx, pos, 0)?;
            kernels::gqa_kv_append(&mut st.vcache[il], &st.v, 1, s.n_head_kv, hd, st.ctx, pos, 0)?;
            kernels::gqa_attention_rel(
                &mut st.heads, &st.q, &st.kcache[il], &st.vcache[il],
                1, s.n_head, s.n_head_kv, hd, st.ctx, pos,
                1.0 / (hd as f32).sqrt(), 0, None, 0, 0,
            )?;
            kernels::qwen35_sigmoid_gate(&mut st.heads, &rt.gate, s.n_head * hd)?;
            kernels::matmul_q8_0(&mut st.attn_out, &l.attn_output, &st.heads, s.n_head * hd, s.n_embd, 1)?;
        } else {
            return Err("qwen35 layer with neither attn nor gdn".into());
        }
        kernels::add(&mut st.after_attn, &st.cur, &st.attn_out, s.n_embd)?;

        // ---- MoE (pre-norm residual: FFN adds onto after_attn)
        kernels::rms_norm(&mut st.normed, &st.after_attn, &l.ffn_norm, s.n_embd, 1, eps)?;
        let Ffn::Moe { gate_inp, shexp, gate_exps, up_exps, down_exps, .. } = &l.ffn else {
            return Err("qwen35 layer without MoE ffn".into());
        };
        kernels::matmul_f32(&mut st.router_logits, gate_inp, &st.normed, s.n_embd, s.n_expert, 1)?;
        // softmax top-k with renorm (qwen3moe mode); zero bias buffer
        // lives in Ffn::Moe already via the loader fallback
        let Ffn::Moe { probs_b, .. } = &l.ffn else { unreachable!() };
        kernels::router_select(
            &mut st.router_selected,
            &mut st.router_weights,
            &st.router_logits,
            probs_b,
            s.n_expert,
            s.n_expert_used,
            s.expert_weight_scale,
            1,
            1, // softmax mode
            0,
        )?;
        // shared expert + its scalar sigmoid gate
        if let Some((sg, su, sd)) = shexp {
            kernels::matmul_q8_0(&mut st.gate_act, sg, &st.normed, s.n_embd, s.n_ff_exp, 1)?;
            kernels::matmul_q8_0(&mut st.up_act, su, &st.normed, s.n_embd, s.n_ff_exp, 1)?;
            kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, s.n_ff_exp, 0.0, 1.0, 0)?;
            kernels::matmul_q8_0(&mut st.shared_out, sd, &st.ffn_mid, s.n_ff_exp, s.n_embd, 1)?;
            kernels::matmul_f32(&mut rt.shg, &w.shexp_gate, &st.normed, s.n_embd, 1, 1)?;
            kernels::sync()?;
            let gl = rt.shg.read_f32(1)?[0];
            kernels::scale(&mut st.shared_out, s.n_embd, sigmoid(gl))?;
        } else {
            kernels::zero(&mut st.shared_out, s.n_embd as usize * 4)?;
        }
        kernels::quantize_q8_k(&mut st.xq, &st.normed, s.n_embd, 1)?;
        kernels::sync()?;
        let selected = st.router_selected.read_i32(s.n_expert_used as usize)?;
        self.dsv4_moe(st, &selected, gate_exps, up_exps, down_exps, 0)?;
        kernels::add(&mut st.ffn_out, &st.moe_out, &st.shared_out, s.n_embd)?;
        kernels::add(&mut st.cur, &st.after_attn, &st.ffn_out, s.n_embd)?;
        Ok(())
    }

    /// Split conv_out into q/k/v scratch and run the delta-rule step.
    fn qwen35_gdn(&self, rt: &mut Qwen35Rt, il: usize, key_dim: u32, value_dim: u32) -> Result {
        let s = self.shape;
        let kd = key_dim as usize * 4;
        let vd = value_dim as usize * 4;
        kernels::copy_d2d(&mut rt.gq, 0, &rt.conv_out, 0, kd)?;
        kernels::copy_d2d(&mut rt.gk, 0, &rt.conv_out, kd, kd)?;
        kernels::copy_d2d(&mut rt.gv, 0, &rt.conv_out, 2 * kd, vd)?;
        let gs = rt.states[il].as_mut().ok_or("gdn state missing")?;
        kernels::qwen35_gdn_step(
            &mut rt.gdn_o, &mut gs.s, &rt.gq, &rt.gk, &rt.gv, &rt.g, &rt.beta,
            s.ssm_v_heads, s.ssm_k_heads, s.ssm_state,
        )?;
        Ok(())
    }
}
