// AI-generated reference; attribution and poem: ../../README.md.
use std::{collections::HashMap, env, error::Error};
use tch::{Kind, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 5 {
        return Err("usage: whisper-decode CHECKPOINT ENCODER CONFIG_TENSORS OUTPUT".into());
    }
    tch::set_num_threads(4);
    let _guard = tch::no_grad_guard();
    let weights: HashMap<_, _> = Tensor::read_safetensors(&args[1])?.into_iter().collect();
    let encoder: HashMap<_, _> = Tensor::read_safetensors(&args[2])?.into_iter().collect();
    let config: HashMap<_, _> = Tensor::read_safetensors(&args[3])?.into_iter().collect();
    let mut tokens = Vec::<i64>::try_from(&config["prompt"])?;
    let suppressed = Vec::<i64>::try_from(&config["suppress"])?;
    let begin = Vec::<i64>::try_from(&config["begin_suppress"])?;
    let initial = tokens.len();
    let w = |name: &str| &weights[&format!("model.decoder.{name}")];
    let linear = |x: &Tensor, name: &str| {
        x.linear(
            w(&format!("{name}.weight")),
            weights.get(&format!("model.decoder.{name}.bias")),
        )
    };
    let norm = |x: &Tensor, name: &str| {
        x.layer_norm(
            [512],
            Some(w(&format!("{name}.weight"))),
            Some(w(&format!("{name}.bias"))),
            1e-5,
            false,
        )
    };
    let split = |x: Tensor| x.view([1, -1, 8, 64]).transpose(1, 2);
    let mut logits_history = Vec::new();
    while tokens.len() < 128 {
        let len = tokens.len() as i64;
        let mut x = w("embed_tokens.weight")
            .index_select(0, &Tensor::from_slice(&tokens))
            .unsqueeze(0)
            + w("embed_positions.weight").narrow(0, 0, len).unsqueeze(0);
        for layer in 0..6 {
            let prefix = format!("layers.{layer}");
            for (attn, norm_name, causal) in [
                ("self_attn", "self_attn_layer_norm", true),
                ("encoder_attn", "encoder_attn_layer_norm", false),
            ] {
                let normalized = norm(&x, &format!("{prefix}.{norm_name}"));
                let memory = if causal {
                    &normalized
                } else {
                    &encoder["enc_final_norm"]
                };
                let q = split(linear(&normalized, &format!("{prefix}.{attn}.q_proj")));
                let k = split(linear(memory, &format!("{prefix}.{attn}.k_proj")));
                let v = split(linear(memory, &format!("{prefix}.{attn}.v_proj")));
                let attended = Tensor::scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    None::<&Tensor>,
                    0.0,
                    causal,
                    None,
                    false,
                )
                .transpose(1, 2)
                .contiguous()
                .view([1, len, 512]);
                x = &x + linear(&attended, &format!("{prefix}.{attn}.out_proj"));
            }
            let normalized = norm(&x, &format!("{prefix}.final_layer_norm"));
            x = &x
                + linear(
                    &linear(&normalized, &format!("{prefix}.fc1")).gelu("none"),
                    &format!("{prefix}.fc2"),
                );
        }
        let logits = norm(&x, "layer_norm")
            .select(1, len - 1)
            .matmul(&w("embed_tokens.weight").transpose(0, 1))
            .squeeze_dim(0);
        logits_history.push(logits.shallow_clone());
        let mut scores = logits.copy();
        let _ = scores.index_fill_(0, &Tensor::from_slice(&suppressed), f64::NEG_INFINITY);
        if tokens.len() == initial {
            let _ = scores.index_fill_(0, &Tensor::from_slice(&begin), f64::NEG_INFINITY);
        }
        let next = i64::try_from(scores.argmax(0, false))?;
        tokens.push(next);
        if next == 50257 {
            break;
        }
    }
    let logits = Tensor::stack(&logits_history, 0).to_kind(Kind::Float);
    Tensor::write_safetensors(
        &[("tokens", Tensor::from_slice(&tokens)), ("logits", logits)],
        &args[4],
    )?;
    println!("tokens={tokens:?}");
    Ok(())
}
