use std::{collections::HashMap, env, error::Error};
use tch::{Kind, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 5 {
        return Err("usage: whisper-reference CHECKPOINT FIXTURE OUTPUT eager|sdpa".into());
    }
    tch::set_num_threads(4);
    let _guard = tch::no_grad_guard();
    let weights: HashMap<_, _> = Tensor::read_safetensors(&args[1])?.into_iter().collect();
    let fixture: HashMap<_, _> = Tensor::read_safetensors(&args[2])?.into_iter().collect();
    let w = |name: &str| &weights[&format!("model.encoder.{name}")];
    let linear = |x: &Tensor, name: &str| {
        let bias = weights.get(&format!("model.encoder.{name}.bias"));
        x.linear(w(&format!("{name}.weight")), bias)
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
    let mut saved: Vec<(String, Tensor)> = Vec::new();
    let mut x = fixture["mel_input"].shallow_clone();
    for (name, stride) in [("conv1", 1), ("conv2", 2)] {
        x = x
            .conv1d(
                w(&format!("{name}.weight")),
                Some(w(&format!("{name}.bias"))),
                stride,
                1,
                1,
                1,
            )
            .gelu("none");
        saved.push((format!("enc_{name}_act"), x.transpose(1, 2).contiguous()));
    }
    x = x.transpose(1, 2) + w("embed_positions.weight");
    saved.push(("enc_pos_embed".into(), x.shallow_clone()));
    for layer in 0..6 {
        let prefix = format!("layers.{layer}");
        let name = format!("enc_block_{}", layer + 1);
        let normalized = norm(&x, &format!("{prefix}.self_attn_layer_norm"));
        saved.push((format!("{name}_attn_norm"), normalized.shallow_clone()));
        let project = |part: &str| {
            linear(&normalized, &format!("{prefix}.self_attn.{part}_proj"))
                .view([1, 1500, 8, 64])
                .transpose(1, 2)
        };
        let (q, k, v) = (project("q"), project("k"), project("v"));
        let attended = match args[4].as_str() {
            "eager" => ((q / 8.0).matmul(&k.transpose(-2, -1)))
                .softmax(-1, Kind::Float)
                .matmul(&v),
            "sdpa" => Tensor::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                None::<&Tensor>,
                0.0,
                false,
                None,
                false,
            ),
            _ => return Err("attention must be eager or sdpa".into()),
        };
        let attended = attended.transpose(1, 2).contiguous().view([1, 1500, 512]);
        let projected = linear(&attended, &format!("{prefix}.self_attn.out_proj"));
        saved.push((format!("{name}_attn_out"), projected.shallow_clone()));
        x = &x + projected;
        saved.push((format!("{name}_attn_residual"), x.shallow_clone()));
        let normalized = norm(&x, &format!("{prefix}.final_layer_norm"));
        saved.push((format!("{name}_ffn_norm"), normalized.shallow_clone()));
        let up = linear(&normalized, &format!("{prefix}.fc1"));
        saved.push((format!("{name}_ffn_up"), up.shallow_clone()));
        let hidden = up.gelu("none");
        saved.push((format!("{name}_ffn_act"), hidden.shallow_clone()));
        let down = linear(&hidden, &format!("{prefix}.fc2"));
        saved.push((format!("{name}_ffn_down"), down.shallow_clone()));
        x = &x + down;
        saved.push((format!("{name}_ffn_residual"), x.shallow_clone()));
    }
    x = norm(&x, "layer_norm");
    let error = (&x - &fixture["expected_encoder_output"]).abs();
    println!(
        "mode={} max_abs={} mean_abs={}",
        args[4],
        f64::try_from(error.max())?,
        f64::try_from(error.mean(Kind::Float))?
    );
    saved.push(("enc_final_norm".into(), x));
    let saved: Vec<_> = saved
        .into_iter()
        .map(|(name, tensor)| (name, tensor.contiguous()))
        .collect();
    Tensor::write_safetensors(&saved, &args[3])?;
    Ok(())
}
