// AI-generated. Attribution, reference revision and limits: docs/mamba2.md.
// Equations adapted from state-spaces/mamba, Copyright (c) 2024 Albert Gu and
// Tri Dao, Apache-2.0. See ../../licenses/mamba-Apache-2.0.txt.
use std::{collections::BTreeMap, env, error::Error};
use tch::{Device, Kind, Tensor};
fn values(shape: &[i64], phase: f64, scale: f64) -> Tensor {
    ((Tensor::arange(shape.iter().product::<i64>(), (Kind::Double, Device::Cpu)) * 0.37 + phase)
        .sin()
        * scale)
        .reshape(shape)
}
fn main() -> Result<(), Box<dyn Error>> {
    let out = env::args()
        .nth(1)
        .ok_or("usage: mamba2-reference OUTPUT.safetensors")?;
    tch::set_num_threads(2);
    let mut saved = Vec::new();
    for (name, length, groups, k, bias, carried) in [
        ("plain", 7, 1, 4, false, false),
        ("grouped", 9, 2, 3, true, true),
        ("short", 1, 2, 4, true, true),
        ("pointwise", 5, 1, 1, false, false),
    ] {
        let (batch, model, inner, heads, width, n) = (2, 4, 8, 4, 2, 3);
        let channels = inner + 2 * groups * n;
        let projection = inner + channels + heads;
        let mut p = BTreeMap::new();
        for (key, shape, phase, scale) in [
            ("x", vec![batch, length, model], 0.1, 0.8),
            ("in_proj.weight", vec![projection, model], 0.3, 0.4),
            ("conv1d.weight", vec![channels, 1, k], 0.8, 0.5),
            ("out_proj.weight", vec![model, inner], 0.6, 0.3),
            ("A_log", vec![heads], 0.4, 0.8),
            ("dt_bias", vec![heads], 0.9, 0.5),
            ("D", vec![heads], 0.2, 0.7),
            ("norm.weight", vec![inner], 0.8, 1.2),
            (
                "h0",
                vec![batch, heads, width, n],
                0.7,
                if carried { 0.3 } else { 0.0 },
            ),
            (
                "conv0",
                vec![batch, (k - 1).max(1), channels],
                0.5,
                if carried { 0.4 } else { 0.0 },
            ),
        ] {
            p.insert(key, values(&shape, phase, scale).set_requires_grad(true));
        }
        if bias {
            for (key, size) in [
                ("in_proj.bias", projection),
                ("conv1d.bias", channels),
                ("out_proj.bias", model),
            ] {
                p.insert(key, values(&[size], 0.5, 0.2).set_requires_grad(true));
            }
        }
        let mut projected = p["x"].matmul(&p["in_proj.weight"].transpose(0, 1));
        if bias {
            projected += &p["in_proj.bias"];
        }
        let z = projected.narrow(-1, 0, inner);
        let xc = projected.narrow(-1, inner, channels);
        let dt = (projected.narrow(-1, inner + channels, heads) + &p["dt_bias"]).softplus();
        let history = if k == 1 {
            xc
        } else {
            Tensor::cat(&[&p["conv0"], &xc], 1)
        };
        let conv_final = if k == 1 {
            p["conv0"].shallow_clone()
        } else {
            history.narrow(1, length, k - 1)
        };
        let activated = history
            .transpose(1, 2)
            .conv1d(&p["conv1d.weight"], p.get("conv1d.bias"), 1, 0, 1, channels)
            .transpose(1, 2)
            .silu();
        let x = activated
            .narrow(-1, 0, inner)
            .reshape([batch, length, heads, width]);
        let indices =
            Tensor::from_slice(&(0..heads).map(|h| h / (heads / groups)).collect::<Vec<_>>());
        let b = activated
            .narrow(-1, inner, groups * n)
            .reshape([batch, length, groups, n])
            .index_select(2, &indices);
        let c = activated
            .narrow(-1, inner + groups * n, groups * n)
            .reshape([batch, length, groups, n])
            .index_select(2, &indices);
        let a = -p["A_log"].exp();
        let mut h = p["h0"].shallow_clone();
        let mut ys = Vec::new();
        for t in 0..length {
            let delta = dt.select(1, t);
            let xt = x.select(1, t);
            h = (&delta * &a).exp().unsqueeze(-1).unsqueeze(-1) * h
                + (&delta.unsqueeze(-1) * &xt).unsqueeze(-1) * b.select(1, t).unsqueeze(2);
            ys.push(
                (&h * c.select(1, t).unsqueeze(2)).sum_dim_intlist(
                    [-1].as_slice(),
                    false,
                    Kind::Double,
                ) + &xt * p["D"].reshape([1, heads, 1]),
            );
        }
        let gated = Tensor::stack(&ys, 1).reshape([batch, length, inner]) * z.silu();
        let grouped = gated.reshape([batch, length, groups, inner / groups]);
        let normalized = &grouped
            / ((&grouped * &grouped).mean_dim([-1].as_slice(), true, Kind::Double) + 1e-5).sqrt();
        let mut y = (normalized.reshape([batch, length, inner]) * &p["norm.weight"])
            .matmul(&p["out_proj.weight"].transpose(0, 1));
        if bias {
            y += &p["out_proj.bias"];
        }
        let weights = values(&y.size(), 0.4, 0.7);
        let loss = (&y * &weights).sum(Kind::Double);
        // k=1 ignores the dummy convolution cache; exclude that unused input.
        let differentiable: Vec<_> = p
            .iter()
            .filter(|(key, _)| !(k == 1 && **key == "conv0"))
            .collect();
        let grads = Tensor::run_backward(
            &[loss],
            &differentiable.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
            false,
            false,
        );
        for ((key, _), grad) in differentiable.iter().zip(grads) {
            saved.push((format!("{name}.grad.{key}"), grad));
        }
        for (key, value) in p {
            saved.push((format!("{name}.{key}"), value.detach()));
        }
        for (key, value) in [
            ("output", y),
            ("h", h),
            ("conv", conv_final),
            ("loss_weights", weights),
        ] {
            saved.push((format!("{name}.{key}"), value.detach()));
        }
    }
    let saved: Vec<_> = saved
        .into_iter()
        .map(|(k, v)| (k, v.contiguous()))
        .collect();
    Tensor::write_safetensors(&saved, out)?;
    Ok(())
}
