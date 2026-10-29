// AI-generated. Provenance, scope, and poem: docs/recurrent_reference_validation.md.
// SSD port: Copyright (c) 2024 Albert Gu and Tri Dao, Apache-2.0.
// Positive MinGRU equation: Copyright (c) 2024 Christoph Heindl, MIT.
// License texts are in ../../licenses/ relative to this source file.
use std::{env, error::Error};
use tch::{Device, Kind, Tensor};

fn values(shape: &[i64], phase: f64, scale: f64) -> Tensor {
    ((Tensor::arange(shape.iter().product::<i64>(), (Kind::Double, Device::Cpu)) * 0.37 + phase)
        .sin()
        * scale)
        .reshape(shape)
}

fn save_case(
    saved: &mut Vec<(String, Tensor)>,
    name: &str,
    inputs: &[(&str, &Tensor)],
    y: &Tensor,
    state: &Tensor,
) {
    let weights = values(&y.size(), 0.4, 0.7);
    let loss = (y * &weights).sum(Kind::Double);
    let tensors: Vec<_> = inputs.iter().map(|(_, t)| *t).collect();
    let gradients = Tensor::run_backward(&[loss], &tensors, false, false);
    for ((key, input), grad) in inputs.iter().zip(gradients) {
        saved.push((format!("{name}.{key}"), input.detach()));
        saved.push((format!("{name}.grad.{key}"), grad));
    }
    saved.push((format!("{name}.output"), y.detach()));
    saved.push((format!("{name}.state"), state.detach()));
    saved.push((format!("{name}.loss_weights"), weights));
}

// Stable segment sums from state-spaces/mamba's ssd_minimal.py.
fn segsum(x: &Tensor) -> Tensor {
    let t = *x.size().last().unwrap();
    let lower = Tensor::ones([t, t], (Kind::Bool, Device::Cpu)).tril(-1);
    let expanded = x.unsqueeze(-1).expand([-1, -1, -1, t, t], true);
    let sum = expanded
        .masked_fill(&lower.logical_not(), 0.0)
        .cumsum(-2, Kind::Double);
    sum.masked_fill(
        &Tensor::ones([t, t], (Kind::Bool, Device::Cpu))
            .tril(0)
            .logical_not(),
        f64::NEG_INFINITY,
    )
}

// Literal four-stage SSD decomposition, including inter-chunk state propagation.
fn ssd(x: &Tensor, a: &Tensor, b: &Tensor, c: &Tensor, q: i64) -> (Tensor, Tensor) {
    let (batch, length, heads, width) = (x.size()[0], x.size()[1], x.size()[2], x.size()[3]);
    let chunks = length / q;
    let x = x.reshape([batch, chunks, q, heads, width]);
    let b = b.reshape([batch, chunks, q, heads, -1]);
    let c = c.reshape([batch, chunks, q, heads, -1]);
    let a = a.reshape([batch, chunks, q, heads]).permute([0, 3, 1, 2]);
    let cumulative = a.cumsum(-1, Kind::Double);
    let l = segsum(&a).exp();
    let diag = Tensor::einsum(
        "bclhn,bcshn,bhcls,bcshp->bclhp",
        &[&c, &b, &l, &x],
        None::<&[i64]>,
    );
    let decay = (cumulative.narrow(-1, q - 1, 1) - &cumulative).exp();
    let states = Tensor::einsum("bclhn,bhcl,bclhp->bchpn", &[&b, &decay, &x], None::<&[i64]>);
    let states = Tensor::cat(&[Tensor::zeros_like(&states.narrow(1, 0, 1)), states], 1);
    let ends = cumulative
        .select(-1, q - 1)
        .constant_pad_nd([1, 0])
        .unsqueeze(2);
    let chunk_decay = segsum(&ends).squeeze_dim(2).exp();
    let new_states = Tensor::einsum(
        "bhzc,bchpn->bzhpn",
        &[&chunk_decay, &states],
        None::<&[i64]>,
    );
    let state = new_states.select(1, chunks);
    let off = Tensor::einsum(
        "bclhn,bchpn,bhcl->bclhp",
        &[&c, &new_states.narrow(1, 0, chunks), &cumulative.exp()],
        None::<&[i64]>,
    );
    ((diag + off).reshape([batch, length, heads, width]), state)
}

fn main() -> Result<(), Box<dyn Error>> {
    let out = env::args()
        .nth(1)
        .ok_or("usage: recurrent-reference OUTPUT.safetensors")?;
    tch::set_num_threads(2);
    let mut saved = Vec::new();
    // Actual ATen GRU, not a handwritten reimplementation of Edifice's cell.
    for (name, batch, length, hidden) in [("gru_short", 1, 1, 3), ("gru", 2, 9, 4)] {
        let x = values(&[batch, length, 3], 0.1, 0.8).set_requires_grad(true);
        let h = values(&[1, batch, hidden], 0.7, 0.3).set_requires_grad(true);
        let wi = values(&[3 * hidden, 3], 0.3, 0.4).set_requires_grad(true);
        let wh = values(&[3 * hidden, hidden], 0.8, 0.35).set_requires_grad(true);
        let bi = values(&[3 * hidden], 0.2, 0.25).set_requires_grad(true);
        let bh = values(&[3 * hidden], 0.9, 0.2).set_requires_grad(true);
        let (y, state) = x.gru(&h, &[&wi, &wh, &bi, &bh], true, 1, 0.0, false, false, true);
        save_case(
            &mut saved,
            name,
            &[
                ("x", &x),
                ("h0", &h),
                ("wi", &wi),
                ("wh", &wh),
                ("bi", &bi),
                ("bh", &bh),
            ],
            &y,
            &state,
        );
    }
    // Real-valued selective_scan_ref equations. The caller explicitly applies
    // Edifice's dt clamp; upstream itself does not impose this clamp.
    for (name, fixed, initial, length) in [
        ("mamba", false, false, 9),
        ("mamba_fixed", true, true, 9),
        ("mamba_long", true, false, 37),
    ] {
        let x = values(&[2, length, 3], 0.3, 0.8).set_requires_grad(true);
        let dt = (values(&[2, length, 3], 0.5, 0.2) + 0.07).set_requires_grad(true);
        let a = if fixed {
            -Tensor::arange_start(1, 4, (Kind::Double, Device::Cpu)).repeat([3, 1])
        } else {
            -(values(&[3, 3], 0.2, 1.0) + 1.5)
        }
        .set_requires_grad(true);
        let b = values(&[2, length, 3], 0.6, 0.7).set_requires_grad(true);
        let c = values(&[2, length, 3], 1.2, 0.9).set_requires_grad(true);
        let h0 = (values(&[2, 3, 3], 0.4, if initial { 0.3 } else { 0.0 })).set_requires_grad(true);
        let mut h = h0.shallow_clone();
        let mut ys = Vec::new();
        for t in 0..length {
            let delta = dt.select(1, t).clamp(0.001, 0.1).unsqueeze(-1);
            h = (&delta * &a).exp() * h
                + &delta * b.select(1, t).unsqueeze(1) * x.select(1, t).unsqueeze(-1);
            ys.push((&h * c.select(1, t).unsqueeze(1)).sum_dim_intlist(
                [-1].as_slice(),
                false,
                Kind::Double,
            ));
        }
        save_case(
            &mut saved,
            name,
            &[
                ("x", &x),
                ("dt", &dt),
                ("a", &a),
                ("b", &b),
                ("c", &c),
                ("h0", &h0),
            ],
            &Tensor::stack(&ys, 1),
            &h,
        );
    }
    // Lift each (channel,state) diagonal coordinate to a scalar SSD head.
    // This validates Edifice's SSD algorithm, NOT canonical Mamba-2 topology.
    let x = values(&[2, 12, 3], 0.2, 0.8).set_requires_grad(true);
    let dt = (values(&[2, 12, 3], 0.8, 0.2) + 0.07).set_requires_grad(true);
    let b = values(&[2, 12, 2], 0.6, 0.7).set_requires_grad(true);
    let delta = dt.clamp(0.001, 0.1).unsqueeze(-1);
    let a = -Tensor::arange_start(1, 3, (Kind::Double, Device::Cpu));
    let log_decay = (&delta * a).reshape([2, 12, 6]);
    let bx = (&delta * b.unsqueeze(2) * x.unsqueeze(-1)).reshape([2, 12, 6, 1]);
    let ones = Tensor::ones([2, 12, 6, 1], (Kind::Double, Device::Cpu));
    let (y, state) = ssd(&bx, &log_decay, &ones, &ones, 3);
    save_case(
        &mut saved,
        "ssd",
        &[("x", &x), ("dt", &dt), ("b", &b)],
        &y.reshape([2, 12, 3, 2]),
        &state.reshape([2, 3, 2]),
    );
    // Independent closed-form affine recurrence for signed MinGRU. Also save
    // the positive-candidate convention used by cheind/mingru for contrast.
    let gates = values(&[2, 9, 4], 0.3, 12.0).set_requires_grad(true);
    let candidates = values(&[2, 9, 4], 0.8, 2.0).set_requires_grad(true);
    let z = gates.sigmoid();
    let log_a = (-&gates).log_sigmoid().transpose(1, 2).unsqueeze(2);
    let l = segsum(&log_a).squeeze_dim(2).exp();
    let y = l
        .matmul(&(&z * &candidates).transpose(1, 2).unsqueeze(-1))
        .squeeze_dim(-1)
        .transpose(1, 2);
    save_case(
        &mut saved,
        "mingru",
        &[("gates", &gates), ("candidates", &candidates)],
        &y,
        &y.select(1, 8),
    );
    let positive = (&candidates + 0.5).where_self(&candidates.ge(0.0), &candidates.sigmoid());
    let positive_y = l
        .matmul(&(&z * positive).transpose(1, 2).unsqueeze(-1))
        .squeeze_dim(-1)
        .transpose(1, 2);
    saved.push(("mingru.positive_output".into(), positive_y.detach()));
    let saved: Vec<_> = saved
        .into_iter()
        .map(|(name, tensor)| (name, tensor.contiguous()))
        .collect();
    Tensor::write_safetensors(&saved, out)?;
    println!(
        "Wrote {} reference tensors using LibTorch {}",
        saved.len(),
        "2.10.0"
    );
    Ok(())
}
