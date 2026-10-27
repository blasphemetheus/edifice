// AI-generated diagnostic; attribution and poem: ../../README.md.
use std::{collections::HashMap, env, error::Error, process::Command};
use tch::{Device, Kind, Tensor};

fn log_mel(samples: &[f32], filters: &Tensor) -> Tensor {
    let mut padded = vec![0.0f32; 480_000];
    let count = samples.len().min(padded.len());
    padded[..count].copy_from_slice(&samples[..count]);
    let audio = Tensor::from_slice(&padded);
    let window = Tensor::hann_window(400, (Kind::Float, Device::Cpu));
    // Explicit reflection padding matches center=true in OpenAI's STFT.
    // tch's non-optional align_to_window argument requires center=false.
    let centered = audio
        .view([1, 1, -1])
        .reflection_pad1d([200, 200])
        .view([-1]);
    let spectrum = centered.stft_center(
        400,
        160,
        400,
        Some(&window),
        false,
        "reflect",
        false,
        true,
        true,
        false,
    );
    let power = spectrum.narrow(1, 0, 3000).abs().square();
    let log = filters.matmul(&power).clamp_min(1e-10).log10();
    ((&log).maximum(&(log.max() - 8.0)) + 4.0) / 4.0
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        return Err("usage: whisper-audio AUDIO MEL_FILTERS_NPZ OUTPUT_SAFETENSORS".into());
    }
    tch::set_num_threads(4);
    let _guard = tch::no_grad_guard();
    let decoded = Command::new(env::var("FFMPEG").unwrap_or_else(|_| "ffmpeg".into()))
        .args([
            "-nostdin",
            "-v",
            "error",
            "-i",
            &args[1],
            "-t",
            "30.001",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-",
        ])
        .output()?;
    if !decoded.status.success() {
        return Err(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&decoded.stderr)
        )
        .into());
    }
    if decoded.stdout.is_empty() || decoded.stdout.len() % 2 != 0 {
        return Err("Expected nonempty 16-bit PCM audio".into());
    }
    let samples: Vec<_> = decoded
        .stdout
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();
    if samples.len() > 480_000 {
        return Err(
            "This runner accepts at most 30 seconds; split longer recordings explicitly".into(),
        );
    }
    let filters: HashMap<_, _> = Tensor::read_npz(&args[2])?.into_iter().collect();
    let filters = filters.get("mel_80").ok_or("Missing mel_80 filter bank")?;
    if filters.size() != [80, 201] {
        return Err("Expected 80 x 201 mel filters".into());
    }
    let mel = log_mel(&samples, filters).unsqueeze(0);
    let audio = Tensor::from_slice(&samples);
    Tensor::write_safetensors(
        &[
            ("mel_input", &mel),
            ("audio", &audio),
            ("mel_filters", filters),
        ],
        &args[3],
    )?;
    println!(
        "decoded_seconds={} mel_shape={:?} min={} max={}",
        samples.len() as f64 / 16000.0,
        mel.size(),
        f64::try_from(mel.min())?,
        f64::try_from(mel.max())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_has_the_whisper_log_floor() {
        tch::set_num_threads(4);
        let filters = Tensor::ones([80, 201], (Kind::Float, Device::Cpu));
        let mel = log_mel(&[0.0; 16000], &filters);
        assert_eq!(mel.size(), [80, 3000]);
        assert_eq!(f64::try_from(mel.min()).unwrap(), -1.5);
        assert_eq!(f64::try_from(mel.max()).unwrap(), -1.5);
    }
}
