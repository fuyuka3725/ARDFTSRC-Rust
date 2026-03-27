use hound::{WavReader, WavWriter, WavSpec, SampleFormat};
use num_complex::Complex64;
use rustfft::{FftPlanner, num_complex::Complex};
use std::env;

// ─────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ─────────────────────────────────────────────
// Audio I/O helpers
// ─────────────────────────────────────────────

/// Interleaved f64 samples, shape [num_samples][num_channels].
struct AudioData {
    samples: Vec<Vec<f64>>, // [channel][sample]
    sample_rate: u32,
    bit_depth: u16,
    num_channels: usize,
    num_samples: usize,
}

fn load_wav(path: &str) -> Result<AudioData> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let num_channels = spec.channels as usize;

    let raw: Vec<f64> = match spec.sample_format {
        SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f64;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f64 / max))
                .collect::<std::result::Result<_, _>>()?
        }
        SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map(|v| v as f64))
            .collect::<std::result::Result<_, _>>()?,
    };

    let total_samples = raw.len() / num_channels;
    let mut channels: Vec<Vec<f64>> = vec![Vec::with_capacity(total_samples); num_channels];
    for (i, sample) in raw.into_iter().enumerate() {
        channels[i % num_channels].push(sample);
    }

    Ok(AudioData {
        num_samples: total_samples,
        num_channels,
        sample_rate: spec.sample_rate,
        bit_depth: spec.bits_per_sample,
        samples: channels,
    })
}

fn save_wav(
    path: &str,
    channels: &[Vec<f64>],
    sample_rate: u32,
    bit_depth: u16,
) -> Result<()> {
    let num_channels = channels.len();
    let num_samples = channels[0].len();

    let spec = WavSpec {
        channels: num_channels as u16,
        sample_rate,
        bits_per_sample: bit_depth,
        sample_format: if bit_depth == 32 {
            SampleFormat::Float
        } else {
            SampleFormat::Int
        },
    };

    let mut writer = WavWriter::create(path, spec)?;

    match (spec.sample_format, bit_depth) {
        (SampleFormat::Float, 32) => {
            for s in 0..num_samples {
                for ch in 0..num_channels {
                    let v = channels[ch][s].clamp(-1.0, 1.0) as f32;
                    writer.write_sample(v)?;
                }
            }
        }
        (SampleFormat::Int, 16) => {
            let max = i16::MAX as f64; // 32767
            for s in 0..num_samples {
                for ch in 0..num_channels {
                    let v = channels[ch][s].clamp(-1.0, 1.0);
                    writer.write_sample((v * max).round() as i16)?;
                }
            }
        }
        (SampleFormat::Int, 24) => {
            let max = (1i32 << 23) as f64;
            for s in 0..num_samples {
                for ch in 0..num_channels {
                    let v = channels[ch][s].clamp(-1.0, 1.0);
                    writer.write_sample((v * max).round() as i32)?;
                }
            }
        }
        _ => {
            let max = (1i64 << (bit_depth - 1)) as f64;
            for s in 0..num_samples {
                for ch in 0..num_channels {
                    let v = channels[ch][s].clamp(-1.0, 1.0);
                    writer.write_sample((v * max).round() as i32)?;
                }
            }
        }
    }

    writer.finalize()?;
    Ok(())
}

// ─────────────────────────────────────────────
// Sigmoid taper
//
// Special bipolar-kernel sigmoid (NOT standard logistic):
//   zbk = t / ((t - n) - 1)  -  t / (n + 1)
//   v   = 1 / (exp(zbk) + 1)
//
// Properties:
//   - v ≈ 1 for n near 0  (flat passband)
//   - v ≈ 0 for n near t  (steep rolloff at bandwidth edge)
//   - C∞ smooth, zero Gibbs ringing compared to brick-wall
// ─────────────────────────────────────────────
fn build_taper(
    in_rdft_size: usize,
    tr_nb_samples: usize,
    taper_samples: usize,
    taper_size: usize, // in_rdft_size/2 + 1
) -> Vec<Complex64> {
    (0..taper_size)
        .map(|idx| {
            let v = if idx < tr_nb_samples.saturating_sub(taper_samples) {
                1.0_f64
            } else if idx < tr_nb_samples.saturating_sub(1) {
                let n = (idx - (tr_nb_samples - taper_samples)) as f64;
                let t = taper_samples as f64;
                // Guard against division by zero at boundary
                let denom_a = (t - n) - 1.0;
                let denom_b = n + 1.0;
                if denom_a.abs() < f64::EPSILON || denom_b.abs() < f64::EPSILON {
                    0.0
                } else {
                    let zbk = t / denom_a - t / denom_b;
                    1.0 / (zbk.exp() + 1.0)
                }
            } else {
                0.0
            };
            Complex64::new(v, 0.0)
        })
        .collect()
}

// ─────────────────────────────────────────────
// GCD (integer, const-capable for Rust < 1.73)
// ─────────────────────────────────────────────
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        a %= b;
        std::mem::swap(&mut a, &mut b);
    }
    a
}

// ─────────────────────────────────────────────
// Core resampler — processes one channel vector
// ─────────────────────────────────────────────
fn resample_channel(
    input: &[f64],
    in_nb_samples: usize,
    out_nb_samples: usize,
    in_rdft_size: usize,
    out_rdft_size: usize,
    in_offset: usize,
    taper: &[Complex64],
    scale: f64,
    planner: &mut FftPlanner<f64>,
) -> Vec<f64> {
    let num_chunks = input.len() / in_nb_samples;

    // Build FFT plans once (rustfft caches internally)
    let fft = planner.plan_fft_forward(in_rdft_size);
    let ifft = planner.plan_fft_inverse(out_rdft_size);

    let freq_copy = std::cmp::min(in_rdft_size / 2 + 1, out_rdft_size / 2 + 1);

    let mut prev_chunk = vec![0.0_f64; out_nb_samples];
    let mut output = Vec::with_capacity(num_chunks * out_nb_samples);

    // Scratch buffers reused per chunk
    let mut fft_buf: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); in_rdft_size];
    let mut ifft_buf: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); out_rdft_size];

    for i in 0..num_chunks {
        let chunk = &input[i * in_nb_samples..(i + 1) * in_nb_samples];

        // Zero-pad → place chunk at in_offset
        for c in &mut fft_buf {
            *c = Complex::new(0.0, 0.0);
        }
        for (k, &s) in chunk.iter().enumerate() {
            fft_buf[in_offset + k] = Complex::new(s, 0.0);
        }

        // Forward FFT (complex, but input is real → only lower half meaningful)
        fft.process(&mut fft_buf);

        // Apply taper and copy to IFFT buffer (zero the rest)
        for c in &mut ifft_buf {
            *c = Complex::new(0.0, 0.0);
        }
        for k in 0..freq_copy {
            let tap = taper[k];
            ifft_buf[k] = Complex::new(
                fft_buf[k].re * tap.re - fft_buf[k].im * tap.im,
                fft_buf[k].re * tap.im + fft_buf[k].im * tap.re,
            );
            // Hermitian symmetry for the second half
            if k > 0 && k < out_rdft_size / 2 {
                let mirror = out_rdft_size - k;
                ifft_buf[mirror] = Complex::new(ifft_buf[k].re, -ifft_buf[k].im);
            }
        }

        // Inverse FFT
        ifft.process(&mut ifft_buf);

        // Normalize: rustfft does not normalise; divide by N
        let norm = out_rdft_size as f64;

        // Current top half = output samples; bottom half = overlap for next chunk
        let mut current = Vec::with_capacity(out_nb_samples);
        for k in 0..out_nb_samples {
            let v = ifft_buf[k].re / norm;
            current.push((v + prev_chunk[k]) * scale);
        }

        // Save overlap (bottom half)
        for k in 0..out_nb_samples {
            prev_chunk[k] = ifft_buf[out_nb_samples + k].re / norm;
        }

        output.extend_from_slice(&current);
    }

    // ── overlap flush ──────────────────────
    let tail: Vec<f64> = prev_chunk.iter().map(|&v| v * scale).collect();
    output.extend_from_slice(&tail);

    output
}

// ─────────────────────────────────────────────
// main
// ─────────────────────────────────────────────
fn main() -> Result<()> {
    println!(
        "ARDFTSRC\n  \
         Powered by mycroft @ dBpoweramp.com\n  \
         Rust port: sigmoid taper variant\n  \
         AI Generated by X.com Grok & Anthropic Claude\n"
    );

    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        println!(
            "Usage: ardftsrc <in.wav> <out.wav> <out_samplerate> <out_bitdepth> \
             [quality=2048] [bandwidth=0.95]\n\n  \
             Example: ardftsrc in.wav out.wav 48000 24 8192 0.99"
        );
        return Ok(());
    }

    let input_file = &args[1];
    let output_file = &args[2];
    let output_samplerate: u32 = args[3].parse()?;
    let output_bitdepth: u16 = args[4].parse()?;
    let quality: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let bandwidth: f64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.95);

    // ── Load input ──────────────────────────────
    let audio = load_wav(input_file)?;
    let input_samplerate = audio.sample_rate;

    println!("Source : {input_file}");
    println!("  Sample rate : {input_samplerate}");
    println!("  Bit depth   : {}", audio.bit_depth);
    println!("  Channels    : {}", audio.num_channels);
    println!("\nOutput: {output_file}");
    println!("  Sample rate : {output_samplerate}");
    println!("  Bit depth   : {output_bitdepth}");
    println!("  Channels    : {}", audio.num_channels);
    println!("\nParameters  Quality: {quality}   Bandwidth: {bandwidth}\n");

    // ── Compute sizes ────────────────────────────
    let gcd_val = gcd(input_samplerate as u64, output_samplerate as u64) as usize;
    let mut in_nb_samples = input_samplerate as usize / gcd_val;
    let mut out_nb_samples = output_samplerate as usize / gcd_val;

    let factor = (2.0 * (quality as f64 / (2.0 * out_nb_samples as f64)).ceil()) as usize;
    in_nb_samples *= factor;
    out_nb_samples *= factor;

    let in_rdft_size = in_nb_samples * 2;
    let out_rdft_size = out_nb_samples * 2;
    let in_offset = (in_rdft_size - in_nb_samples) / 2;
    let tr_nb_samples = in_nb_samples.min(out_nb_samples);

    // taper_size: use smaller half-spectrum
    let taper_size = if output_samplerate > input_samplerate {
        in_rdft_size / 2 + 1
    } else {
        out_rdft_size / 2 + 1
    };

    let taper_samples = (tr_nb_samples as f64 * (1.0 - bandwidth)).ceil() as usize;
    let scale = out_nb_samples as f64 / in_nb_samples as f64;

    // ── Build taper ──────────────────────────────
    let taper = build_taper(in_rdft_size, tr_nb_samples, taper_samples, taper_size);

    // ── Pad & resample each channel ──────────────
    let mut planner = FftPlanner::<f64>::new();
    let mut output_channels: Vec<Vec<f64>> = Vec::with_capacity(audio.num_channels);

    let num_chunks_approx = {
        let padded = {
            let r = audio.num_samples % in_nb_samples;
            if r > 0 { audio.num_samples + (in_nb_samples - r) } else { audio.num_samples }
        };
        padded / in_nb_samples
    };

    for ch in 0..audio.num_channels {
        // Pad channel to multiple of in_nb_samples
        let mut channel = audio.samples[ch].clone();
        let r = channel.len() % in_nb_samples;
        if r > 0 {
            channel.resize(channel.len() + (in_nb_samples - r), 0.0);
        }
        let num_chunks = channel.len() / in_nb_samples;

        // Progress on channel 0 only (matches C++ single-channel progress bar)
        if ch == 0 {
            // Print initial progress
            print!("Resampling  0.00 %\r");
        }

        let out = resample_channel(
            &channel,
            in_nb_samples,
            out_nb_samples,
            in_rdft_size,
            out_rdft_size,
            in_offset,
            &taper,
            scale,
            &mut planner,
        );

        output_channels.push(out);
    }

    println!("Resampling 100.00 %");

    if output_channels.is_empty() || output_channels[0].is_empty() {
        eprintln!("No output was generated");
        std::process::exit(1);
    }

    // ── Save output ──────────────────────────────
    save_wav(output_file, &output_channels, output_samplerate, output_bitdepth)?;
    println!("Saved → {output_file}");

    Ok(())
}
