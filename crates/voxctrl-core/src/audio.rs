//! Audio capture — cpal always-on input stream with optional VAD gating.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleRate, StreamConfig};

use crate::config::Config;
use crate::{AppStatus, SharedState};

/// Find an input device matching `pattern` and build a `StreamConfig` at the
/// requested sample rate (mono if supported), falling back to the device
/// default when the exact rate isn't supported.
/// Returns `(device, config, actual_sample_rate, channels)`.
fn resolve_device_and_config(pattern: &str, sample_rate: u32) -> Result<(Device, StreamConfig, u32, u16)> {
    let host = cpal::default_host();
    let pat = pattern.to_lowercase();
    let device = host
        .input_devices()
        .context("Failed to enumerate input devices")?
        .find(|d| {
            d.name()
                .map(|n| n.to_lowercase().contains(&pat))
                .unwrap_or(false)
        })
        .or_else(|| host.default_input_device())
        .context("No input audio device found")?;

    let desired_rate = SampleRate(sample_rate);
    let stream_config: StreamConfig = match device
        .supported_input_configs()
        .context("Cannot query device input configs")?
        .find(|c| {
            c.channels() >= 1
                && c.min_sample_rate() <= desired_rate
                && desired_rate <= c.max_sample_rate()
        }) {
        Some(range) => {
            let mut sc: StreamConfig = range.with_sample_rate(desired_rate).into();
            sc.channels = 1;
            sc
        }
        None => {
            let default = device
                .default_input_config()
                .context("No default input config")?;
            let device_name = device.name().unwrap_or_else(|_| "<unknown>".into());
            log::warn!(
                "{}Hz not supported by '{}'; falling back to {}Hz, {}ch",
                sample_rate,
                device_name,
                default.sample_rate().0,
                default.channels(),
            );
            default.into()
        }
    };

    let actual_rate = stream_config.sample_rate.0;
    let channels = stream_config.channels;
    Ok((device, stream_config, actual_rate, channels))
}

/// Downmix interleaved multi-channel audio to mono by averaging channels per frame.
#[inline]
fn downmix_to_mono(data: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    let ch = channels as usize;
    data.chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

/// Start the always-on audio capture stream.
pub fn start_capture(state: Arc<SharedState>, cfg: &Config) -> Result<cpal::Stream> {
    let (device, stream_config, actual_rate, channels) =
        resolve_device_and_config(&cfg.audio.device_pattern, cfg.audio.sample_rate)?;

    let device_name = device.name().unwrap_or_else(|_| "<unknown>".into());
    log::info!("Audio device: {device_name}");
    log::info!(
        "Stream: {}Hz, {}ch{}",
        actual_rate,
        channels,
        if channels > 1 { " (downmixing to mono)" } else { "" },
    );

    let state_cb = Arc::clone(&state);
    let stream = device
        .build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let is_recording = {
                    let status = state_cb.status.lock().unwrap();
                    *status == AppStatus::Recording
                };
                if is_recording {
                    let mut chunks = state_cb.chunks.lock().unwrap();
                    if channels <= 1 {
                        chunks.extend_from_slice(data);
                    } else {
                        let mono = downmix_to_mono(data, channels);
                        chunks.extend_from_slice(&mono);
                    }
                }
            },
            |err| log::error!("Audio capture error: {err}"),
            None,
        )
        .context("Failed to build audio input stream")?;

    stream.play().context("Failed to start audio stream")?;
    Ok(stream)
}

/// List all available audio input device names.
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devices) => devices.filter_map(|d| d.name().ok()).collect(),
        Err(e) => {
            log::warn!("Failed to enumerate input devices: {e}");
            vec![]
        }
    }
}

/// Start a test audio capture stream that sends RMS levels and raw samples.
///
/// `test_chunks` receives mono f32 samples when `recording` is true.
/// Returns `(stream, actual_sample_rate)`.
pub fn start_test_capture(
    device_pattern: &str,
    sample_rate: u32,
    level_tx: std::sync::mpsc::Sender<f32>,
    test_chunks: Arc<std::sync::Mutex<Vec<f32>>>,
    recording: Arc<AtomicBool>,
) -> Result<(cpal::Stream, u32)> {
    let (device, stream_config, actual_rate, channels) =
        resolve_device_and_config(device_pattern, sample_rate)?;

    log::info!(
        "Test capture: device={:?}, actual_rate={}Hz, channels={}",
        device.name(), actual_rate, channels
    );

    let stream = device
        .build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Compute RMS for level meter (use raw data, all channels)
                if !data.is_empty() {
                    let sum: f32 = data.iter().map(|s| s * s).sum();
                    let rms = (sum / data.len() as f32).sqrt();
                    let _ = level_tx.send(rms);
                }
                // Record mono samples if recording flag is set
                if recording.load(std::sync::atomic::Ordering::Relaxed) {
                    let mut chunks = test_chunks.lock().unwrap();
                    if channels <= 1 {
                        chunks.extend_from_slice(data);
                    } else {
                        let mono = downmix_to_mono(data, channels);
                        chunks.extend_from_slice(&mono);
                    }
                }
            },
            |err| log::error!("Test audio capture error: {err}"),
            None,
        )
        .context("Failed to build test audio input stream")?;

    stream.play().context("Failed to start test audio stream")?;
    Ok((stream, actual_rate))
}
