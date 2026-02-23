//! Audio capture — cpal always-on input stream with optional VAD gating.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, StreamConfig};

use crate::config::Config;
use crate::{AppStatus, SharedState};

/// Start the always-on audio capture stream.
pub fn start_capture(state: Arc<SharedState>, cfg: &Config) -> Result<cpal::Stream> {
    let host = cpal::default_host();

    let pattern = cfg.audio.device_pattern.to_lowercase();
    let device = host
        .input_devices()
        .context("Failed to enumerate input devices")?
        .find(|d| {
            d.name()
                .map(|n| n.to_lowercase().contains(&pattern))
                .unwrap_or(false)
        })
        .or_else(|| host.default_input_device())
        .context("No input audio device found")?;

    let device_name = device.name().unwrap_or_else(|_| "<unknown>".into());
    log::info!("Audio device: {device_name}");

    let desired_rate = SampleRate(cfg.audio.sample_rate);
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
            log::warn!(
                "{}Hz not supported by '{}'; falling back to {}Hz",
                cfg.audio.sample_rate,
                device_name,
                default.sample_rate().0,
            );
            default.into()
        }
    };

    log::info!(
        "Stream: {}Hz, {}ch",
        stream_config.sample_rate.0,
        stream_config.channels,
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
                    chunks.extend_from_slice(data);
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
/// Returns (stream, level_receiver). The stream must be kept alive.
/// `test_chunks` receives raw f32 samples when `recording` is true.
/// Returns `(stream, actual_sample_rate)`.
pub fn start_test_capture(
    device_pattern: &str,
    sample_rate: u32,
    level_tx: std::sync::mpsc::Sender<f32>,
    test_chunks: Arc<std::sync::Mutex<Vec<f32>>>,
    recording: Arc<AtomicBool>,
) -> Result<(cpal::Stream, u32)> {
    let host = cpal::default_host();
    let pattern = device_pattern.to_lowercase();
    let device = host
        .input_devices()
        .context("Failed to enumerate input devices")?
        .find(|d| {
            d.name()
                .map(|n| n.to_lowercase().contains(&pattern))
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
            log::warn!("Device does not support {}Hz, using default {}Hz",
                sample_rate, default.sample_rate().0);
            default.into()
        }
    };

    let actual_rate = stream_config.sample_rate.0;
    log::info!("Test capture: device={:?}, actual_rate={}Hz", device.name(), actual_rate);

    let stream = device
        .build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Compute RMS for level meter
                if !data.is_empty() {
                    let sum: f32 = data.iter().map(|s| s * s).sum();
                    let rms = (sum / data.len() as f32).sqrt();
                    let _ = level_tx.send(rms);
                }
                // Record raw samples if recording flag is set
                if recording.load(std::sync::atomic::Ordering::Relaxed) {
                    let mut chunks = test_chunks.lock().unwrap();
                    chunks.extend_from_slice(data);
                }
            },
            |err| log::error!("Test audio capture error: {err}"),
            None,
        )
        .context("Failed to build test audio input stream")?;

    stream.play().context("Failed to start test audio stream")?;
    Ok((stream, actual_rate))
}
