//! voxctrl-core — Core library for the voxctrl voice-to-action pipeline.
//!
//! Provides traits, config, pipeline, audio, models, and lightweight backends.
//! Heavy ML inference backends (whisper-native, whisper-cpp, voxtral-native)
//! live in the separate `voxctrl-stt` crate.

pub mod action;
pub mod audio;
pub mod config;
pub mod models;
pub mod pipeline;
pub mod recording;
pub mod router;
pub mod stt;
pub mod stt_client;
pub mod stt_server;
pub mod vad;
pub mod gpu;

use std::sync::Mutex;

// ── IPC ──────────────────────────────────────────────────────────────────────

/// Named-pipe name shared between STT client and server.
pub const PIPE_NAME: &str = "voxctrl-stt";

// ── Shared state ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppStatus {
    Idle,
    Recording,
    Transcribing,
}

pub struct SharedState {
    pub status: Mutex<AppStatus>,
    pub chunks: Mutex<Vec<f32>>,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            status: Mutex::new(AppStatus::Idle),
            chunks: Mutex::new(Vec::new()),
        }
    }
}

impl Default for SharedState {
    fn default() -> Self {
        Self::new()
    }
}
