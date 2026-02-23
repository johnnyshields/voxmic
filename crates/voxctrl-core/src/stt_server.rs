//! STT named-pipe server — proxies transcription requests to the pipeline's STT backend.
//!
//! Wire protocol (PCM-based):
//!   Request:  [4 bytes: sample_rate u32 BE] [4 bytes: num_samples u32 BE] [N*4 bytes: f32 samples LE]
//!   Response: [1 byte: status 0=ok 1=err] [4 bytes: text len u32 BE] [N bytes: UTF-8 text]

use std::io::{Read, Write};
use std::sync::Arc;

use anyhow::{Context, Result};
use interprocess::local_socket::{ListenerOptions, ToNsName};
use interprocess::local_socket::traits::ListenerExt;

use crate::pipeline::Pipeline;

const MAX_PAYLOAD: u32 = 100_000_000; // 100 MB

/// Start the STT named-pipe server on a background thread.
pub fn start(pipeline: Arc<Pipeline>) -> Result<()> {
    let name = crate::PIPE_NAME.to_ns_name::<interprocess::local_socket::GenericNamespaced>()
        .context("Failed to create namespaced pipe name")?;

    let listener = ListenerOptions::new()
        .name(name)
        .create_sync()
        .context("Failed to create STT named-pipe listener")?;

    log::info!("STT server listening on named pipe: {}", crate::PIPE_NAME);

    std::thread::Builder::new()
        .name("stt-server".into())
        .spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let pipeline = pipeline.clone();
                        std::thread::spawn(move || {
                            if let Err(e) = handle_connection(stream, &pipeline) {
                                log::warn!("STT server connection error: {e}");
                            }
                        });
                    }
                    Err(e) => log::warn!("STT server accept error: {e}"),
                }
            }
        })
        .context("Failed to spawn STT server thread")?;

    Ok(())
}

fn handle_connection(mut stream: impl Read + Write, pipeline: &Pipeline) -> Result<()> {
    let mut header = [0u8; 4];

    // Read sample rate
    stream.read_exact(&mut header)?;
    let sample_rate = u32::from_be_bytes(header);

    // Read number of samples
    stream.read_exact(&mut header)?;
    let num_samples = u32::from_be_bytes(header);

    let payload_bytes = num_samples.saturating_mul(4);
    if payload_bytes > MAX_PAYLOAD {
        send_error(&mut stream, &format!("Payload too large: {payload_bytes} bytes"))?;
        return Ok(());
    }

    // Read raw f32 PCM samples (little-endian)
    let mut pcm_bytes = vec![0u8; payload_bytes as usize];
    stream.read_exact(&mut pcm_bytes)?;

    let samples: Vec<f32> = pcm_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let result = pipeline.stt.transcribe_pcm(&samples, sample_rate);

    match result {
        Ok(text) => send_ok(&mut stream, &text),
        Err(e) => send_error(&mut stream, &format!("{e:#}")),
    }
}

fn send_ok(stream: &mut impl Write, text: &str) -> Result<()> {
    let text_bytes = text.as_bytes();
    let mut buf = Vec::with_capacity(1 + 4 + text_bytes.len());
    buf.push(0); // status: ok
    buf.extend_from_slice(&(text_bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(text_bytes);
    stream.write_all(&buf)?;
    Ok(())
}

fn send_error(stream: &mut impl Write, msg: &str) -> Result<()> {
    let msg_bytes = msg.as_bytes();
    let mut buf = Vec::with_capacity(1 + 4 + msg_bytes.len());
    buf.push(1); // status: error
    buf.extend_from_slice(&(msg_bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(msg_bytes);
    stream.write_all(&buf)?;
    Ok(())
}
