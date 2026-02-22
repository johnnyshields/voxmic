//! STT TCP server — proxies transcription requests to the pipeline's STT backend.
//!
//! Wire protocol (localhost only, length-prefixed):
//!   Request:  [4 bytes: WAV len u32 BE] [N bytes: WAV data]
//!   Response: [1 byte: status 0=ok 1=err] [4 bytes: text len u32 BE] [N bytes: UTF-8 text]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::pipeline::Pipeline;

const MAX_PAYLOAD: u32 = 100_000_000; // 100 MB
const IO_TIMEOUT: Duration = Duration::from_secs(30);

/// Start the STT TCP server on a background thread.
///
/// Binds to `127.0.0.1:{port}` and spawns a thread per connection.
pub fn start(pipeline: Arc<Pipeline>, port: u16) -> Result<()> {
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr)
        .with_context(|| format!("Failed to bind STT server on {addr}"))?;

    log::info!("STT server listening on {addr}");

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

fn handle_connection(mut stream: TcpStream, pipeline: &Pipeline) -> Result<()> {
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))?;

    // Read WAV length
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let wav_len = u32::from_be_bytes(len_buf);

    if wav_len > MAX_PAYLOAD {
        send_error(&mut stream, &format!("Payload too large: {wav_len} bytes"))?;
        return Ok(());
    }

    // Read WAV data
    let mut wav_data = vec![0u8; wav_len as usize];
    stream.read_exact(&mut wav_data)?;

    // Write to tempfile and transcribe
    let result = (|| -> Result<String> {
        let mut tmp = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .context("Failed to create temp WAV file")?;
        tmp.write_all(&wav_data)?;
        tmp.flush()?;
        let path = tmp.path().to_path_buf();
        pipeline.stt.transcribe(&path)
    })();

    match result {
        Ok(text) => send_ok(&mut stream, &text),
        Err(e) => send_error(&mut stream, &format!("{e:#}")),
    }
}

fn send_ok(stream: &mut TcpStream, text: &str) -> Result<()> {
    let text_bytes = text.as_bytes();
    let mut buf = Vec::with_capacity(1 + 4 + text_bytes.len());
    buf.push(0); // status: ok
    buf.extend_from_slice(&(text_bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(text_bytes);
    stream.write_all(&buf)?;
    Ok(())
}

fn send_error(stream: &mut TcpStream, msg: &str) -> Result<()> {
    let msg_bytes = msg.as_bytes();
    let mut buf = Vec::with_capacity(1 + 4 + msg_bytes.len());
    buf.push(1); // status: error
    buf.extend_from_slice(&(msg_bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(msg_bytes);
    stream.write_all(&buf)?;
    Ok(())
}
