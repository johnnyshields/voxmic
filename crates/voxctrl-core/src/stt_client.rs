//! STT named-pipe client — sends WAV data to the main process's STT server.
//!
//! Used by the Settings subprocess to test STT without needing direct
//! access to the pipeline.

use std::io::{Read, Write};

use anyhow::{bail, Context, Result};
use interprocess::local_socket::{ConnectOptions, ToNsName};

/// Send WAV data to the STT server and return the transcript.
pub fn transcribe_via_server(wav_data: &[u8]) -> Result<String> {
    let name = crate::PIPE_NAME.to_ns_name::<interprocess::local_socket::GenericNamespaced>()
        .context("Failed to create namespaced pipe name")?;

    let mut stream = ConnectOptions::new()
        .name(name)
        .connect_sync()
        .context("Cannot connect to STT server \u{2014} is voxctrl running?")?;

    // Send: [4 bytes: len] [N bytes: WAV]
    let len = wav_data.len() as u32;
    stream.write_all(&len.to_be_bytes())?;
    stream.write_all(wav_data)?;
    stream.flush()?;

    // Read: [1 byte: status] [4 bytes: text len] [N bytes: text]
    let mut status_buf = [0u8; 1];
    stream.read_exact(&mut status_buf)?;
    let status = status_buf[0];

    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let text_len = u32::from_be_bytes(len_buf) as usize;

    let mut text_buf = vec![0u8; text_len];
    stream.read_exact(&mut text_buf)?;

    let text = String::from_utf8(text_buf).context("STT server returned invalid UTF-8")?;

    if status == 0 {
        Ok(text)
    } else {
        bail!("STT error: {text}")
    }
}
