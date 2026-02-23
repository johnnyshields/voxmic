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

use crate::pipeline::{Pipeline, SharedPipeline};

const MAX_PAYLOAD: u32 = 100_000_000; // 100 MB

/// Start the STT named-pipe server on a background thread.
pub fn start(pipeline: Arc<SharedPipeline>) -> Result<()> {
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
                        // Snapshot the current pipeline per connection
                        let snap = pipeline.get();
                        std::thread::spawn(move || {
                            if let Err(e) = handle_connection(stream, &snap) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use crate::action::ActionExecutor;
    use crate::router::{Intent, IntentRouter};
    use crate::stt::Transcriber;

    struct EchoTranscriber;
    impl Transcriber for EchoTranscriber {
        fn transcribe(&self, _: &std::path::Path) -> Result<String> { Ok("file".into()) }
        fn transcribe_pcm(&self, samples: &[f32], sample_rate: u32) -> Result<String> {
            Ok(format!("n={},sr={}", samples.len(), sample_rate))
        }
        fn name(&self) -> &str { "echo" }
        fn is_available(&self) -> bool { true }
    }

    struct FailTranscriber;
    impl Transcriber for FailTranscriber {
        fn transcribe(&self, _: &std::path::Path) -> Result<String> { anyhow::bail!("fail") }
        fn transcribe_pcm(&self, _: &[f32], _: u32) -> Result<String> { anyhow::bail!("stt error") }
        fn name(&self) -> &str { "fail" }
        fn is_available(&self) -> bool { false }
    }

    struct NoopRouter;
    impl IntentRouter for NoopRouter {
        fn route(&self, text: &str) -> Result<Intent> { Ok(Intent::Dictate(text.into())) }
        fn name(&self) -> &str { "noop" }
    }

    struct NoopAction;
    impl ActionExecutor for NoopAction {
        fn execute(&self, _: &Intent) -> Result<()> { Ok(()) }
        fn name(&self) -> &str { "noop" }
    }

    /// Encode a PCM request per the wire protocol.
    fn encode_request(sample_rate: u32, samples: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&sample_rate.to_be_bytes());
        buf.extend_from_slice(&(samples.len() as u32).to_be_bytes());
        for &s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf
    }

    /// Decode a response from raw bytes: returns (status, text).
    fn decode_response(bytes: &[u8]) -> (u8, String) {
        let status = bytes[0];
        let len = u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
        let text = String::from_utf8(bytes[5..5 + len].to_vec()).unwrap();
        (status, text)
    }

    fn make_pipeline(stt: Box<dyn Transcriber>) -> Pipeline {
        Pipeline {
            stt,
            router: Box::new(NoopRouter),
            action: Box::new(NoopAction),
        }
    }

    #[test]
    fn wire_protocol_success() {
        let pipeline = make_pipeline(Box::new(EchoTranscriber));
        let samples = vec![0.1f32, 0.2, 0.3];
        let request = encode_request(16000, &samples);

        let mut stream = Cursor::new(request);
        handle_connection(&mut stream, &pipeline).unwrap();

        let data = stream.into_inner();
        // Response starts after the request bytes (8 header + 12 PCM = 20 bytes)
        let response = &data[20..];
        let (status, text) = decode_response(response);
        assert_eq!(status, 0);
        assert_eq!(text, "n=3,sr=16000");
    }

    #[test]
    fn wire_protocol_error() {
        let pipeline = make_pipeline(Box::new(FailTranscriber));
        let request = encode_request(16000, &[1.0]);

        let mut stream = Cursor::new(request);
        handle_connection(&mut stream, &pipeline).unwrap();

        let data = stream.into_inner();
        let response = &data[12..]; // 8 header + 4 PCM
        let (status, text) = decode_response(response);
        assert_eq!(status, 1);
        assert!(text.contains("stt error"), "error text: {text}");
    }

    #[test]
    fn wire_protocol_payload_too_large() {
        // Craft a request claiming a huge number of samples, with only a small body.
        let mut buf = Vec::new();
        buf.extend_from_slice(&16000u32.to_be_bytes()); // sample_rate
        let huge_count = MAX_PAYLOAD / 4 + 1;
        buf.extend_from_slice(&huge_count.to_be_bytes()); // num_samples
        // Provide enough trailing bytes that the stream doesn't EOF before the error response.
        // handle_connection sends the error before trying to read PCM.
        // Actually, it checks payload size before reading, so we just need the header.
        // Pad with enough zeros for the write to land correctly.
        buf.resize(buf.len() + 256, 0);

        let mut stream = Cursor::new(buf);
        handle_connection(&mut stream, &make_pipeline(Box::new(EchoTranscriber))).unwrap();

        let data = stream.into_inner();
        // Response starts after the 8-byte header we consumed
        let response = &data[8..];
        // Find the response: first byte after header area should be status=1 (error)
        // The cursor position after reading 8 bytes is at offset 8, writes go there
        let (status, text) = decode_response(response);
        assert_eq!(status, 1);
        assert!(text.contains("too large"), "error text: {text}");
    }

    #[test]
    fn wire_protocol_empty_samples() {
        let pipeline = make_pipeline(Box::new(EchoTranscriber));
        let request = encode_request(44100, &[]);

        let mut stream = Cursor::new(request);
        handle_connection(&mut stream, &pipeline).unwrap();

        let data = stream.into_inner();
        let response = &data[8..]; // 8 header + 0 PCM
        let (status, text) = decode_response(response);
        assert_eq!(status, 0);
        assert_eq!(text, "n=0,sr=44100");
    }
}
