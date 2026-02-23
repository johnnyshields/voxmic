use std::path::PathBuf;
use voxctrl_core::config::SttConfig;
use voxctrl_core::stt::Transcriber;
use voxctrl_stt::whisper_native::WhisperNativeTranscriber;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let wav_path = std::env::args().nth(1).expect("Usage: test_whisper <wav_path>");
    let wav_path = PathBuf::from(&wav_path);

    // Try standard cache dir, then Windows AppData path
    let model_dir = dirs::cache_dir()
        .map(|d| d.join("huggingface/hub/models--openai--whisper-tiny/snapshots/main"))
        .filter(|d| d.join("model.safetensors").exists())
        .or_else(|| {
            let p = PathBuf::from("/mnt/c/Users/John/AppData/Local/huggingface/hub/models--openai--whisper-tiny/snapshots/main");
            if p.join("model.safetensors").exists() { Some(p) } else { None }
        });

    log::info!("Model dir: {:?}", model_dir);

    let cfg = SttConfig {
        backend: "whisper-native".into(),
        whisper_model: "tiny".into(),
        whisper_device: "cpu".into(),
        ..Default::default()
    };

    log::info!("Creating transcriber...");
    let transcriber = WhisperNativeTranscriber::new(&cfg, model_dir)?;

    log::info!("Transcribing {:?}...", wav_path);
    let result = transcriber.transcribe(&wav_path)?;

    println!("\n=== TRANSCRIPTION RESULT ===");
    println!("{result}");
    println!("============================\n");

    Ok(())
}
