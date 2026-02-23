use std::path::PathBuf;
use voxctrl_core::models::cache_scanner;
use voxctrl_core::stt::Transcriber;
use voxctrl_stt::voxtral_native::VoxtralNativeTranscriber;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let wav_path = std::env::args().nth(1).expect("Usage: test_voxtral <wav_path>");
    let wav_path = PathBuf::from(&wav_path);

    let model_dir = cache_scanner::find_hf_model(
        "mistralai/Voxtral-Mini-4B-Realtime-2602",
        &["consolidated.safetensors"],
    );

    log::info!("Model dir: {:?}", model_dir);

    log::info!("Creating transcriber...");
    let transcriber = VoxtralNativeTranscriber::new(model_dir)?;

    log::info!("Transcribing {:?}...", wav_path);
    let result = transcriber.transcribe(&wav_path)?;

    println!("\n=== TRANSCRIPTION RESULT ===");
    println!("{result}");
    println!("============================\n");

    Ok(())
}
