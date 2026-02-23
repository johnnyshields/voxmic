//! Pipeline — wires together STT → Router → Action.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::action::{ActionExecutor, ActionFactory};
use crate::config::Config;
use crate::router::{Intent, IntentRouter};
use crate::stt::{SttFactory, Transcriber};

// ── SharedPipeline ──────────────────────────────────────────────────────────

/// Thread-safe wrapper allowing atomic pipeline replacement.
///
/// In-flight operations keep the old pipeline alive via `Arc`; new requests
/// pick up the replacement after `swap()`.
pub struct SharedPipeline {
    inner: Mutex<Arc<Pipeline>>,
}

impl SharedPipeline {
    pub fn new(p: Pipeline) -> Self {
        Self { inner: Mutex::new(Arc::new(p)) }
    }

    /// Cheap `Arc` clone — callers snapshot the current pipeline.
    pub fn get(&self) -> Arc<Pipeline> {
        self.inner.lock().unwrap().clone()
    }

    /// Atomically replace the pipeline. Existing `Arc` holders are unaffected.
    pub fn swap(&self, new: Pipeline) {
        *self.inner.lock().unwrap() = Arc::new(new);
    }
}

pub struct Pipeline {
    pub stt: Box<dyn Transcriber>,
    pub router: Box<dyn IntentRouter>,
    pub action: Box<dyn ActionExecutor>,
}

impl Pipeline {
    /// Build a pipeline from config, creating all backends.
    ///
    /// `stt_model_dir` is the resolved local model path for backends that need
    /// local model files (e.g. voxtral-native). Other backends ignore it.
    ///
    /// `stt_factory` allows external crates to inject heavy STT backends.
    /// `action_factory` allows external crates to inject action backends (e.g. computer-use).
    pub fn from_config(
        cfg: &Config,
        stt_model_dir: Option<PathBuf>,
        stt_factory: Option<&SttFactory>,
        action_factory: Option<&ActionFactory>,
    ) -> anyhow::Result<Self> {
        let stt = crate::stt::create_transcriber(&cfg.stt, stt_model_dir, stt_factory)?;
        let router = crate::router::create_router(&cfg.router)?;
        let action = crate::action::create_action(&cfg.action, action_factory)?;

        log::info!(
            "Pipeline: STT={}, Router={}, Action={}",
            stt.name(),
            router.name(),
            action.name(),
        );

        Ok(Self {
            stt,
            router,
            action,
        })
    }

    /// Run the full pipeline from raw PCM: transcribe → route → execute.
    pub fn process_pcm(&self, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
        let start = std::time::Instant::now();

        // STT
        let text = self.stt.transcribe_pcm(samples, sample_rate)?;
        let stt_elapsed = start.elapsed().as_secs_f64();

        self.route_and_execute(start, stt_elapsed, text)
    }

    /// Shared tail of the pipeline: log STT result, route, execute.
    fn route_and_execute(
        &self,
        start: std::time::Instant,
        stt_elapsed: f64,
        text: String,
    ) -> anyhow::Result<()> {
        if text.is_empty() {
            log::info!("STT returned empty text ({:.1}s), skipping", stt_elapsed);
            return Ok(());
        }

        let preview = if text.len() > 80 { &text[..80] } else { &text };
        log::info!("STT ({:.1}s): {}", stt_elapsed, preview);

        // Route
        let intent = self.router.route(&text)?;
        match &intent {
            Intent::Dictate(t) => log::debug!("Router → Dictate({} chars)", t.len()),
            Intent::Command { action, .. } => log::info!("Router → Command({})", action),
        }

        // Execute
        self.action.execute(&intent)?;

        log::info!("Pipeline complete in {:.1}s", start.elapsed().as_secs_f64());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::action::ActionExecutor;
    use crate::router::{Intent, IntentRouter};
    use crate::stt::Transcriber;

    struct MockTranscriber {
        response: String,
    }
    impl Transcriber for MockTranscriber {
        fn transcribe(&self, _: &std::path::Path) -> anyhow::Result<String> {
            Ok(self.response.clone())
        }
        fn transcribe_pcm(&self, _: &[f32], _: u32) -> anyhow::Result<String> {
            Ok(self.response.clone())
        }
        fn name(&self) -> &str { "mock" }
        fn is_available(&self) -> bool { true }
    }

    struct MockRouter {
        routed: Arc<Mutex<Vec<String>>>,
    }
    impl IntentRouter for MockRouter {
        fn route(&self, text: &str) -> anyhow::Result<Intent> {
            self.routed.lock().unwrap().push(text.to_string());
            Ok(Intent::Dictate(text.to_string()))
        }
        fn name(&self) -> &str { "mock" }
    }

    struct MockAction {
        executed: Arc<Mutex<Vec<String>>>,
    }
    impl ActionExecutor for MockAction {
        fn execute(&self, intent: &Intent) -> anyhow::Result<()> {
            match intent {
                Intent::Dictate(t) => self.executed.lock().unwrap().push(t.clone()),
                Intent::Command { action, .. } => self.executed.lock().unwrap().push(action.clone()),
            }
            Ok(())
        }
        fn name(&self) -> &str { "mock" }
    }

    #[test]
    fn process_pcm_chains_stt_router_action() {
        let routed = Arc::new(Mutex::new(vec![]));
        let executed = Arc::new(Mutex::new(vec![]));

        let pipeline = Pipeline {
            stt: Box::new(MockTranscriber { response: "hello world".into() }),
            router: Box::new(MockRouter { routed: routed.clone() }),
            action: Box::new(MockAction { executed: executed.clone() }),
        };

        pipeline.process_pcm(&[0.1, 0.2], 16000).unwrap();

        assert_eq!(&*routed.lock().unwrap(), &["hello world"]);
        assert_eq!(&*executed.lock().unwrap(), &["hello world"]);
    }

    #[test]
    fn process_pcm_skips_empty_text() {
        let routed = Arc::new(Mutex::new(vec![]));
        let executed = Arc::new(Mutex::new(vec![]));

        let pipeline = Pipeline {
            stt: Box::new(MockTranscriber { response: "".into() }),
            router: Box::new(MockRouter { routed: routed.clone() }),
            action: Box::new(MockAction { executed: executed.clone() }),
        };

        pipeline.process_pcm(&[0.1], 16000).unwrap();

        assert!(routed.lock().unwrap().is_empty(), "router should not be called for empty text");
        assert!(executed.lock().unwrap().is_empty(), "action should not be called for empty text");
    }

    #[test]
    fn process_pcm_calls_transcribe_pcm_not_transcribe() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct TrackingTranscriber {
            pcm_called: Arc<AtomicBool>,
            file_called: Arc<AtomicBool>,
        }
        impl Transcriber for TrackingTranscriber {
            fn transcribe(&self, _: &std::path::Path) -> anyhow::Result<String> {
                self.file_called.store(true, Ordering::SeqCst);
                Ok("from-file".into())
            }
            fn transcribe_pcm(&self, samples: &[f32], _: u32) -> anyhow::Result<String> {
                self.pcm_called.store(true, Ordering::SeqCst);
                Ok(format!("{} samples", samples.len()))
            }
            fn name(&self) -> &str { "tracking" }
            fn is_available(&self) -> bool { true }
        }

        let pcm_called = Arc::new(AtomicBool::new(false));
        let file_called = Arc::new(AtomicBool::new(false));
        let pipeline = Pipeline {
            stt: Box::new(TrackingTranscriber {
                pcm_called: pcm_called.clone(),
                file_called: file_called.clone(),
            }),
            router: Box::new(MockRouter { routed: Arc::new(Mutex::new(vec![])) }),
            action: Box::new(MockAction { executed: Arc::new(Mutex::new(vec![])) }),
        };

        pipeline.process_pcm(&[0.1; 1600], 16000).unwrap();
        assert!(pcm_called.load(Ordering::SeqCst), "transcribe_pcm should have been called");
        assert!(!file_called.load(Ordering::SeqCst), "transcribe (file) should NOT have been called");
    }

    #[test]
    fn process_pcm_propagates_stt_error() {
        struct FailTranscriber;
        impl Transcriber for FailTranscriber {
            fn transcribe(&self, _: &std::path::Path) -> anyhow::Result<String> { anyhow::bail!("fail") }
            fn transcribe_pcm(&self, _: &[f32], _: u32) -> anyhow::Result<String> { anyhow::bail!("stt failed") }
            fn name(&self) -> &str { "fail" }
            fn is_available(&self) -> bool { false }
        }

        let pipeline = Pipeline {
            stt: Box::new(FailTranscriber),
            router: Box::new(MockRouter { routed: Arc::new(Mutex::new(vec![])) }),
            action: Box::new(MockAction { executed: Arc::new(Mutex::new(vec![])) }),
        };

        let result = pipeline.process_pcm(&[0.1], 16000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("stt failed"));
    }

    // ── SharedPipeline tests ──────────────────────────────────────────

    fn make_shared(response: &str) -> SharedPipeline {
        SharedPipeline::new(Pipeline {
            stt: Box::new(MockTranscriber { response: response.into() }),
            router: Box::new(MockRouter { routed: Arc::new(Mutex::new(vec![])) }),
            action: Box::new(MockAction { executed: Arc::new(Mutex::new(vec![])) }),
        })
    }

    #[test]
    fn shared_pipeline_get_returns_current() {
        let sp = make_shared("hello");
        let p = sp.get();
        assert_eq!(p.stt.name(), "mock");
    }

    #[test]
    fn shared_pipeline_swap_replaces_pipeline() {
        let sp = make_shared("v1");
        let old = sp.get();

        sp.swap(Pipeline {
            stt: Box::new(MockTranscriber { response: "v2".into() }),
            router: Box::new(MockRouter { routed: Arc::new(Mutex::new(vec![])) }),
            action: Box::new(MockAction { executed: Arc::new(Mutex::new(vec![])) }),
        });

        let new = sp.get();
        // Old holder still works
        assert_eq!(old.stt.transcribe_pcm(&[], 0).unwrap(), "v1");
        // New holder gets the replacement
        assert_eq!(new.stt.transcribe_pcm(&[], 0).unwrap(), "v2");
    }

    #[test]
    fn shared_pipeline_inflight_survives_swap() {
        let sp = Arc::new(make_shared("original"));
        let snapshot = sp.get();

        sp.swap(Pipeline {
            stt: Box::new(MockTranscriber { response: "replaced".into() }),
            router: Box::new(MockRouter { routed: Arc::new(Mutex::new(vec![])) }),
            action: Box::new(MockAction { executed: Arc::new(Mutex::new(vec![])) }),
        });

        // snapshot still works with original pipeline
        assert_eq!(snapshot.stt.transcribe_pcm(&[], 0).unwrap(), "original");
        // new get() returns the replacement
        assert_eq!(sp.get().stt.transcribe_pcm(&[], 0).unwrap(), "replaced");
    }
}
