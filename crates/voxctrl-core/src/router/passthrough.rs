//! Passthrough router — all text becomes dictation.

use super::{Intent, IntentRouter};

/// Routes every transcription as [`Intent::Dictate`]. No classification needed.
pub struct PassthroughRouter;

impl IntentRouter for PassthroughRouter {
    fn route(&self, text: &str) -> anyhow::Result<Intent> {
        Ok(Intent::Dictate(text.to_string()))
    }

    fn name(&self) -> &str {
        "passthrough"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictate_wraps_text() {
        let router = PassthroughRouter;
        let intent = router.route("hello world").unwrap();
        match intent {
            Intent::Dictate(text) => assert_eq!(text, "hello world"),
            Intent::Command { .. } => panic!("expected Dictate, got Command"),
        }
    }

    #[test]
    fn dictate_empty_string() {
        let router = PassthroughRouter;
        let intent = router.route("").unwrap();
        match intent {
            Intent::Dictate(text) => assert_eq!(text, ""),
            Intent::Command { .. } => panic!("expected Dictate, got Command"),
        }
    }

    #[test]
    fn name_is_passthrough() {
        assert_eq!(PassthroughRouter.name(), "passthrough");
    }
}
