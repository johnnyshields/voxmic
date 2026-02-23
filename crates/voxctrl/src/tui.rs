//! TUI mode — ratatui + crossterm terminal interface.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use voxctrl_core::config::Config;
use voxctrl_core::pipeline::SharedPipeline;
use voxctrl_core::{AppStatus, SharedState};

pub fn run_tui(
    state: Arc<SharedState>,
    cfg: Config,
    pipeline: Arc<SharedPipeline>,
    _audio_stream: cpal::Stream,
) -> Result<()> {
    // Setup terminal
    terminal::enable_raw_mode()?;
    std::io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;

    log::info!("TUI mode — Space=toggle, q/Ctrl-C=quit");

    loop {
        let status = *state.status.lock().unwrap();

        terminal.draw(|frame| {
            let area = frame.area();

            let (label, style) = match status {
                AppStatus::Idle => (
                    " IDLE ",
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                ),
                AppStatus::Recording => (
                    " RECORDING ",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                ),
                AppStatus::Transcribing => (
                    " TRANSCRIBING ",
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                ),
            };

            let text = Line::from(vec![Span::styled(label, style)]);
            let para = Paragraph::new(text)
                .alignment(Alignment::Center)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" voxctrl ")
                        .title_bottom(" Space=toggle  q/Ctrl-C=quit "),
                );

            // Center vertically
            let v_pad = area.height.saturating_sub(3) / 2;
            let centered = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(v_pad),
                    Constraint::Length(3),
                    Constraint::Min(0),
                ])
                .split(area);

            frame.render_widget(para, centered[1]);
        })?;

        // Poll for input with 100ms timeout for status refresh
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match (key.code, key.modifiers) {
                    (KeyCode::Char('c'), m) if m.contains(KeyModifiers::CONTROL) => break,
                    (KeyCode::Char('q'), _) | (KeyCode::Esc, _) => break,
                    (KeyCode::Char(' '), _) => {
                        voxctrl_core::recording::toggle_recording(&state, &cfg, &pipeline);
                    }
                    (_, _) => {}
                }
            }
        }
    }

    // Restore terminal
    terminal::disable_raw_mode()?;
    std::io::stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}
