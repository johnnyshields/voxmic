use anyhow::{Context, Result};
use muda::{Menu, MenuId, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

use crate::AppStatus;

/// Menu item IDs returned by `build_tray()` for event matching in the event loop.
pub struct TrayMenuIds {
    pub quit: MenuId,
    pub manage_models: MenuId,
}

/// Generate a 64x64 RGBA icon with a colored circle.
pub fn make_icon(status: AppStatus) -> Icon {
    let size = 64u32;
    let mut rgba = vec![0u8; (size * size * 4) as usize];

    let (r, g, b) = match status {
        AppStatus::Idle => (0x22, 0xBB, 0x44),         // green
        AppStatus::Recording => (0xCC, 0x22, 0x22),     // red
        AppStatus::Transcribing => (0xCC, 0x99, 0x00),  // amber
    };

    let center = (size / 2) as f64;
    let radius = 30.0f64;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f64 - center;
            let dy = y as f64 - center;
            if dx * dx + dy * dy <= radius * radius {
                let idx = ((y * size + x) * 4) as usize;
                rgba[idx] = r;
                rgba[idx + 1] = g;
                rgba[idx + 2] = b;
                rgba[idx + 3] = 255;
            }
        }
    }

    Icon::from_rgba(rgba, size, size).expect("failed to create icon")
}

/// Build the system tray icon with menu.
pub fn build_tray() -> Result<(TrayIcon, TrayMenuIds)> {
    let menu = Menu::new();
    let label = MenuItem::new("voxctrl Dictation", false, None);
    let manage_models = MenuItem::new("Manage Models...", true, None);
    let quit = MenuItem::new("Quit", true, None);

    let menu_ids = TrayMenuIds {
        quit: quit.id().clone(),
        manage_models: manage_models.id().clone(),
    };

    menu.append(&label).context("menu append label")?;
    menu.append(&PredefinedMenuItem::separator()).context("menu append separator")?;
    menu.append(&manage_models).context("menu append manage models")?;
    menu.append(&PredefinedMenuItem::separator()).context("menu append separator 2")?;
    menu.append(&quit).context("menu append quit")?;

    let icon = make_icon(AppStatus::Idle);
    let tray = TrayIconBuilder::new()
        .with_icon(icon)
        .with_tooltip("voxctrl Dictation")
        .with_menu(Box::new(menu))
        .build()
        .context("build tray icon")?;

    Ok((tray, menu_ids))
}

/// Update the tray icon color to reflect current status.
#[allow(dead_code)]
pub fn update_tray_icon(tray: &TrayIcon, status: AppStatus) -> Result<()> {
    tray.set_icon(Some(make_icon(status)))
        .context("set tray icon")?;
    Ok(())
}
