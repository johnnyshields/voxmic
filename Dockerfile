FROM rust:1.85-bookworm

# System libraries needed by voxctrl dependencies:
#   libasound2-dev    — cpal (ALSA audio capture)
#   libgtk-3-dev      — tray-icon, muda, rfd (system tray, menus, file dialogs)
#   libxdo-dev        — enigo (simulated keyboard input via xdotool)
#   mingw-w64         — Windows cross-compilation (x86_64-pc-windows-gnu)
#   nsis              — NSIS installer compiler (makensis)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2-dev \
    libgtk-3-dev \
    libxdo-dev \
    mingw-w64 \
    nsis \
    && rm -rf /var/lib/apt/lists/*

# Add Windows cross-compilation target
RUN rustup target add x86_64-pc-windows-gnu

WORKDIR /app
COPY . .

# Build for Linux (native)
RUN cargo build --release --manifest-path voxctrl/Cargo.toml

# Build for Windows (cross-compile)
RUN cargo build --release --manifest-path voxctrl/Cargo.toml --target x86_64-pc-windows-gnu

# Build Windows installer
RUN makensis voxctrl/installer.nsi
