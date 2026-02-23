Build a debug binary (Windows cross-compile).

## Steps

1. Run the Windows cross-compile debug build:
   ```bash
   cargo build --target x86_64-pc-windows-gnu
   ```
2. Report the size of the artifact:
   - `target/x86_64-pc-windows-gnu/debug/voxctrl.exe`

If the build fails, stop and report the error.
