Build a release: Windows binary + MSI installer.

## Steps

1. Read `Cargo.toml` to get the current version from `[package] version`.
2. Run the Windows cross-compile release build:
   ```bash
   cargo build --release --target x86_64-pc-windows-gnu
   ```
3. Build the MSI installer (wixl needs License.rtf in cwd):
   ```bash
   cp wix/License.rtf . && wixl --ext ui -o "target/voxctrl-${VERSION}-x86_64.msi" wix/main.wxs && rm License.rtf
   ```
   Replace `${VERSION}` with the version from step 1.
4. Report the sizes of both artifacts:
   - `target/x86_64-pc-windows-gnu/release/voxctrl.exe`
   - `target/voxctrl-${VERSION}-x86_64.msi`

If any step fails, stop and report the error.
