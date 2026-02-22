; NSIS installer script for voxctrl
; Build (from repo root): makensis voxctrl/installer.nsi
; Output: voxctrl/voxctrl-0.2.0-setup.exe (OutFile resolves relative to script dir)

!define PRODUCT_NAME "Voxctrl"
!define PRODUCT_EXE "voxctrl.exe"
!define PRODUCT_VERSION "0.2.0"
!define PRODUCT_PUBLISHER "voxctrl"
!define PRODUCT_DESCRIPTION "Pluggable voice-to-action pipeline"
!define INSTALL_DIR "$LOCALAPPDATA\${PRODUCT_NAME}"
!define UNINSTALL_REG_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
!define AUTORUN_REG_KEY "Software\Microsoft\Windows\CurrentVersion\Run"

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
; OutFile and File paths use forward slashes for Linux cross-build compatibility.
; Backslashes in registry keys and Windows paths below are installer-runtime paths.
OutFile "voxctrl-${PRODUCT_VERSION}-setup.exe"
InstallDir "${INSTALL_DIR}"
RequestExecutionLevel user
SetCompressor /SOLID lzma

; ---------------------------------------------------------------------------
; Pages
; ---------------------------------------------------------------------------
Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

; ---------------------------------------------------------------------------
; Install section
; ---------------------------------------------------------------------------
Section "Install"
  SetOutPath "$INSTDIR"

  ; Copy executable
  File "target/x86_64-pc-windows-gnu/release/${PRODUCT_EXE}"

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"

  ; Auto-start via HKCU Run key
  WriteRegStr HKCU "${AUTORUN_REG_KEY}" "${PRODUCT_NAME}" '"$INSTDIR\${PRODUCT_EXE}"'

  ; Start Menu shortcuts
  CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
  CreateShortcut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" "$INSTDIR\${PRODUCT_EXE}"
  CreateShortcut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall ${PRODUCT_NAME}.lnk" "$INSTDIR\uninstall.exe"

  ; Add/Remove Programs entry
  WriteRegStr HKCU "${UNINSTALL_REG_KEY}" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr HKCU "${UNINSTALL_REG_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr HKCU "${UNINSTALL_REG_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
  WriteRegStr HKCU "${UNINSTALL_REG_KEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKCU "${UNINSTALL_REG_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKCU "${UNINSTALL_REG_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTALL_REG_KEY}" "NoRepair" 1
SectionEnd

; ---------------------------------------------------------------------------
; Uninstall section
; ---------------------------------------------------------------------------
Section "Uninstall"
  ; Remove auto-start
  DeleteRegValue HKCU "${AUTORUN_REG_KEY}" "${PRODUCT_NAME}"

  ; Remove Add/Remove Programs entry
  DeleteRegKey HKCU "${UNINSTALL_REG_KEY}"

  ; Remove Start Menu shortcuts
  Delete "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk"
  Delete "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall ${PRODUCT_NAME}.lnk"
  RMDir "$SMPROGRAMS\${PRODUCT_NAME}"

  ; Remove installed files
  Delete "$INSTDIR\${PRODUCT_EXE}"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"
SectionEnd
