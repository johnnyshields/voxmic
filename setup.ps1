#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Voxtral Windows Setup - llama-server + DJI mic transcription
.DESCRIPTION
    Uses Windows Scheduled Tasks only (no third-party tools):
      VoxtralLLM  - starts at boot as SYSTEM (before any user logs in)
      VoxtralMic  - starts at logon as current user (needs audio session)
.EXAMPLE
    .\setup.ps1
#>

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ---- Configuration ----------------------------------------------------------
$WORKSPACE    = "C:\workspace\voxtral"
$LLAMA_URL    = "https://github.com/ggml-org/llama.cpp/releases/download/b8116/llama-b8116-bin-win-cpu-x64.zip"
$MODEL_REPO   = "ggml-org/Voxtral-Mini-3B-2507-GGUF"
$LLM_PORT     = 5200
$SCRIPT_DIR   = $PSScriptRoot
$CURRENT_USER = $env:USERNAME

function Step($n, $total, $msg) {
    Write-Host ""
    Write-Host "[$n/$total] $msg" -ForegroundColor Cyan
}
function OK($msg)   { Write-Host "    OK   $msg" -ForegroundColor Green }
function WARN($msg) { Write-Host "    WARN $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "=== Voxtral Windows Setup ===" -ForegroundColor Cyan
Write-Host "    Workspace : $WORKSPACE"
Write-Host "    User      : $CURRENT_USER"
Write-Host "    Model     : $MODEL_REPO"
Write-Host "    Port      : $LLM_PORT"

# ---- 1. Directories ---------------------------------------------------------
Step 1 6 "Creating directory structure"
foreach ($d in @($WORKSPACE, "$WORKSPACE\llama-win", "$WORKSPACE\models",
                 "$WORKSPACE\logs", "$WORKSPACE\output")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
OK $WORKSPACE

# ---- 2. llama.cpp -----------------------------------------------------------
Step 2 6 "Setting up llama.cpp"
$llamaExe = "$WORKSPACE\llama-win\llama-server.exe"
if (-not (Test-Path $llamaExe)) {
    $zip = "$WORKSPACE\llama-win.zip"
    Write-Host "    Downloading llama.cpp ..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $LLAMA_URL -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath "$WORKSPACE\llama-tmp" -Force
    Remove-Item $zip

    $found = Get-ChildItem "$WORKSPACE\llama-tmp" -Recurse -Filter "llama-server.exe" |
             Select-Object -First 1
    if (-not $found) { throw "llama-server.exe not found in archive - check URL." }
    Get-ChildItem $found.DirectoryName | Copy-Item -Destination "$WORKSPACE\llama-win" -Force
    Remove-Item -Recurse "$WORKSPACE\llama-tmp"
    OK "Extracted to $WORKSPACE\llama-win"
} else {
    OK "Already present - skipping download"
}

# ---- 3. Python packages -----------------------------------------------------
Step 3 6 "Installing Python packages"
$pythonExe = (Get-Command python -ErrorAction Stop).Source
Write-Host "    Python: $pythonExe"
& $pythonExe -m pip install sounddevice numpy scipy requests faster-whisper pystray pillow pynput --quiet
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
OK "sounddevice, numpy, scipy, requests, faster-whisper, pystray, pillow, pynput"

# ---- 4. Deploy scripts ------------------------------------------------------
Step 4 6 "Deploying scripts"
foreach ($f in @("transcribe.py", "tray_app.py", "config.json", "list-devices.py", "monitor.bat")) {
    $src = Join-Path $SCRIPT_DIR $f
    if (Test-Path $src) {
        Copy-Item $src "$WORKSPACE\$f" -Force
        OK $f
    } else {
        WARN "$f not found in $SCRIPT_DIR - skipping"
    }
}

# Batch file wrapper for llama-server (handles stdout/stderr -> log file)
@"
@echo off
set HF_HOME=$WORKSPACE\models
"$llamaExe" -hf "$MODEL_REPO" --port $LLM_PORT --host 127.0.0.1 >> "$WORKSPACE\logs\llm.log" 2>&1
"@ | Out-File "$WORKSPACE\start-llm.bat" -Encoding ascii
OK "start-llm.bat (log wrapper)"

# ---- 5. VoxtralLLM scheduled task (boot, SYSTEM, no user session needed) ----
Step 5 6 "Installing VoxtralLLM scheduled task (at startup, SYSTEM)"

Unregister-ScheduledTask -TaskName "VoxtralLLM" -Confirm:$false -ErrorAction SilentlyContinue

$llmAction = New-ScheduledTaskAction `
    -Execute          "cmd.exe" `
    -Argument         "/C `"$WORKSPACE\start-llm.bat`"" `
    -WorkingDirectory "$WORKSPACE\llama-win"

$llmTrigger = New-ScheduledTaskTrigger -AtStartup

$llmSettings = New-ScheduledTaskSettingsSet `
    -RestartCount              5 `
    -RestartInterval           (New-TimeSpan -Minutes 2) `
    -ExecutionTimeLimit        (New-TimeSpan -Hours 0) `
    -RunOnlyIfNetworkAvailable:$false `
    -StartWhenAvailable `
    -DisallowDemandStart:$false

$llmPrincipal = New-ScheduledTaskPrincipal `
    -UserId    "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel  Highest

Register-ScheduledTask `
    -TaskName    "VoxtralLLM" `
    -Action      $llmAction `
    -Trigger     $llmTrigger `
    -Settings    $llmSettings `
    -Principal   $llmPrincipal `
    -Description "llama-server: Voxtral-Mini-3B audio transcription endpoint" `
    -Force | Out-Null

OK "VoxtralLLM task installed (runs at startup as SYSTEM)"

# ---- 6. VoxtralMic scheduled task (logon, interactive user for audio) -------
Step 6 6 "Installing VoxtralMic scheduled task (at logon, $CURRENT_USER)"

Unregister-ScheduledTask -TaskName "VoxtralMic" -Confirm:$false -ErrorAction SilentlyContinue

$micAction = New-ScheduledTaskAction `
    -Execute          $pythonExe `
    -Argument         "`"$WORKSPACE\tray_app.py`"" `
    -WorkingDirectory $WORKSPACE

$micTrigger = New-ScheduledTaskTrigger -AtLogOn -User $CURRENT_USER

$micSettings = New-ScheduledTaskSettingsSet `
    -RestartCount              5 `
    -RestartInterval           (New-TimeSpan -Minutes 2) `
    -ExecutionTimeLimit        (New-TimeSpan -Hours 0) `
    -RunOnlyIfNetworkAvailable:$false `
    -StartWhenAvailable

$micPrincipal = New-ScheduledTaskPrincipal `
    -UserId    $CURRENT_USER `
    -LogonType Interactive `
    -RunLevel  Highest

Register-ScheduledTask `
    -TaskName    "VoxtralMic" `
    -Action      $micAction `
    -Trigger     $micTrigger `
    -Settings    $micSettings `
    -Principal   $micPrincipal `
    -Description "Voxtral system tray dictation app (Ctrl+Win to dictate)" `
    -Force | Out-Null

OK "VoxtralMic task installed (runs at logon as $CURRENT_USER)"

# ---- Start LLM task now -----------------------------------------------------
Write-Host ""
Write-Host "Starting VoxtralLLM now ..." -ForegroundColor Cyan
Write-Host "    First run will download ~3 GB of model weights." -ForegroundColor Yellow
Start-ScheduledTask -TaskName "VoxtralLLM"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Tasks installed:"
Write-Host "  VoxtralLLM  Runs at boot  (SYSTEM, no user login needed)"
Write-Host "  VoxtralMic  Runs at logon ($CURRENT_USER, interactive audio session)"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Wait for Voxtral model download (~3 GB) — optional if using faster-whisper:"
Write-Host "       Get-Content '$WORKSPACE\logs\llm.log' -Wait"
Write-Host ""
Write-Host "  2. Verify Voxtral server is ready (optional):"
Write-Host "       Invoke-RestMethod http://127.0.0.1:$LLM_PORT/health"
Write-Host ""
Write-Host "  3. Find your DJI mic device:"
Write-Host "       python '$WORKSPACE\list-devices.py'"
Write-Host "     Edit config if pattern doesn't match 'DJI':"
Write-Host "       notepad '$WORKSPACE\config.json'"
Write-Host ""
Write-Host "  4. Start tray app (or log out/back in to auto-trigger):"
Write-Host "       Start-ScheduledTask -TaskName VoxtralMic"
Write-Host "     A mic icon appears in the system tray."
Write-Host ""
Write-Host "  5. Dictate:"
Write-Host "       Hold Ctrl+Win while speaking."
Write-Host "       Release Win — text is transcribed and typed at your cursor."
Write-Host "       Right-click the tray icon to switch backend or download models."
Write-Host ""
Write-Host "  6. Watch the log:"
Write-Host "       Get-Content '$env:LOCALAPPDATA\Voxtral\logs\tray.log' -Wait"
Write-Host ""
Write-Host "Manage:"
Write-Host "  Start-ScheduledTask / Stop-ScheduledTask -TaskName VoxtralLLM"
Write-Host "  Start-ScheduledTask / Stop-ScheduledTask -TaskName VoxtralMic"
Write-Host "  Get-ScheduledTask VoxtralLLM, VoxtralMic | Select TaskName, State"
Write-Host ""
Write-Host "Logs  : $WORKSPACE\logs\"
Write-Host "Output: $WORKSPACE\output\transcription.log"
