# Parameters
param (
    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)


# Configuration
$BusyBoxDir = Join-Path $HOME "busybox"
$BusyBoxExe = Join-Path $BusyBoxDir "busybox.exe"
$Wrapper    = Join-Path $BusyBoxDir "sh.ps1"

# BusyBox Windows official URL
$BusyBoxUrl = "https://frippery.org/files/busybox/busybox64.exe"

# Create the busybox folder if it does not exist
if (-not (Test-Path $BusyBoxDir)) {
    New-Item -ItemType Directory -Path $BusyBoxDir | Out-Null
}

# Download BusyBox only if it is not already installed
if (-not (Test-Path $BusyBoxExe)) {
    Write-Host "Downloading BusyBox..."
    Invoke-WebRequest -Uri $BusyBoxUrl -OutFile $BusyBoxExe
} else {
    Write-Host "BusyBox already installed."
}

# Create the sh.ps1 wrapper
$WrapperContent = @"
& '$BusyBoxExe' sh @args
"@

$WrapperContent | Out-File -FilePath $Wrapper -Encoding utf8

Write-Host "Installation complete."
Write-Host "Launch the shell with: $Wrapper"

# Check the bash script existence
if (-not (Test-Path $ScriptPath)) {
    Write-Error "Script bash introuvable : $ScriptPath"
    exit 1
}

# Run the bash script via le wrapper ---
Write-Host "Run the bash script via BusyBox...`n"

& $Wrapper $ScriptPath @ScriptArgs
