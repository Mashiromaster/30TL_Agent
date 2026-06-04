# create_shortcut.ps1 — 在桌面生成启动 Dashboard 的快捷方式
# 使用: 在项目根双击或在 PowerShell 中执行：
# powershell -ExecutionPolicy Bypass -File .\create_shortcut.ps1

$projectDir = $PSScriptRoot
$targetDir = Join-Path $projectDir "src"
$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "F_Agent Dashboard.lnk"

$cmd = Join-Path $env:SystemRoot "System32\cmd.exe"
$args = "/c cd /d `"$targetDir`" && python -m streamlit run dashboard.py"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($lnkPath)
$Shortcut.TargetPath = $cmd
$Shortcut.Arguments = $args
$Shortcut.WorkingDirectory = $targetDir
$Shortcut.IconLocation = "$env:SystemRoot\system32\shell32.dll, 1"
$Shortcut.Description = "启动 TL 策略 Dashboard"
$Shortcut.Save()

Write-Output "快捷方式已创建: $lnkPath"
