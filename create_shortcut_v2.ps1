$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut("D:\桌面\F_Agent_TL策略.lnk")
$Shortcut.TargetPath = "D:\桌面\F_Agent\launch_dashboard.bat"
$Shortcut.WorkingDirectory = "D:\桌面\F_Agent"
$Shortcut.Description = "F_Agent TL 30Y Dashboard (port 8503)"
$Shortcut.Save()
Write-Host "OK"
