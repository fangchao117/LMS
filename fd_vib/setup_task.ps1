# 注册 Windows 定时任务：日盘 15:10 + 夜盘 23:10 更新 30m 缓存 + live_signal
# 以管理员身份运行 PowerShell：
#   cd D:\LMS\fd_vib
#   .\setup_task.ps1

$TaskName = "fd_vib_daily"
$BatPath  = "D:\LMS\fd_vib\daily_job.bat"

if (-not (Test-Path $BatPath)) {
    Write-Error "找不到 $BatPath"
    exit 1
}

# 周一~周五 15:10（日盘收盘后）+ 23:10（夜盘收盘后）
$TriggerDay   = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "15:10"
$TriggerNight = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "23:10"
$Action  = New-ScheduledTaskAction -Execute $BatPath -WorkingDirectory "D:\LMS\fd_vib"
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger @($TriggerDay, $TriggerNight) -Settings $Settings -Force `
    -Description "fd_vib 日盘/夜盘收盘后更新30m缓存并导出live_signal.json"

Write-Host "已创建计划任务: $TaskName"
Write-Host "  时间: 周一至周五 15:10（日盘）+ 23:10（夜盘）"
Write-Host "  脚本: $BatPath"
Write-Host ""
Write-Host "手动测试: python D:\LMS\fd_vib\daily_job.py"
Write-Host "查看日志: D:\LMS\fd_vib\artifacts\daily_job.log"
Write-Host "删除任务: Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
