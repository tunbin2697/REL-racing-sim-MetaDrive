# Ablation một biến: công thức ref_111s (code ee99f9d) BỎ r_center, 2 seed song song.
# Worktree ..\REL_111s_nocenter (detached ee99f9d) đã sửa đúng 1 dòng trong game_env/racing_env.py:
#     r_center = -0.04 * (lat/4.5)^2   ->   r_center = 0.0
# Đối chứng: log gốc seed 42 (git show ee99f9d:logs/training_metrics.json) và seed 43
# (logs/anchor/seed43_training_metrics.json). Sau khi xong: eval best dưới luật 4 bánh với
#   python training/evaluate.py --model <best> --map silverstone --vehicle gt3_anchor --episodes 20 --laps 2
# Không có chữ đỏ: không gom stderr của MetaDrive (log [INFO]) vào pipeline.
$py = "$env:USERPROFILE\miniconda3\envs\rel_racing\python.exe"
$wt = "C:\Users\tunbi\Project\school\REL_111s_nocenter"

foreach ($seed in 42, 43) {
    $cmd = "Set-Location '$wt'; `$env:PYTHONIOENCODING='utf-8'; & '$py' training/train_racing.py --total-steps 4000000 --seed $seed --save-dir models/nocenter_s$seed --log-dir logs/nocenter_s$seed --tb-logdir logs/nocenter_s$seed/tb"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd
}
Write-Host ">> Đã mở 2 cửa sổ (seed 42, 43). Log: $wt\logs\nocenter_s42\ và nocenter_s43\"
