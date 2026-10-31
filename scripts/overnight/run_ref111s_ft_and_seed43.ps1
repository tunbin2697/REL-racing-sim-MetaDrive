# Chạy qua đêm 31/10 (KHÔNG commit). Mở 2 cửa sổ PowerShell chạy song song:
#  (1) Fine-tune ref_111s trên env chặt (luật 4 bánh), cờ reward khớp lúc train 111s, 1M step
#  (2) Lặp lại công thức 111s gốc với seed 43 (code ee99f9d, worktree riêng), 4M step
# Mỗi run tự dừng khi đủ step. Log console ghi vào file .log bên cạnh.
$py   = "$env:USERPROFILE\miniconda3\envs\rel_racing\python.exe"
$main = "C:\Users\tunbi\Project\school\REL Assignment"
$wt   = "C:\Users\tunbi\Project\school\REL_111s_seed43"

# (1) Fine-tune
New-Item -ItemType Directory -Force "$main\models\ref_111s_ft_strict", "$main\logs\ref_111s_ft_strict" | Out-Null
Copy-Item "$main\models\ref_111s\best_agent.pt" "$main\models\ref_111s_ft_strict\start_ref_111s.pt" -Force
$ft = "Set-Location '$main'; `$env:PYTHONIOENCODING='utf-8'; & '$py' training/train_racing.py --resume models/ref_111s_ft_strict/start_ref_111s.pt --steps-to-run 1000000 --map silverstone --vehicle gt3_anchor --max-laps 1 --obs-mode pure --action-mode split --offtrack-mode terminate --spawn-mode grid --center-coef 0.04 --time-penalty 0 --no-logstd-projection --ent-coef 0.01 --lr 3e-4 --eval-freq 5 --eval-episodes 3 --save-dir models/ref_111s_ft_strict --log-dir logs/ref_111s_ft_strict --tb-logdir logs/ref_111s_ft_strict/tb *>&1 | Tee-Object -FilePath logs/ref_111s_ft_strict/console.log"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $ft

# (2) Seed 43
$s43 = "Set-Location '$wt'; `$env:PYTHONIOENCODING='utf-8'; & '$py' training/train_racing.py --total-steps 4000000 --seed 43 --save-dir models/seed43 --log-dir logs/seed43 --tb-logdir logs/seed43/tb *>&1 | Tee-Object -FilePath logs/seed43_console.log"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $s43

Write-Host ">> Đã mở 2 cửa sổ. Sáng mai gọi trợ lý đọc:"
Write-Host "   $main\logs\ref_111s_ft_strict\training_metrics.json"
Write-Host "   $wt\logs\seed43\training_metrics.json"
