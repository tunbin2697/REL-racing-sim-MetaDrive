# Run Log: multicar_5h_finetune

- **Branch**: `feature/multicar-demo-phat-bk-finetune`
- **Commit**: `58efc01`
- **Warm-start / Resume**: `models/multicar_final_1M/best_agent.pt`
- **Vehicle**: `gt3_anchor` | **Lanes**: `3` | **Cars**: `6` | **Obs**: `60D`
- **Smoothness Coefs (§4.2)**: `c1 (steer_smooth_coef) = 0.1`, `c2 (pedal_smooth_coef) = 0.0`, `center_coef = 0.0`
- **Collision Penalty**: `4.0` (with proximity buffer & WI-L5 clean overtake bonus)
- **Command**:
```powershell
python training/train_multicar.py --resume models/multicar_final_1M/best_agent.pt --num-cars 6 --total-samples 8000000 --lr 3e-5 --grid staggered --rollout-steps 512 --batch-size 128 --milestone-freq 1000000 --eval-freq 10 --save-dir models/multicar_5h_finetune --log-dir logs/multicar_5h_finetune
```
