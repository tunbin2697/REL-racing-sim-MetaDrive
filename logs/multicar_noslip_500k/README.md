# Run Log: multicar_noslip_500k

- **Branch**: `feature/multicar-demo-phat-bk-finetune`
- **Commit**: `58efc01`
- **Warm-start / Resume**: `models/multicar_5h_finetune/agent_8000k.pt`
- **Vehicle**: `gt3_anchor` | **Lanes**: `3` | **Cars**: `6` | **Obs**: `60D`
- **Smoothness Coefs (§4.2)**: `c1 (steer_smooth_coef) = 0.1`, `c2 (pedal_smooth_coef) = 0.0`, `center_coef = 0.0`
- **Collision Penalty**: `4.0` (with proximity buffer & WI-L5 clean overtake bonus)
- **Command**:
```powershell
python training/train_multicar.py --resume models/multicar_5h_finetune/agent_8000k.pt --num-cars 6 --total-samples 500000 --lr 3e-5 --grid staggered --rollout-steps 512 --batch-size 128 --milestone-freq 100000 --eval-freq 5 --save-dir models/multicar_noslip_500k --log-dir logs/multicar_noslip_500k
```
