# Run Log: multicar_nof16_111s

- **Branch**: `feature/multicar-demo-phat-bk-finetune`
- **Commit**: `58efc01`
- **Warm-start / Resume**: `models/ref_111s/best_agent.pt`
- **Vehicle**: `gt3_anchor` | **Lanes**: `3` | **Cars**: `6` | **Obs**: `60D`
- **Smoothness Coefs (§4.2)**: `c1 (steer_smooth_coef) = 0.1`, `c2 (pedal_smooth_coef) = 0.0`, `center_coef = 0.0`
- **Collision Penalty**: `4.0` (with proximity buffer & WI-L5 clean overtake bonus)
- **Command**:
```powershell
python training/train_multicar.py --init-model models/ref_111s/best_agent.pt --num-cars 6 --lane-num 3 --grid staggered --total-samples 3000000 --lr 5e-5 --rollout-steps 512 --batch-size 128 --milestone-freq 200000 --eval-freq 10 --save-dir models/multicar_nof16_111s --log-dir logs/multicar_nof16_111s
```
