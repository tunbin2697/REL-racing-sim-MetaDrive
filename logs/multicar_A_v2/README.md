# Run Log: multicar_A_v2

- **Branch**: `feature/multicar-demo-phat`
- **Commit**: `f4d8ef9`
- **Warm-start / Resume**: `models/multicar_A/best_agent.pt`
- **Vehicle**: `gt3_anchor` | **Lanes**: `3` | **Cars**: `6` | **Obs**: `60D`
- **Smoothness Coefs (§4.2)**: `c1 (steer_smooth_coef) = 1.0`, `c2 (pedal_smooth_coef) = 0.05`, `center_coef = 0.0`
- **Collision Penalty**: `4.0` (with proximity buffer & WI-L5 clean overtake bonus)
- **Command**:
```powershell
python training/train_multicar.py --resume models/multicar_A/best_agent.pt --total-samples 300000 --preset-smooth A --collision-penalty 4.0 --save-dir models/multicar_A_v2 --log-dir logs/multicar_A_v2
```
