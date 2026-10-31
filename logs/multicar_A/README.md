# Run Log: multicar_A

- **Branch**: `feature/multicar-demo-phat`
- **Commit**: `f4d8ef9`
- **Warm-start**: `models/ref_111s/best_agent.pt`
- **Vehicle**: `gt3_anchor` | **Lanes**: `3` | **Cars**: `6` | **Obs**: `60D`
- **Smoothness Coefs (§4.2)**: `c1 (steer_smooth_coef) = 1.0`, `c2 (pedal_smooth_coef) = 0.05`, `center_coef = 0.0`
- **Command**:
```powershell
python training/train_multicar.py --total-samples 300000 --preset-smooth A --save-dir models/multicar_A --log-dir logs/multicar_A
```
