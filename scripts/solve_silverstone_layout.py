import numpy as np
from scipy.optimize import minimize
import json
import matplotlib.pyplot as plt

def get_block_delta(b):
    if b["type"] == "S":
        dx = b["len"]
        dy = 0.0
        d_theta = 0.0
    else:
        r = b["r"]
        ang = np.radians(b["ang"])
        dir_cw = b["dir"]
        flen = b["len"]
        sign = -1.0 if dir_cw == 1 else 1.0
        
        arc_dx = r * np.sin(ang)
        arc_dy = sign * r * (1.0 - np.cos(ang))
        d_theta = sign * ang
        
        str_dx = flen * np.cos(d_theta)
        str_dy = flen * np.sin(d_theta)
        
        dx = arc_dx + str_dx
        dy = arc_dy + str_dy
        
    return dx, dy, d_theta

def build_silverstone(params):
    # Free parameters for optimization:
    (hamilton_len, wellington_len, hangar_len, vale_len, national_len,
     abbey_ang, arena_ang, brooklands_ang, luffield_ang, copse_ang, stowe_ang, club_ang) = params
     
    blocks = [
        # 0: Hamilton Straight (Start / Finish line)
        {"type": "S", "len": float(hamilton_len)},
        # 1: Turn 1 Abbey (Right-hand fast sharp bend)
        {"type": "C", "dir": 1, "r": 75.0, "ang": float(abbey_ang), "len": 12.0},
        # 2: Turn 2 Farm (Sweeping left)
        {"type": "C", "dir": 0, "r": 95.0, "ang": 30.0, "len": 10.0},
        # 3: Turn 3 Village (Sharp right into Arena)
        {"type": "C", "dir": 1, "r": 30.0, "ang": 85.0, "len": 8.0},
        # 4: Turn 4 Arena / The Loop (Tight 180 left hairpin)
        {"type": "C", "dir": 0, "r": 25.0, "ang": float(arena_ang), "len": 10.0},
        # 5: Turn 5 Aintree (Sharp left onto Wellington Straight)
        {"type": "C", "dir": 0, "r": 60.0, "ang": 45.0, "len": 12.0},
        # 6: Wellington Straight (Flat out straight)
        {"type": "S", "len": float(wellington_len)},
        # 7: Turn 6 Brooklands (Sharp left entry)
        {"type": "C", "dir": 0, "r": 45.0, "ang": float(brooklands_ang), "len": 10.0},
        # 8: Turn 7 Luffield (Right-hand carousel hairpin)
        {"type": "C", "dir": 1, "r": 36.0, "ang": float(luffield_ang), "len": 10.0},
        # 9: Turn 8 Woodcote (Sweeping right onto National Straight)
        {"type": "C", "dir": 1, "r": 85.0, "ang": 35.0, "len": 12.0},
        # 10: National Pit Straight
        {"type": "S", "len": float(national_len)},
        # 11: Turn 9 Copse (Fast 90-degree right)
        {"type": "C", "dir": 1, "r": 65.0, "ang": float(copse_ang), "len": 20.0},
        # 12: Turn 10 Maggots 1 (Left flick)
        {"type": "C", "dir": 0, "r": 75.0, "ang": 35.0, "len": 8.0},
        # 13: Turn 11 Maggots 2 (Right flick)
        {"type": "C", "dir": 1, "r": 60.0, "ang": 55.0, "len": 8.0},
        # 14: Turn 12 Becketts 1 (Left)
        {"type": "C", "dir": 0, "r": 45.0, "ang": 75.0, "len": 8.0},
        # 15: Turn 13 Becketts 2 (Right)
        {"type": "C", "dir": 1, "r": 38.0, "ang": 90.0, "len": 8.0},
        # 16: Turn 14 Chapel (Left opening onto Hangar)
        {"type": "C", "dir": 0, "r": 70.0, "ang": 42.0, "len": 15.0},
        # 17: Hangar Straight (Longest flat-out straight)
        {"type": "S", "len": float(hangar_len)},
        # 18: Turn 15 Stowe (Heavy braking right turn)
        {"type": "C", "dir": 1, "r": 50.0, "ang": float(stowe_ang), "len": 15.0},
        # 19: Vale Straight
        {"type": "S", "len": float(vale_len)},
        # 20: Turn 16 Vale (Sharp chicane left turn)
        {"type": "C", "dir": 0, "r": 28.0, "ang": 85.0, "len": 8.0},
        # 21: Turn 17 Club Entry (Right)
        {"type": "C", "dir": 1, "r": 42.0, "ang": 75.0, "len": 8.0},
        # 22: Turn 18 Club Exit (Right back onto Hamilton Straight)
        {"type": "C", "dir": 1, "r": 45.0, "ang": float(club_ang), "len": 15.0},
    ]
    
    x, y, heading = 0.0, 0.0, 0.0
    pts = [[x, y]]
    
    for b in blocks:
        loc_dx, loc_dy, d_theta = get_block_delta(b)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        x += cos_h * loc_dx - sin_h * loc_dy
        y += sin_h * loc_dx + cos_h * loc_dy
        heading += d_theta
        pts.append([x, y])
        
    return np.array(pts), x, y, heading, blocks

def total_turn_constraint(params):
    _, _, _, end_heading, _ = build_silverstone(params)
    return -np.degrees(end_heading) - 360.0

def objective(params):
    _, end_x, end_y, _, _ = build_silverstone(params)
    return end_x**2 + end_y**2

init_params = [
    240.0, # hamilton
    350.0, # wellington
    550.0, # hangar
    110.0, # vale
    260.0, # national
    50.0,  # abbey
    160.0, # arena
    100.0, # brooklands
    180.0, # luffield
    88.0,  # copse
    105.0, # stowe
    85.0,  # club
]

bounds = [
    (150.0, 350.0), # hamilton
    (250.0, 480.0), # wellington
    (400.0, 700.0), # hangar
    (60.0, 180.0),  # vale
    (180.0, 350.0), # national
    (35.0, 65.0),   # abbey
    (140.0, 180.0), # arena
    (80.0, 120.0),  # brooklands
    (160.0, 200.0), # luffield
    (75.0, 105.0),  # copse
    (90.0, 120.0),  # stowe
    (70.0, 105.0),  # club
]

constraints = [{'type': 'eq', 'fun': total_turn_constraint}]

res = minimize(objective, init_params, bounds=bounds, constraints=constraints, method='SLSQP', options={'maxiter': 500, 'ftol': 1e-12})
print("Solver Success:", res.success)
pts, ex, ey, eh, blocks = build_silverstone(res.x)
pos_err = float(np.sqrt(ex**2 + ey**2))
head_err = float(np.degrees(eh + 2*np.pi))
print(f"End Pos Error: {pos_err:.6f} m, Heading Error: {head_err:.6f} deg")

total_len = 0.0
for b in blocks:
    if b["type"] == "S":
        total_len += b["len"]
    else:
        total_len += b["r"] * np.radians(b["ang"]) + b["len"]
print(f"Total Silverstone Length: {total_len:.1f} meters")

with open("scripts/silverstone_blocks.json", "w") as f:
    json.dump({
        "total_length": total_len,
        "closure_error_m": pos_err,
        "heading_error_deg": head_err,
        "blocks": blocks
    }, f, indent=2)

print("Saved scripts/silverstone_blocks.json!")
