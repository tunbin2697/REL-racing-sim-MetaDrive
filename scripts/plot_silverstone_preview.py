import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.solve_silverstone_layout import get_block_delta

with open('scripts/silverstone_blocks.json') as f:
    data = json.load(f)

blocks = data['blocks']
x, y, heading = 0.0, 0.0, 0.0
pts = [[x, y]]

labels = {
    0: 'Start/Finish',
    1: 'T1 Abbey',
    2: 'T2 Farm',
    3: 'T3 Village',
    4: 'T4 The Loop',
    5: 'T5 Aintree',
    6: 'Wellington Straight',
    7: 'T6 Brooklands',
    8: 'T7 Luffield',
    9: 'T8 Woodcote',
    10: 'National Pit Straight',
    11: 'T9 Copse',
    12: 'T10 Maggots',
    13: 'T11 Maggots 2',
    14: 'T12 Becketts',
    15: 'T13 Becketts 2',
    16: 'T14 Chapel',
    17: 'Hangar Straight',
    18: 'T15 Stowe',
    19: 'Vale Straight',
    20: 'T16 Vale',
    21: 'T17 Club',
    22: 'T18 Club Exit',
}

plt.figure(figsize=(11, 9))
labeled_pts = []
for i, b in enumerate(blocks):
    loc_dx, loc_dy, d_theta = get_block_delta(b)
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    x += cos_h * loc_dx - sin_h * loc_dy
    y += sin_h * loc_dx + cos_h * loc_dy
    heading += d_theta
    pts.append([x, y])
    if i in labels:
        labeled_pts.append((i, x, y, labels[i]))

pts = np.array(pts)
plt.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=3, label='Silverstone Circuit Track')
plt.scatter([pts[0, 0]], [pts[0, 1]], color='green', s=140, zorder=5, label='Start / Finish')

for idx, lx, ly, name in labeled_pts:
    plt.annotate(name, (lx, ly), fontsize=8, weight='bold', alpha=0.85)

plt.title(f'Silverstone Grand Prix Circuit (MetaDrive 3D) - Length: {data["total_length"]:.1f}m')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('plots/silverstone_circuit_preview.png', dpi=150)
print('Saved plots/silverstone_circuit_preview.png successfully!')
