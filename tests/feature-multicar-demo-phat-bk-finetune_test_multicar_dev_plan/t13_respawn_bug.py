"""T13. Vì sao có race bị kẹt vòng lặp hồi sinh? Đưa xe vào tình trạng hỏng, gọi respawn_car rồi xem xe có văng lại ngay không.
So sánh respawn_car hiện tại với cách sửa thử: đặt lại cả độ cao và góc lật ngay sau khi hồi sinh.
    python t13_respawn_bug.py
"""
import json

import numpy as np

from common import make_env, save

env = make_env(num_cars=2)
rows = []
CASES = {
    "roi_xuong_thap": lambda o: o.setZ(-1.0),
    "lat_ngang_60do": lambda o: o.setR(60.0),
    "chuc_dau_60do": lambda o: o.setP(60.0),
    "khong_hong": lambda o: None,
}


def fix_pose(raw):
    raw.origin.setZ(0.6)
    raw.origin.setP(0.0)
    raw.origin.setR(0.0)


for case, damage in CASES.items():
    for fix in (False, True):
        env.reset()
        actions = np.tile(np.array([0.0, 0.8, -1.0], dtype=np.float32), (2, 1))
        for _ in range(100):
            env.step(actions)
        st = env.racers[1]
        damage(st.raw_vehicle.origin)
        _, _, dones, _, _ = env.step(actions)
        before = {"done_after_damage": bool(dones[1])}
        if not dones[1]:
            st.active = True
        loops, survived = 0, 0
        for _ in range(60):
            if not st.active or loops == 0:
                env.respawn_car(1)
                if fix:
                    fix_pose(st.raw_vehicle)
            _, _, dones, _, _ = env.step(actions)
            if dones[1]:
                loops += 1
            else:
                survived += 1
                if survived >= 30:
                    break
        hpr, z = st.raw_vehicle.origin.get_hpr(), st.raw_vehicle.origin.get_pos()[2]
        rows.append({"case": case, "fix": fix, **before, "done_again": loops, "steps_ok": survived, "z": round(float(z), 2), "pitch": round(float(hpr[1]), 1), "roll": round(float(hpr[2]), 1)})
        print(rows[-1], flush=True)
env.close()
save("t13_respawn", {"rows": rows})
