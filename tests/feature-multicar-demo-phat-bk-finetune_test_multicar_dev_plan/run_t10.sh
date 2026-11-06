#!/bin/bash
# PY=đường_dẫn_python bash run_t10.sh  (cần results/ref_profile.npz từ "t8_mpc.py profile")
cd "$(dirname "$0")"
PY="${PY:-python}"
for c in c0_base c1_grid c2_grid_map c3_attack c4_attack_close c5_close_noattack c6_attack_nomap; do
  echo "$c pack 12 none"; echo "$c pack 12 smart"; echo "$c duel"
done | xargs -P 8 -I{} bash -c '"'"$PY"'" -u t10_mpc_options.py {} > /dev/null 2>&1 </dev/null; echo "{}: exit $?"'
echo ALL_DONE
