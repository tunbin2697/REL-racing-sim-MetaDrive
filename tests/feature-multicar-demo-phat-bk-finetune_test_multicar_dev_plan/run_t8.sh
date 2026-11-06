#!/bin/bash
# Chạy các bài đo adapter MPC song song, tối đa 5 tiến trình. Cần chạy "t8_mpc.py profile" trước.
# PY=đường_dẫn_python bash run_t8.sh
cd "$(dirname "$0")"
PY="${PY:-python}"
mkdir -p results/console
printf '%s\n' \
  "solo 12" \
  "pack ref 24 none" \
  "pack mpc 24 none" \
  "pack ref 12 brake50" \
  "pack mpc 12 brake50" \
  "duel ref" \
  "duel mpc" |
xargs -P 5 -I{} bash -c 'name=$(echo "{}" | tr " " "_"); '"\"$PY\""' -u t8_mpc.py {} > "results/console/t8_$name.txt" 2>&1 </dev/null; echo "{}: exit $?"'
echo ALL_DONE
