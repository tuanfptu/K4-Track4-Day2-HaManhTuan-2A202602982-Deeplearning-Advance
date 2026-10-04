#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
submission_dir="$repo_dir/submissions/2A202602982_HaManhTuan"
data_dir="${DEEPWEEDS_DATA:-$repo_dir/data}"
output_dir="${DEEPWEEDS_OUTPUT:-$repo_dir/lab_output}"
python_cmd="${DEEPWEEDS_PYTHON:-python3}"
if ! command -v "$python_cmd" >/dev/null; then
  echo "Python executable not found: $python_cmd" >&2
  exit 1
fi
"$python_cmd" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print(torch.cuda.get_device_name(0))'
"$python_cmd" "$submission_dir/code/prepare_data.py" --data "$data_dir"
export DEEPWEEDS_DATA="$data_dir"
"$python_cmd" -m unittest discover -s "$submission_dir/code" -p 'test_*.py' -q
"$python_cmd" "$submission_dir/code/run_lab.py" smoke --data "$data_dir" --output "$output_dir/smoke"
"$python_cmd" "$submission_dir/code/run_lab.py" full --data "$data_dir" --output "$output_dir/full" \
  --epochs "${DEEPWEEDS_EPOCHS:-10}" --final-epochs "${DEEPWEEDS_FINAL_EPOCHS:-12}"
