"""
변환된 .pt 파일의 통계 + tracking error plot.

usage:
    python scripts/data_check.py --file data/611/logs/300Nm_120_0.00007_input_position_data.pt
    python scripts/data_check.py --file ... --motor_type 300 --data_type input

--file 만 주면 파일명에서 motor_type / data_type 을 자동 추론한다.
"""

import argparse
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


MOTOR_TYPES = [20, 100, 300]
DATA_TYPES = ["input", "output"]


def _running_in_ipykernel() -> bool:
    return "ipykernel" in pathlib.Path(sys.argv[0]).name


def _infer_motor_type(filename: str) -> int | None:
    """파일명에서 'NNNm' 패턴을 찾아 motor_type 추출."""
    m = re.search(r"(\d+)Nm", filename)
    if m is None:
        return None
    val = int(m.group(1))
    return val if val in MOTOR_TYPES else None


def _infer_data_type(filename: str) -> str | None:
    """파일명에서 'input' / 'output' 키워드 추출."""
    name = filename.lower()
    if "input_position_data" in name:
        return "input"
    if "output_position_data" in name:
        return "output"
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a converted .pt file (stats + tracking-error plot)."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to a .pt file (예: data/611/logs/300Nm_120_0.00007_input_position_data.pt).",
    )
    parser.add_argument(
        "--motor_type",
        type=int,
        choices=MOTOR_TYPES,
        default=None,
        help="Motor type (20/100/300). 미지정 시 파일명에서 추론.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=DATA_TYPES,
        default=None,
        help="input / output. 미지정 시 파일명에서 추론.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="plt.show() 생략 (headless).",
    )
    return parser.parse_args()


args = _parse_args()
pt_path = pathlib.Path(args.file).expanduser().resolve()
if not pt_path.exists():
    raise FileNotFoundError(f".pt not found: {pt_path}")

# motor_type / data_type 자동 추론
motor_type = args.motor_type if args.motor_type is not None else _infer_motor_type(pt_path.name)
data_type = args.data_type if args.data_type is not None else _infer_data_type(pt_path.name)

motor_label = f"{motor_type}Nm" if motor_type else "?Nm"
data_label = data_type if data_type else "?"

# %%
data = torch.load(pt_path, map_location="cpu")

time = data["time"].cpu().numpy() if torch.is_tensor(data["time"]) else np.asarray(data["time"])

def _squeeze_last(arr_or_tensor):
    if torch.is_tensor(arr_or_tensor):
        arr = arr_or_tensor.cpu().numpy()
    else:
        arr = np.asarray(arr_or_tensor)
    if arr.ndim >= 2 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return arr

dof_pos = _squeeze_last(data["dof_pos"])
des_dof_pos = _squeeze_last(data["des_dof_pos"])
joint_names = data.get("joint_names", [f"joint{i}" for i in range(dof_pos.shape[1] if dof_pos.ndim > 1 else 1)])

print(f"[INFO] loaded {pt_path}")
print(f"  motor_type  : {motor_label}  (inferred from filename)" if args.motor_type is None and motor_type else f"  motor_type  : {motor_label}")
print(f"  data_type   : {data_label}")
print(f"  joint_names : {joint_names}")
print(f"  samples     : {len(time)}")
print(
    f"  time range  : {time[0]:.4f} ~ {time[-1]:.4f} s  (dt avg ≈ {np.mean(np.diff(time)):.5f} s)"
)
print(
    f"  dof_pos     : range [{dof_pos.min():+.4f}, {dof_pos.max():+.4f}] rad, "
    f"amp ≈ ±{0.5 * (dof_pos.max() - dof_pos.min()):.4f} rad"
)
print(
    f"  des_dof_pos : range [{des_dof_pos.min():+.4f}, {des_dof_pos.max():+.4f}] rad, "
    f"amp ≈ ±{0.5 * (des_dof_pos.max() - des_dof_pos.min()):.4f} rad"
)

mae = float(np.abs(dof_pos - des_dof_pos).mean())
rmse = float(np.sqrt(np.mean((dof_pos - des_dof_pos) ** 2)))
max_error = float(np.abs(dof_pos - des_dof_pos).max())
print(f"  tracking    : MAE = {mae:.4f} rad,  RMSE = {rmse:.4f} rad")
print(f"  max error   : {max_error:.4f} rad")

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax = axes[0]
ax.plot(time, des_dof_pos, color="blue", linewidth=1.2, label="des_dof_pos")
ax.plot(time, dof_pos, color="red", linewidth=1.0, alpha=0.8, label="dof_pos")
ax.set_title(
    f"{motor_label} {data_label} position  |  {pt_path.name}  |  "
    f"MAE = {mae:.4f} rad, RMSE = {rmse:.4f} rad"
)
ax.set_ylabel("position (rad)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(
    time,
    dof_pos - des_dof_pos,
    color="purple",
    linewidth=1.0,
    alpha=0.8,
    label="dof_pos - des_dof_pos",
)
ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Time (sec)")
ax.set_ylabel("tracking error (rad)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()

if not args.no_show:
    plt.show()
