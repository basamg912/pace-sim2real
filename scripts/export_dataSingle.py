# %%
import argparse
import glob
import math
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# 터미널 실행 예:
#   # 폴더 → 안의 모든 csv 를 "각각 별도로" 변환 (batch)
#   python scripts/export_dataSingle.py --input_folder data/611/logs --motor_type 300
#   # 단일 파일
#   python scripts/export_dataSingle.py --input_folder data/611/logs/120_0.00007.csv --motor_type 300
#
# 출력 파일명 규칙:
#   입력 csv 파일명(stem)을 그대로 사용해 결정한다.
#     <input_csv_stem>.csv  →  {motor_type}Nm_<input_csv_stem>_input_position_data.pt
#   예) logs/120_0.00007.csv  (motor_type=300)
#        → logs/300Nm_120_0.00007_input_position_data.pt
#   (data/611/logs 의 파일명 규칙: "동작범위_주기당증가Hz.csv")
#
# 지원 CSV 포맷:
#   1) Elmo Chart format ([Signals Data Group 1] 헤더 후 데이터)
#        cols: Time (sec), {motor}Nm_HGI.Position [cnt], ...Command [cnt], OutputEncoder.Position [cnt], Index
#   2) Plain CSV format (헤더 첫 줄, data/611/ 패턴)
#        cols: Timestamp, TargetPos_0, ActualPos_0, TargetVel_0, ActualVel_0,
#              TargetTor_0, ActualTor_0, Mode_0
#        → OutputEncoder 정보 없음, motor-side input 만 export
NOTEBOOK_INPUT_FOLDER = "/home/kist/work/workspace/pace-humanoid/data/611/logs"
NOTEBOOK_MOTOR_TYPE = 300
NOTEBOOK_SAVE_OUTPUT_REFERENCE = False


def _running_in_ipykernel():
    return "ipykernel" in pathlib.Path(sys.argv[0]).name


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to a CSV file or CSV folder. 폴더면 안의 *.csv 를 각각 별도로 변환.",
    )
    parser.add_argument(
        "--motor_type",
        type=int,
        required=True,
        choices=[300, 100, 20],
        help="300, 100, 20",
    )
    parser.add_argument(
        "--save_output_reference",
        action="store_true",
        help=(
            "Also save an output-side reference .pt. This is not the default because "
            "the CSV does not log a true output-side desired position. "
            "Plain CSV (data/611/) format 에는 OutputEncoder 가 없어 무시됨."
        ),
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="설정 시 plot 창 표시 생략 (png 저장만). batch(폴더)에서는 자동 생략.",
    )
    return parser.parse_args()


if _running_in_ipykernel() and "--input_folder" not in sys.argv:
    if NOTEBOOK_INPUT_FOLDER is None or NOTEBOOK_MOTOR_TYPE is None:
        raise RuntimeError(
            "Jupyter에서 실행할 때는 NOTEBOOK_INPUT_FOLDER와 "
            "NOTEBOOK_MOTOR_TYPE을 먼저 설정하세요."
        )
    input_path = pathlib.Path(NOTEBOOK_INPUT_FOLDER)
    motor_type = int(NOTEBOOK_MOTOR_TYPE)
    save_output_reference = bool(NOTEBOOK_SAVE_OUTPUT_REFERENCE)
    no_show = False
    if motor_type not in [300, 100, 20]:
        raise ValueError("NOTEBOOK_MOTOR_TYPE must be one of: 300, 100, 20")
else:
    args = _parse_args()
    input_path = pathlib.Path(args.input_folder)
    motor_type = args.motor_type
    save_output_reference = args.save_output_reference
    no_show = args.no_show


GEAR_RATIOS = {20: 26.0, 100: 23.04, 300: 23.04}
GEAR_RATIO = GEAR_RATIOS[motor_type]
NOMINAL_ROTOR_INERTIAS = {
    20: 3.305e-6,
    100: 1.60e-4,
    300: 5.29e-4,
}
CNT_PER_REV = 8192

# fit_actuator.py 는 데이터를 "인덱스" 기준으로 sim 에 매핑하고 sim.dt 간격으로 재생한다
# (data["time"] 은 stepping 에 안 쓰임). 데이터 rate 가 sim.dt 와 다르면 궤적이
# 빠르게/느리게 재생되어 armature·friction 추정이 체계적으로 편향된다
# (예: 333Hz 데이터를 400Hz 로 재생 → α=1.2, armature 가 α²=1.44 배 저평가).
# 또한 원본 로깅에는 지터(불균일 dt)가 있을 수 있다.
# → export 단계에서 데이터를 SIM_HZ 균일 그리드로 보간하여 두 문제를 모두 제거한다.
# SIM_HZ 는 반드시 사용하는 task 의 sim.dt 와 일치시킬 것:
#   kapex_act_env_cfg.py: self.sim.dt = 0.0025  →  400 Hz
SIM_HZ = 400.0

cnt_to_rad = 2.0 * math.pi / CNT_PER_REV


# ========== Format detection ==========
def detect_csv_format(path: str) -> str:
    """Elmo Chart vs plain CSV 판별.

    Elmo Chart: 어딘가 '[Signals Data Group 1]' 헤더 존재
    Plain:      첫 줄에 'Timestamp,TargetPos_0,...' 같은 컬럼명
    """
    with open(path, "r", errors="ignore") as f:
        head = f.read(4096)
    if "[Signals Data Group 1]" in head:
        return "elmo"
    first_line = head.splitlines()[0] if head else ""
    if (
        "Timestamp" in first_line
        and "TargetPos" in first_line
        and "ActualPos" in first_line
    ):
        return "plain"
    raise ValueError(f"Unknown CSV format: {path}\nFirst line: {first_line!r}")


# ========== Elmo Chart loader ==========
ELMO_TIME = "Time (sec)"
ELMO_INPUT_ENC_POS_ACT = f"{motor_type}Nm_HGI.Position [cnt]"
ELMO_INPUT_ENC_POS_CMD = f"{motor_type}Nm_HGI.Position Command [cnt]"
ELMO_OUTPUT_ENC_POS_ACT = "OutputEncoder.Position [cnt]"
ELMO_INDEX_COL = "Index"

ELMO_COLS = [
    ELMO_TIME,
    ELMO_INPUT_ENC_POS_ACT,
    ELMO_INPUT_ENC_POS_CMD,
    ELMO_OUTPUT_ENC_POS_ACT,
    ELMO_INDEX_COL,
]


def load_elmo_csv(path):
    """Elmo chart CSV에서 [Signals Data Group 1] 이후 numeric data를 읽는다."""
    with open(path, "r") as f:
        lines = f.readlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("[Signals Data Group 1]")
    )
    df = pd.read_csv(
        path,
        skiprows=start + 2,
        names=ELMO_COLS,
        usecols=range(len(ELMO_COLS)),
        index_col=False,
    )
    if ELMO_INDEX_COL in df.columns:
        df = df.set_index(ELMO_INDEX_COL)
    return df


# ========== Plain CSV loader (data/611/...) ==========
PLAIN_TIME = "Timestamp"
PLAIN_POS_CMD = "TargetPos_0"
PLAIN_POS_ACT = "ActualPos_0"


def load_plain_csv(path):
    """첫 줄에 컬럼 헤더가 있는 일반 CSV. 끝에 trailing comma 가 있어
    빈 컬럼이 생기는 경우 자동 제거."""
    df = pd.read_csv(path)
    # trailing 빈 컬럼 (이름이 'Unnamed: N' 형태) 제거
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    needed = [PLAIN_TIME, PLAIN_POS_CMD, PLAIN_POS_ACT]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Plain CSV missing columns: {missing}. Got columns: {list(df.columns)}"
        )
    return df


# ========== 출력 파일명 규칙 (입력 csv 이름 기반) ==========
def output_paths(csv_path, save_dir):
    """입력 csv 파일명(stem) 으로 출력 .pt / .png 경로를 결정한다.

    <stem>.csv → {motor_type}Nm_<stem>_input_position_data.pt
    """
    stem = pathlib.Path(csv_path).stem.strip()
    base = f"{motor_type}Nm_{stem}"
    return (
        save_dir / f"{base}_input_position_data.pt",
        save_dir / f"{base}_output_position_data.pt",
        save_dir / f"{base}_export.png",
        base,
    )


# ========== 단일 csv → 단일 pt 변환 ==========
def convert_one(fp, save_dir, show_plot):
    """csv 파일 1개를 로드·보간하여 pt 1개(+ png)로 저장한다."""
    name = os.path.basename(fp)
    fmt = detect_csv_format(fp)
    print(f"[FILE] {name}: format={fmt}")

    if fmt == "elmo":
        df = load_elmo_csv(fp)
        local_time = df[ELMO_TIME].to_numpy(dtype=float)
        local_time = local_time - local_time[0]
        input_cmd_rel_cnt = (
            df[ELMO_INPUT_ENC_POS_CMD].to_numpy(dtype=float)
            - df[ELMO_INPUT_ENC_POS_CMD].iloc[0]
        )
        input_act_rel_cnt = (
            df[ELMO_INPUT_ENC_POS_ACT].to_numpy(dtype=float)
            - df[ELMO_INPUT_ENC_POS_ACT].iloc[0]
        )
        output_act_rel_cnt = (
            df[ELMO_OUTPUT_ENC_POS_ACT].to_numpy(dtype=float)
            - df[ELMO_OUTPUT_ENC_POS_ACT].iloc[0]
        )
        has_output_encoder = True
    elif fmt == "plain":
        df = load_plain_csv(fp)
        local_time = df[PLAIN_TIME].to_numpy(dtype=float)
        local_time = local_time - local_time[0]
        input_cmd_rel_cnt = (
            df[PLAIN_POS_CMD].to_numpy(dtype=float) - df[PLAIN_POS_CMD].iloc[0]
        )
        input_act_rel_cnt = (
            df[PLAIN_POS_ACT].to_numpy(dtype=float) - df[PLAIN_POS_ACT].iloc[0]
        )
        output_act_rel_cnt = np.zeros_like(input_act_rel_cnt)  # placeholder
        has_output_encoder = False
    else:
        raise ValueError(f"Unhandled format: {fmt}")

    if len(local_time) < 2:
        print(f"  {name}: skipped (<2 rows)")
        return

    # cnt → rad
    motor_cmd_rad = input_cmd_rel_cnt * cnt_to_rad
    motor_actual_rad = input_act_rel_cnt * cnt_to_rad
    output_actual_rad = output_act_rel_cnt * cnt_to_rad  # plain 에선 0 vector

    src_dt = float(local_time[1] - local_time[0])
    src_hz = (len(local_time) - 1) / (local_time[-1] - local_time[0])
    print(
        f"    raw rows={len(local_time)}, dur={local_time[-1]:.4f}s, "
        f"first dt≈{src_dt * 1000:.3f}ms, avg {src_hz:.1f} Hz"
    )

    # ===== SIM_HZ 균일 그리드로 보간 =====
    t_uniform = np.arange(0.0, local_time[-1], 1.0 / SIM_HZ)
    motor_cmd_rad = np.interp(t_uniform, local_time, motor_cmd_rad)
    motor_actual_rad = np.interp(t_uniform, local_time, motor_actual_rad)
    output_actual_rad = np.interp(t_uniform, local_time, output_actual_rad)
    output_cmd = motor_cmd_rad / GEAR_RATIO
    print(
        f"    resampled {src_hz:.1f}Hz → {SIM_HZ:.0f}Hz: {len(t_uniform)} rows "
        f"(dt={1000.0 / SIM_HZ:.3f}ms)"
    )

    input_pt, output_pt, png_path, base = output_paths(fp, save_dir)

    # ----- plot -----
    n_plots = 2 if has_output_encoder else 1
    fig, axes = plt.subplots(
        n_plots, 1, figsize=(14, 4 * n_plots), sharex=True, squeeze=False
    )
    axes = axes.flatten()

    ax = axes[0]
    ax.plot(t_uniform, motor_cmd_rad, color="blue", label="motor command")
    ax.plot(t_uniform, motor_actual_rad, color="red", alpha=0.8, label="input encoder")
    in_mae = float(np.mean(np.abs(motor_cmd_rad - motor_actual_rad)))
    in_rmse = float(np.sqrt(np.mean((motor_cmd_rad - motor_actual_rad) ** 2)))
    ax.set_title(
        f"{base} | input/motor position | MAE={in_mae:.4f} rad, RMSE={in_rmse:.4f} rad"
    )
    ax.set_ylabel("motor-shaft rad")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if has_output_encoder:
        ax = axes[1]
        ax.plot(t_uniform, output_cmd, color="red", alpha=0.8, label="output cmd (synthetic)")
        ax.plot(
            t_uniform, output_actual_rad, color="green", linestyle=":", alpha=0.8,
            label="output actual",
        )
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("output-shaft rad")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel("Time (sec)")

    plt.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"    saved plot -> {png_path}")
    if show_plot:
        plt.show()
    plt.close(fig)

    # ----- save input .pt -----
    input_data = {
        "dof_pos": torch.tensor(motor_actual_rad, dtype=torch.float32).unsqueeze(1),
        "des_dof_pos": torch.tensor(motor_cmd_rad, dtype=torch.float32).unsqueeze(1),
        "time": torch.tensor(t_uniform, dtype=torch.float32),
        "joint_names": [f"{motor_type}Nm_J1"],
        "position_source": "input_encoder_pre_gearbox",
        "position_units": "motor_shaft_rad",
        "gear_ratio": GEAR_RATIO,
        "cnt_per_rev": CNT_PER_REV,
        "resample_hz": SIM_HZ,
        "sample_dt": 1.0 / SIM_HZ,
        "nominal_rotor_inertia": NOMINAL_ROTOR_INERTIAS[motor_type],
        "source_csv": os.path.basename(fp),
    }
    torch.save(input_data, input_pt)
    print(f"    saved -> {input_pt} ({len(t_uniform)} samples, input encoder)")

    # ----- save output reference .pt (옵션) -----
    if save_output_reference and has_output_encoder:
        output_reference_data = {
            "dof_pos": torch.tensor(output_actual_rad, dtype=torch.float32).unsqueeze(1),
            # True output-side desired is not logged; motor command / gear_ratio 일 뿐.
            "des_dof_pos": torch.tensor(output_cmd, dtype=torch.float32).unsqueeze(1),
            "time": torch.tensor(t_uniform, dtype=torch.float32),
            "joint_names": [f"{motor_type}Nm_J1"],
            "position_source": "output_encoder_reference",
            "position_units": "output_shaft_rad",
            "gear_ratio": GEAR_RATIO,
            "cnt_per_rev": CNT_PER_REV,
            "resample_hz": SIM_HZ,
            "sample_dt": 1.0 / SIM_HZ,
            "source_csv": os.path.basename(fp),
            "note": (
                "True output-side desired position is not logged; des_dof_pos is "
                "motor command divided by gear_ratio and should not be used for "
                "rotor-inertia validation."
            ),
        }
        torch.save(output_reference_data, output_pt)
        print(f"    saved -> {output_pt} (output reference)")
    elif save_output_reference and not has_output_encoder:
        print("    --save_output_reference 지정됐지만 OutputEncoder 없음 → 생략.")


# %%
# 입력이 파일이면 그 1개, 폴더면 안의 *.csv 를 "각각 별도로" 변환.
if input_path.is_file():
    files = [str(input_path)]
    save_dir = input_path.parent
else:
    files = sorted(glob.glob(os.path.join(str(input_path), "*.csv")))
    save_dir = input_path

if not files:
    raise RuntimeError(f"변환할 csv 가 없습니다: {input_path}")

# batch(여러 파일)에서는 plot 창을 띄우지 않고 png 만 저장한다.
batch = len(files) > 1
show_plot = (not no_show) and (not batch)

print(f"[INFO] {len(files)} file(s) to convert (motor_type={motor_type}Nm):")
for f in files:
    print(f"  - {os.path.basename(f)}")

n_ok, n_fail = 0, 0
for fp in files:
    try:
        convert_one(fp, save_dir, show_plot)
        n_ok += 1
    except Exception as e:  # noqa: BLE001 — batch 중 한 파일 실패가 전체를 막지 않게
        n_fail += 1
        print(f"[ERROR] {os.path.basename(fp)} 변환 실패: {e}")

print(f"[DONE] converted {n_ok}/{len(files)} file(s), failed {n_fail}.")
