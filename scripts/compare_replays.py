"""
두 replay_data.pt (서로 다른 actuator 파라미터셋의 trajectory replay 결과) 를
한 figure 에 겹쳐 비교.

각 파일은 fit_actuator2.py 가 저장한 형식 가정:
    {
        "time": (T,),
        "des_dof_pos":         (T, J),
        "measured_dof_pos":    (T, J),
        "sim_actual_dof_pos":  (T, J),
        "joint_order":         list[str],
        "actuator_cfg_snapshot": dict,   # 옵션
    }

Usage:
    fit_actuator2.py 로 replay_data.pt 생성 후
    python scripts/compare_replays.py --pt2 logs/pace_replay/Robot/26_05_26_14-25-23/replay_data.pt

    python scripts/compare_replays.py \
        --pt1 logs/pace_replay/Robot/300Nm-handtuned-427-parameter/replay_data.pt \
        --pt2 logs/pace_replay/Robot/300Nm-handtuned-pace-param/replay_data.pt

생성:
    <첫번째 pt 의 dir>/compare_replays.png   (--save 로 변경 가능)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pt1",
        type=str,
        default="logs/pace_replay/Robot/300Nm-handtuned-427-parameter/replay_data.pt",
        help="첫 번째 replay_data.pt 경로 (e.g. handtuned 427).",
    )
    p.add_argument(
        "--pt2",
        type=str,
        default="logs/pace_replay/Robot/300Nm-handtuned-pace-param/replay_data.pt",
        help="두 번째 replay_data.pt 경로 (e.g. pace fit 결과).",
    )
    p.add_argument(
        "--label1", type=str, default=None, help="legend 라벨 1. 기본: pt1 폴더명."
    )
    p.add_argument(
        "--label2", type=str, default=None, help="legend 라벨 2. 기본: pt2 폴더명."
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="저장 PNG 경로. 기본: <pt1>/compare_replays.png",
    )
    p.add_argument("--no_show", action="store_true", help="plt.show() 생략.")
    return p.parse_args()


def load_replay(pt_path: Path):
    if not pt_path.exists():
        raise FileNotFoundError(f"replay_data.pt not found: {pt_path}")
    d = torch.load(pt_path, map_location="cpu")
    t = d["time"].numpy() if torch.is_tensor(d["time"]) else np.asarray(d["time"])
    des = d["des_dof_pos"].numpy()
    meas = d["measured_dof_pos"].numpy()
    sim = d["sim_actual_dof_pos"].numpy()
    joint_order = d.get("joint_order", [f"joint{i}" for i in range(des.shape[1])])
    snap = d.get("actuator_cfg_snapshot", {})
    return t, des, meas, sim, joint_order, snap


def _to_scalar(v):
    """tensor / np / list 어떤 형태든 첫 element 의 float 반환."""
    if torch.is_tensor(v):
        return float(v.flatten()[0].item())
    arr = np.asarray(v).flatten()
    return float(arr[0]) if arr.size else float("nan")


def cfg_summary(snap: dict) -> str:
    """actuator_cfg_snapshot 의 주요 값을 한 줄로 요약."""
    if not snap:
        return ""
    parts = []
    for key in (
        "stiffness",
        "damping",
        "armature",
        "friction_static",
        "friction_viscous",
    ):
        if key in snap:
            try:
                parts.append(f"{key}={_to_scalar(snap[key]):.4g}")
            except Exception:
                pass
    return "  |  ".join(parts)


def main():
    args = parse_args()

    pt1 = Path(args.pt1).expanduser().resolve()
    pt2 = Path(args.pt2).expanduser().resolve()
    label1 = args.label1 or pt1.parent.name
    label2 = args.label2 or pt2.parent.name

    t1, des1, meas1, sim1, joints1, snap1 = load_replay(pt1)
    t2, des2, meas2, sim2, joints2, snap2 = load_replay(pt2)

    if des1.shape != des2.shape:
        print(
            f"[WARN] shape mismatch — pt1={des1.shape}, pt2={des2.shape}. "
            "공통 길이로 자릅니다."
        )
        T = min(des1.shape[0], des2.shape[0])
        J = min(des1.shape[1], des2.shape[1])
        t1 = t1[:T]
        t2 = t2[:T]
        des1, meas1, sim1 = des1[:T, :J], meas1[:T, :J], sim1[:T, :J]
        des2, meas2, sim2 = des2[:T, :J], meas2[:T, :J], sim2[:T, :J]
        joints1 = joints1[:J]

    # 시간축은 pt1 기준 (두 파일 같은 데이터셋 가정)
    t = t1
    n_joints = des1.shape[1]

    print(f"[INFO] pt1 ({label1}):  {pt1}")
    print(f"       cfg: {cfg_summary(snap1)}")
    print(f"[INFO] pt2 ({label2}):  {pt2}")
    print(f"       cfg: {cfg_summary(snap2)}")
    print(f"[INFO] joints={joints1}, T={len(t)}, dt~={t[1] - t[0]:.4f}s")

    # 색 컨벤션
    C_DES = "C0"  # desired
    C_MEAS = "C2"  # measured (real)
    C_SIM1 = "C3"  # sim from pt1
    C_SIM2 = "C1"  # sim from pt2

    fig, axes = plt.subplots(
        2, n_joints, figsize=(8 * n_joints, 9), squeeze=False, sharex=True
    )

    for j in range(n_joints):
        ax_pos = axes[0, j]
        ax_err = axes[1, j]

        # 상단: 모든 라인 겹치기
        ax_pos.plot(
            t, des1[:, j], color=C_DES, lw=1.2, alpha=0.7, label="desired (cmd)"
        )
        ax_pos.plot(t, meas1[:, j], color=C_MEAS, lw=1.8, label="measured (real)")
        ax_pos.plot(
            t, sim1[:, j], color=C_SIM1, lw=1.3, ls="--", label=f"sim — {label1}"
        )
        ax_pos.plot(
            t, sim2[:, j], color=C_SIM2, lw=1.3, ls=":", label=f"sim — {label2}"
        )

        ax_pos.set_title(f"{joints1[j]} — position comparison")
        ax_pos.set_ylabel("position [rad]")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(loc="best", fontsize=9)

        # 하단: 두 sim vs measured 에러
        err1 = sim1[:, j] - meas1[:, j]
        err2 = sim2[:, j] - meas2[:, j]
        ax_err.plot(
            t, err1, color=C_SIM1, lw=1.2, ls="--", label=f"sim − real  ({label1})"
        )
        ax_err.plot(
            t, err2, color=C_SIM2, lw=1.2, ls=":", label=f"sim − real  ({label2})"
        )
        ax_err.axhline(0, color="k", lw=0.5)
        ax_err.set_ylabel("error [rad]")
        ax_err.set_xlabel("time [s]")
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(loc="best", fontsize=9)

        rmse1 = float(np.sqrt(np.mean(err1**2)))
        mae1 = float(np.mean(np.abs(err1)))
        rmse2 = float(np.sqrt(np.mean(err2**2)))
        mae2 = float(np.mean(np.abs(err2)))
        ax_err.set_title(
            f"{label1}: RMSE={rmse1:.4f}, MAE={mae1:.4f}    |    "
            f"{label2}: RMSE={rmse2:.4f}, MAE={mae2:.4f}   [rad]"
        )

    # 전체 title — cfg snapshot 도 표기
    s1 = cfg_summary(snap1)
    s2 = cfg_summary(snap2)
    title = "Replay comparison"
    if s1 or s2:
        title += f"\n[{label1}] {s1}\n[{label2}] {s2}"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = (
        Path(args.save).expanduser().resolve()
        if args.save
        else pt1.parent / "compare_replays.png"
    )
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] saved -> {save_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
