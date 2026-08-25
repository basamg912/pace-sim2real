"""
여러 *_input_position_data.pt 를 하나로 stitch 해서 통합 데이터셋 생성.

각 segment 의 시작 위치 / 시간을 이전 segment 의 끝값에 이어붙여
연속 trajectory 로 만든다 (fit_actuator 의 sim 이 끊김 없이 재생 가능).

Usage:
    # 기본 — data/611/logs/ 의 300Nm_(80|120)_*_input_position_data.pt 7개를 stitch
    python scripts/merge_datasets.py

    # 직접 파일 지정 (순서대로 이어붙임)
    python scripts/merge_datasets.py \
        --files data/611/logs/300Nm_80_0.00005_input_position_data.pt \
                data/611/logs/300Nm_80_0.00007_input_position_data.pt \
                data/611/logs/300Nm_80_0.0001_input_position_data.pt

    # 패턴 + 정렬
    python scripts/merge_datasets.py \
        --root data/611/logs --pattern "300Nm_120_*_input_position_data.pt"

생성:
    --output (기본 data/611/logs/300Nm_merged_input_position_data.pt)
    + 같은 폴더의 merged_preview.png
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_FILES = [
    "300Nm_120_0.00005_input_position_data.pt",
    "300Nm_120_0.00007_input_position_data.pt",
    "300Nm_120_0.0002_input_position_data.pt",
    "300Nm_80_0.00005_input_position_data.pt",
    "300Nm_80_0.00007_input_position_data.pt",
    "300Nm_80_0.0001_input_position_data.pt",
    "300Nm_80_0.0002_input_position_data.pt",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        default="data/611/logs",
        help="입력 .pt 들의 폴더. --files 가 절대경로면 무시됨.",
    )
    p.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="이어붙일 파일 목록 (이 순서대로). 미지정시 default 7개 사용.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="--files 미지정 시 root 안에서 glob. 미지정이면 default 7 개.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 .pt 경로. 기본 <root>/300Nm_merged_input_position_data.pt",
    )
    p.add_argument(
        "--continuous",
        action="store_true",
        default=True,
        help="segment 의 시작 pos/cmd 를 이전 끝값에 align (연속 trajectory). 기본 True.",
    )
    p.add_argument(
        "--no_continuous",
        dest="continuous",
        action="store_false",
        help="segment 각각의 절대값을 그대로 사용 (segment 경계에서 jump 발생).",
    )
    p.add_argument(
        "--no_show", action="store_true", help="preview plot 표시 생략."
    )
    return p.parse_args()


def resolve_files(args):
    root = Path(args.root).expanduser().resolve()
    if args.files:
        files = []
        for f in args.files:
            fp = Path(f).expanduser()
            if not fp.is_absolute():
                fp = root / fp
            if not fp.exists():
                raise FileNotFoundError(f"file not found: {fp}")
            files.append(fp.resolve())
        return files

    if args.pattern:
        files = sorted(root.glob(args.pattern))
        if not files:
            raise RuntimeError(f"no files matching {args.pattern!r} in {root}")
        return [f.resolve() for f in files]

    files = []
    for name in DEFAULT_FILES:
        fp = root / name
        if not fp.exists():
            raise FileNotFoundError(f"default file not found: {fp}")
        files.append(fp.resolve())
    return files


def load_one(path: Path):
    d = torch.load(path, map_location="cpu")
    t = d["time"]
    des = d["des_dof_pos"]
    pos = d["dof_pos"]
    joints = d.get("joint_names", ["J1"])
    if t.ndim != 1:
        t = t.squeeze()
    if des.ndim == 1:
        des = des.unsqueeze(1)
    if pos.ndim == 1:
        pos = pos.unsqueeze(1)
    return t.float(), des.float(), pos.float(), joints, d


def main():
    args = parse_args()
    files = resolve_files(args)

    print(f"[INFO] merging {len(files)} files (continuous={args.continuous})")
    for i, f in enumerate(files):
        print(f"   {i+1:2d}. {f.name}")

    merged_t = []
    merged_des = []
    merged_pos = []
    boundaries = [0]  # 각 segment 시작 인덱스 (0, len(seg0), len(seg0)+len(seg1), ...)
    source_files = []
    joint_names_set = None

    t_offset = 0.0
    pos_offset = None   # (J,) 텐서 — segment 마다 누적
    cmd_offset = None

    for f in files:
        t, des, pos, joints, _meta = load_one(f)

        # joint_names 일관성 체크
        if joint_names_set is None:
            joint_names_set = joints
        elif list(joints) != list(joint_names_set):
            print(
                f"[WARN] joint_names 불일치 — first={joint_names_set}, this={joints}. "
                "첫 번째 기준으로 진행."
            )

        # shape: (T,), (T, J), (T, J)
        T_seg = t.shape[0]

        # 시간 offset — 직전 segment 의 마지막 시각 + 평균 dt
        if len(merged_t) > 0:
            prev_t = merged_t[-1]
            dt_avg = float((prev_t[-1] - prev_t[0]) / max(1, len(prev_t) - 1))
            t_offset = float(prev_t[-1].item() + dt_avg)
        t_seg = t + t_offset

        # position / command continuity
        if args.continuous:
            # 첫 segment 는 그대로, 이후 segment 는 시작값을 이전 끝값에 맞춤
            if pos_offset is None:
                pos_off = torch.zeros(pos.shape[1])
                cmd_off = torch.zeros(des.shape[1])
            else:
                # 새 segment 가 자체적으로 0 에서 시작한다고 가정 (export_dataSingle 의 동작)
                # → 이전 끝값으로 평행이동
                pos_off = pos_offset - pos[0]
                cmd_off = cmd_offset - des[0]
            pos_seg = pos + pos_off
            des_seg = des + cmd_off
            # 다음 iteration 용 누적값 update
            pos_offset = pos_seg[-1].clone()
            cmd_offset = des_seg[-1].clone()
        else:
            pos_seg = pos
            des_seg = des

        merged_t.append(t_seg)
        merged_des.append(des_seg)
        merged_pos.append(pos_seg)
        source_files.append(str(f))
        boundaries.append(boundaries[-1] + T_seg)

        print(
            f"   loaded {f.name:50s}  T={T_seg:>5d}  "
            f"t_offset={t_offset:.3f}s  pos_range=[{pos_seg.min():.3f}, {pos_seg.max():.3f}]"
        )

    full_t = torch.cat(merged_t, dim=0)
    full_des = torch.cat(merged_des, dim=0)
    full_pos = torch.cat(merged_pos, dim=0)

    print(
        f"\n[INFO] merged: T={full_t.shape[0]}, J={full_pos.shape[1]}, "
        f"duration={full_t[-1].item() - full_t[0].item():.2f}s"
    )
    print(f"       des range: [{full_des.min():.3f}, {full_des.max():.3f}]")
    print(f"       pos range: [{full_pos.min():.3f}, {full_pos.max():.3f}]")

    # --- 저장 ---
    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(args.root).expanduser().resolve() / "300Nm_merged_input_position_data.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "time": full_t,
        "des_dof_pos": full_des,
        "dof_pos": full_pos,
        "joint_names": joint_names_set,
        "position_source": "merged_input_encoder_pre_gearbox",
        "position_units": "motor_shaft_rad",
        "n_segments": len(files),
        "segment_boundaries": boundaries,    # len = n_segments + 1
        "source_files": source_files,
        "continuous_stitch": bool(args.continuous),
    }
    torch.save(payload, out_path)
    print(f"\n[INFO] saved merged -> {out_path}")

    # --- preview plot ---
    t_np = full_t.numpy()
    des_np = full_des.numpy()
    pos_np = full_pos.numpy()
    n_j = pos_np.shape[1]

    fig, ax = plt.subplots(2, n_j, figsize=(14, 6), squeeze=False)
    for j in range(n_j):
        ax_top = ax[0, j]
        ax_bot = ax[1, j]
        ax_top.plot(t_np, des_np[:, j], color="C0", lw=0.8, alpha=0.7, label="desired")
        ax_top.plot(t_np, pos_np[:, j], color="C2", lw=1.0, label="measured (real)")
        # segment 경계 vertical line
        for b in boundaries[1:-1]:
            ax_top.axvline(t_np[b], color="r", ls=":", alpha=0.5, lw=0.8)
        ax_top.set_title(f"merged trajectory — joint {joint_names_set[j]}  (n={len(files)} segs)")
        ax_top.set_ylabel("position [rad]")
        ax_top.legend(loc="best", fontsize=8)
        ax_top.grid(True, alpha=0.3)

        err = pos_np[:, j] - des_np[:, j]
        ax_bot.plot(t_np, err, color="C3", lw=0.6, label="real − desired")
        for b in boundaries[1:-1]:
            ax_bot.axvline(t_np[b], color="r", ls=":", alpha=0.5, lw=0.8)
        ax_bot.axhline(0, color="k", lw=0.4)
        ax_bot.set_xlabel("time [s]"); ax_bot.set_ylabel("err [rad]")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Merged dataset (continuous={args.continuous})  —  "
        f"7 segments, total {full_t.shape[0]} samples, "
        f"{full_t[-1].item() - full_t[0].item():.1f}s",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = out_path.with_suffix(".png")
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] saved preview -> {png_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
