"""
logs/pace/Robot/ 안의 모든 fit 실험을 자동 스캔하여:
    1. 각 실험의 최종 fitted parameter 들을 console 에 표로 print
    2. 각 실험의 desired / real / best-sim trajectory 를 한 figure 에 겹쳐 plot

각 실험 디렉토리 구조 (fit_actuator.py 가 생성):
    logs/pace/Robot/<exp_name>/<timestamp>/
        config.pt          # bounds, joint_order, dof_pos (real), des_dof_pos, time
        mean_NNN.pt        # best fitted params at iteration NNN (마지막이 최종)
        best_trajectory.pt # sim trajectory with best params
        events.out.tfevents.*

Usage:
    python scripts/summarize_all_fits.py
    python scripts/summarize_all_fits.py --logs_root logs/pace/Robot --pattern "300Nm_*"
    python scripts/summarize_all_fits.py --no_show --save logs/summary.png
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------- arg ----------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--logs_root",
        type=str,
        default="logs/pace/Robot",
        help="실험 디렉토리들이 모여있는 root. 기본 logs/pace/Robot.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="실험 이름 glob 패턴. 기본 '*' (전부).",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="저장 PNG 경로. 기본 <logs_root>/summary_all_fits.png",
    )
    p.add_argument("--no_show", action="store_true", help="plt.show() 생략.")
    p.add_argument(
        "--cols", type=int, default=2, help="figure 의 열 수 (실험들 배치). 기본 2."
    )
    return p.parse_args()


# ---------- helpers ----------


def latest_run_dir(exp_dir: Path) -> Path | None:
    """exp_dir 안에 있는 timestamp subdir 중 가장 최근 것을 반환."""
    subs = [d for d in exp_dir.iterdir() if d.is_dir()]
    if not subs:
        return None
    # 이름이 timestamp 형식이라 그대로 sort 해도 시간순
    return sorted(subs)[-1]


def latest_mean_file(run_dir: Path) -> Path | None:
    """run_dir 의 mean_NNN.pt 중 가장 큰 iteration 의 파일."""
    files = sorted(run_dir.glob("mean_*.pt"))
    if not files:
        return None
    pat = re.compile(r"mean_(\d+)\.pt")
    pairs = []
    for f in files:
        m = pat.match(f.name)
        if m is None:
            continue
        pairs.append((int(m.group(1)), f))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    return pairs[-1][1]


def load_fit_run(run_dir: Path):
    """한 run 디렉토리에서 (config, last_mean_iter, last_mean_params, best_traj) 로드.

    실패 항목은 None 반환.
    """
    config_path = run_dir / "config.pt"
    if not config_path.exists():
        return None, None, None, None

    config = torch.load(config_path, map_location="cpu")

    mean_file = latest_mean_file(run_dir)
    if mean_file is None:
        return config, None, None, None
    last_iter = int(re.match(r"mean_(\d+)\.pt", mean_file.name).group(1))
    last_mean = torch.load(mean_file, map_location="cpu").flatten()

    best_traj_path = run_dir / "best_trajectory.pt"
    best_traj = (
        torch.load(best_traj_path, map_location="cpu")
        if best_traj_path.exists()
        else None
    )

    return config, last_iter, last_mean, best_traj


def split_params(params: torch.Tensor, n_joints: int):
    """파라미터 vector 를 group 별 dict 로 분리.

    group 순서: armature, damping (viscous), friction (static), [dynamic_friction]
    """
    n_groups = params.numel() // n_joints
    names = ["armature", "damping", "friction"]
    if n_groups >= 4:
        names.append("dynamic_friction")
    names = names[:n_groups]

    out = {}
    for g, name in enumerate(names):
        out[name] = params[g * n_joints : (g + 1) * n_joints].numpy()
    return out


def trajectory_rmse(sim: np.ndarray, real: np.ndarray) -> float:
    return float(np.sqrt(np.mean((sim - real) ** 2)))


# ---------- main ----------


def main():
    args = parse_args()
    logs_root = Path(args.logs_root).expanduser().resolve()
    if not logs_root.is_dir():
        raise FileNotFoundError(f"logs_root not found: {logs_root}")

    exp_dirs = sorted(p for p in logs_root.glob(args.pattern) if p.is_dir())
    if not exp_dirs:
        raise RuntimeError(f"No exp dirs found under {logs_root} matching {args.pattern!r}")

    runs = []  # [(exp_name, run_dir, config, last_iter, params_dict, best_traj, rmse), ...]
    for exp_dir in exp_dirs:
        run_dir = latest_run_dir(exp_dir)
        if run_dir is None:
            print(f"[WARN] no run in {exp_dir.name} — skip")
            continue
        config, last_iter, last_mean, best_traj = load_fit_run(run_dir)
        if config is None:
            print(f"[WARN] no config.pt in {run_dir} — skip")
            continue
        if last_mean is None:
            print(f"[WARN] no mean_*.pt in {run_dir} — skip")
            continue

        joint_order = config["joint_order"]
        n_joints = len(joint_order)
        params = split_params(last_mean, n_joints)

        # trajectory RMSE
        real = config["dof_pos"]
        if torch.is_tensor(real):
            real = real.cpu().numpy()
        if best_traj is not None:
            sim_arr = best_traj.cpu().numpy() if torch.is_tensor(best_traj) else best_traj
            T_common = min(real.shape[0], sim_arr.shape[0])
            rmse = trajectory_rmse(sim_arr[:T_common], real[:T_common])
        else:
            sim_arr = None
            rmse = float("nan")

        runs.append(
            dict(
                exp_name=exp_dir.name,
                run_dir=run_dir,
                config=config,
                last_iter=last_iter,
                params=params,
                joint_order=joint_order,
                real=real,
                sim=sim_arr,
                rmse=rmse,
            )
        )

    if not runs:
        raise RuntimeError("No valid runs to summarize.")

    # =========== 1. 파라미터 표 print ===========
    # 사용된 param group 목록 (모든 run 의 union)
    all_group_names = []
    for r in runs:
        for g in r["params"].keys():
            if g not in all_group_names:
                all_group_names.append(g)

    print("\n" + "=" * 100)
    print(f"{'experiment':40s}  {'iter':>5s}  " + "  ".join(f"{g:>14s}" for g in all_group_names) + "  " + f"{'RMSE [rad]':>10s}")
    print("=" * 100)
    for r in runs:
        line = f"{r['exp_name']:40s}  {r['last_iter']:>5d}  "
        for g in all_group_names:
            if g in r["params"]:
                vals = r["params"][g]
                # joint 가 1개라 가정 (현재 setup) — 첫 값만
                line += f"{vals[0]:>14.6g}  "
            else:
                line += f"{'-':>14s}  "
        line += f"{r['rmse']:>10.4g}"
        print(line)
    print("=" * 100 + "\n")

    # =========== 2. trajectory overlay plot ===========
    n_exps = len(runs)
    ncols = max(1, args.cols)
    nrows = (n_exps + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(8 * ncols, 3.2 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    for i, r in enumerate(runs):
        ax = axes_flat[i]
        config = r["config"]
        real = r["real"]
        sim = r["sim"]
        des = config["des_dof_pos"]
        t = config["time"]
        if torch.is_tensor(des):
            des = des.cpu().numpy()
        if torch.is_tensor(t):
            t = t.cpu().numpy()

        # 1-D 보정
        if real.ndim == 2 and real.shape[1] == 1:
            real_1d, des_1d = real[:, 0], des[:, 0]
        else:
            real_1d, des_1d = real.squeeze(), des.squeeze()
        sim_1d = (sim.squeeze() if sim is not None else None)

        T_plot = min(len(t), len(real_1d), len(des_1d))
        if sim_1d is not None:
            T_plot = min(T_plot, len(sim_1d))
        t_plot = t[:T_plot] if len(t) >= T_plot else np.arange(T_plot)

        ax.plot(t_plot, des_1d[:T_plot], color="C0", lw=1.0, alpha=0.6, label="desired")
        ax.plot(t_plot, real_1d[:T_plot], color="C2", lw=1.5, label="real (fit input)")
        if sim_1d is not None:
            ax.plot(
                t_plot,
                sim_1d[:T_plot],
                color="C3",
                lw=1.2,
                ls="--",
                label="sim (best fit)",
            )

        # 제목에 파라미터 + RMSE 요약
        param_str = "  ".join(
            f"{g[:3]}={r['params'][g][0]:.3g}" for g in all_group_names if g in r["params"]
        )
        ax.set_title(
            f"{r['exp_name']}\niter={r['last_iter']}, RMSE={r['rmse']:.4g}  |  {param_str}",
            fontsize=9,
        )
        ax.set_xlabel("time [s]")
        ax.set_ylabel("pos [rad]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    # 남는 axes 숨김
    for ax in axes_flat[n_exps:]:
        ax.set_visible(False)

    fig.suptitle(
        f"All Pace fit experiments under {logs_root.name}/  (n={n_exps})",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = (
        Path(args.save).expanduser().resolve()
        if args.save
        else logs_root / "summary_all_fits.png"
    )
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] saved figure -> {save_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
