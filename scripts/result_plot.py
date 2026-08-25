"""
CMA-ES fitting 결과 종합 plot.

한 figure 에 아래를 모두 그린다:
    1. 파라미터 추이 (armature, damping, friction, [dynamic_friction]) — bound 표시 포함
    2. score (loss) 추이 — progress.pt 또는 TensorBoard event 에서 읽음
    3. position tracking 비교 — desired / real / best_sim_trajectory
    4. position error (best_sim − real)

Usage:
    python scripts/result_plot.py --log_dir logs/pace/Robot/26_05_22_HH-MM-SS

저장:
    {log_dir}/fitting_curves.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# ---------- I/O ----------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--log_dir",
        required=True,
        help="CMA-ES 한 run 의 로그 디렉토리 (mean_*.pt, config.pt 가 있는 곳).",
    )
    p.add_argument(
        "--save",
        default=None,
        help="저장 PNG 경로. 기본: <log_dir>/fitting_curves.png",
    )
    p.add_argument(
        "--no_show", action="store_true", help="설정 시 plt.show() 생략, 저장만."
    )
    return p.parse_args()


def load_mean_checkpoints(log_dir: Path):
    """mean_NNN.pt 모음 → (iters: (N,), params: (N, n_params))."""
    files = sorted(log_dir.glob("mean_*.pt"))
    if not files:
        raise FileNotFoundError(f"No mean_*.pt found in {log_dir}")

    pat = re.compile(r"mean_(\d+)\.pt")
    pairs = []
    for f in files:
        m = pat.match(f.name)
        if m is None:
            continue
        pairs.append((int(m.group(1)), torch.load(f, map_location="cpu").flatten()))
    pairs.sort(key=lambda x: x[0])

    iters = torch.tensor([p[0] for p in pairs])
    params = torch.stack([p[1] for p in pairs])
    return iters, params


def load_progress_scores(log_dir: Path):
    """progress.pt → scores_buffer (T, pop), 없으면 None."""
    pp = log_dir / "progress.pt"
    if not pp.exists():
        return None
    d = torch.load(pp, map_location="cpu")
    s = d.get("scores_buffer", None)
    if s is None:
        return None
    # 끝까지 안 돈 경우 빈 row 제거
    valid = s.abs().sum(dim=1) > 0
    return s[valid]


def load_tensorboard_scores(log_dir: Path):
    """event 파일 → {tag: (iters, values)} dict. 우선순위: tensorboard EventAccumulator > tbparse.
    필요한 scalar tag 들 (best score, max score 등) 을 한 번에 가져온다.
    """
    events = list(log_dir.glob("events.out.tfevents.*"))
    if not events:
        return None

    target_tags = ("0_Episode/score", "0_Episode/max_score", "0_Episode/diff_score")

    # 1순위: tensorboard 가 이미 의존성에 있음 (SummaryWriter 가 이걸 사용함)
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))
        result = {}
        for tag in target_tags:
            if tag in available:
                ev = ea.Scalars(tag)
                result[tag] = (
                    torch.tensor([e.step for e in ev]),
                    torch.tensor([e.value for e in ev]),
                )
        if result:
            return result
    except ImportError:
        pass

    # 2순위: tbparse
    try:
        from tbparse import SummaryReader

        reader = SummaryReader(str(log_dir), pivot=False)
        df = reader.scalars
        if df is None or df.empty:
            return None
        result = {}
        for tag in target_tags:
            sub = df[df["tag"] == tag]
            if not sub.empty:
                result[tag] = (
                    torch.tensor(sub["step"].to_numpy()),
                    torch.tensor(sub["value"].to_numpy()),
                )
        return result if result else None
    except ImportError:
        pass

    return None


def load_best_trajectory(log_dir: Path):
    """best_trajectory.pt → (T, n_joints) 또는 None."""
    p = log_dir / "best_trajectory.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu")


# ---------- Plot panels ----------


def plot_params(
    ax_list, iters, params, n_joints, n_groups, group_names, bounds, joint_order
):
    """그룹별 axes 에 파라미터 추이 + bound 라인 그리기."""
    for g, name in enumerate(group_names):
        ax = ax_list[g]
        s = slice(g * n_joints, (g + 1) * n_joints)
        vals = params[:, s]  # (n_iters, n_joints)

        for j in range(n_joints):
            ax.plot(
                iters,
                vals[:, j],
                lw=1.5,
                marker=".",
                markersize=4,
                label=joint_order[j],
            )

        lo = bounds[s, 0].mean().item()
        hi = bounds[s, 1].mean().item()
        ax.axhline(lo, color="gray", ls="--", alpha=0.5, label=f"low = {lo:.4g}")
        ax.axhline(hi, color="gray", ls="--", alpha=0.5, label=f"high = {hi:.4g}")

        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

        if lo > 0 and hi / max(lo, 1e-12) > 1e3:
            ax.set_yscale("log")


def plot_score(ax, log_dir: Path):
    """score 추이.
    우선순위: progress.pt (full population) → tensorboard event (best/max scalar).
    """
    # 1순위: progress.pt (population 분포까지 가능)
    scores_buf = load_progress_scores(log_dir)
    if scores_buf is not None:
        it = torch.arange(scores_buf.shape[0])
        mn = scores_buf.min(dim=1).values
        mx = scores_buf.max(dim=1).values
        md = scores_buf.median(dim=1).values

        ax.fill_between(it, mn, mx, color="C0", alpha=0.2, label="population min~max")
        ax.plot(it, md, color="C0", lw=1.0, label="median")
        ax.plot(it, mn, color="C1", lw=1.5, label="best (min)")
        ax.set_title("score (MSE [rad²]) — from progress.pt")
        source_found = True
    else:
        # 2순위: TensorBoard event (best/max scalar)
        tb = load_tensorboard_scores(log_dir)
        if tb is not None:
            if "0_Episode/score" in tb:
                it, sc = tb["0_Episode/score"]
                ax.plot(it, sc, color="C1", lw=1.5, label="best (min)")
            if "0_Episode/max_score" in tb:
                it, sc = tb["0_Episode/max_score"]
                ax.plot(
                    it,
                    sc,
                    color="C0",
                    lw=1.0,
                    alpha=0.7,
                    label="max (population worst)",
                )
            ax.set_title("score (MSE [rad²]) — from TensorBoard")
            source_found = True
        else:
            ax.text(
                0.5,
                0.5,
                "No score data\n(progress.pt 또는 events.out.tfevents.* 필요)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
            )
            ax.set_title("score — unavailable")
            source_found = False

    ax.set_xlabel("iteration")
    ax.set_ylabel("score")
    if source_found:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    if source_found:
        ax.legend(loc="best", fontsize=8)


def plot_trajectory(ax_pos, ax_err, config, best_traj):
    """desired/real/sim 비교 + error."""
    t = config["time"].squeeze()
    real = config["dof_pos"].squeeze()  # (T,) or (T, J)
    des = config["des_dof_pos"].squeeze()
    sim = best_traj.squeeze() if best_traj is not None else None

    if real.ndim == 1:
        real = real.unsqueeze(-1)
        des = des.unsqueeze(-1)
        if sim is not None:
            sim = sim.unsqueeze(-1)

    n_joints = real.shape[1]
    for j in range(n_joints):
        ax_pos.plot(
            t,
            des[:, j],
            color="C0",
            lw=1.0,
            alpha=0.6,
            label="desired" if j == 0 else None,
        )
        ax_pos.plot(t, real[:, j], color="C2", lw=1.5, label="real" if j == 0 else None)
        if sim is not None:
            ax_pos.plot(
                t,
                sim[:, j],
                color="C3",
                lw=1.2,
                ls="--",
                label="best sim" if j == 0 else None,
            )
            ax_err.plot(
                t,
                sim[:, j] - real[:, j],
                color="C3",
                lw=1.0,
                label="sim − real" if j == 0 else None,
            )

    ax_pos.set_ylabel("position [rad]")
    ax_pos.legend(loc="best", fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    ax_err.set_ylabel("error [rad]")
    ax_err.set_xlabel("time [s]")
    ax_err.axhline(0, color="k", lw=0.5)
    ax_err.legend(loc="best", fontsize=8)
    ax_err.grid(True, alpha=0.3)

    if sim is not None:
        rmse = ((sim - real) ** 2).mean().sqrt().item()
        mae = (sim - real).abs().mean().item()
        ax_err.set_title(f"RMSE = {rmse:.4f} rad,  MAE = {mae:.4f} rad")


# ---------- Main ----------


def main():
    args = parse_args()
    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.is_dir():
        raise FileNotFoundError(f"log_dir not found: {log_dir}")

    # 메타 로드
    cfg_path = log_dir / "config.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.pt not found in {log_dir}")
    config = torch.load(cfg_path, map_location="cpu")
    joint_order = config["joint_order"]
    bounds = config["bounds"]
    n_joints = len(joint_order)
    n_params = bounds.shape[0]
    n_groups = n_params // n_joints

    group_names_full = ["armature", "damping", "friction", "dynamic_friction"]
    group_names = group_names_full[:n_groups]

    # 파라미터 시계열
    iters, params = load_mean_checkpoints(log_dir)
    print(f"[INFO] Loaded {len(iters)} checkpoints from {log_dir.name}")
    print(f"       iter range: {iters[0].item()} ~ {iters[-1].item()}")
    print(f"       n_groups={n_groups}, n_joints={n_joints}")

    if params.shape[1] != n_params:
        common = min(params.shape[1], n_params)
        print(
            f"[WARN] mean_*.pt ({params.shape[1]}) vs bounds ({n_params}) 불일치 → {common} 까지만 사용"
        )
        params = params[:, :common]
        bounds = bounds[:common, :]
        n_params = common
        n_groups = n_params // n_joints
        group_names = group_names_full[:n_groups]

    # best trajectory
    best_traj = load_best_trajectory(log_dir)
    if best_traj is None:
        print("[WARN] best_trajectory.pt not found — position panel 생략")

    # ============== Figure layout (2 columns grid) ==============
    # 패널 = 파라미터 그룹 (n_groups) + score 1
    # ncols = 2 고정, nrows = ceil(n_panels / 2)
    n_panels = n_groups + 1
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows), squeeze=False)
    flat_axes = [axes[r, c] for r in range(nrows) for c in range(ncols)]

    # 파라미터 axes — flat 순서대로 채움
    param_axes = flat_axes[:n_groups]
    plot_params(
        param_axes, iters, params, n_joints, n_groups, group_names, bounds, joint_order
    )
    for ax in param_axes:
        ax.set_xlabel("CMA-ES iteration")

    # score axes
    score_ax = flat_axes[n_groups]
    plot_score(score_ax, log_dir)

    # 남는 axes 는 숨김 (예: 5 패널 → 3x2 의 6번째)
    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(f"Pace fitting — {log_dir.name}", fontsize=13, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    save_path = (
        Path(args.save).expanduser().resolve()
        if args.save
        else log_dir / "fitting_curves.png"
    )
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] saved -> {save_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
