# Based on fit_actuator.py — but no CMA-ES iteration.
# 1) actuator cfg 에 적힌 고정 파라미터로 sim 을 만들고
# 2) 실제 측정 데이터의 desired position 을 그대로 action 으로 주입
# 3) sim 의 joint_pos 를 기록
# 4) desired / measured(real) / sim_actual 셋을 한 figure 에 plot.

"""Launch Isaac Sim Simulator first."""


import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Replay desired trajectory with fixed actuator cfg and compare positions."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of envs (1 이면 충분 — 동일 cfg 로 단일 시뮬).",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Pace-Actuator-300Nm",
    help="Name of the task.",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default=None,
    help="결과 plot/데이터 저장 경로. 기본: <project>/logs/pace_replay/<robot>/<timestamp>",
)
parser.add_argument(
    "--no_show",
    action="store_true",
    help="설정 시 plot 표시 생략 (저장만).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pace_sim2real.tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg
from pace_sim2real.utils import project_root


def _unwrap_joint_positions(current_pos, previous_pos, unwrapped_pos):
    """Wrapped continuous joint pos → multi-turn unwrapped pos (fit_actuator 와 동일)."""
    delta = current_pos - previous_pos
    delta = torch.remainder(delta + math.pi, 2.0 * math.pi) - math.pi
    return unwrapped_pos + delta


def main():
    """단일 trajectory replay — 고정 cfg, CMA-ES 없음."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] {env.unwrapped.scene['robot'].cfg.spawn.joint_drive}")
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    articulation = env.unwrapped.scene["robot"]

    joint_order = env_cfg.sim2real.joint_order

    # --- 데이터 로드 ---
    data_file = project_root() / "data" / env_cfg.sim2real.data_dir
    data = torch.load(data_file)
    time_data = data["time"].to(env.unwrapped.device)
    target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)  # (T, J)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)  # (T, J)

    # --- joint mapping (fit_actuator 와 동일 로직) ---
    raw_data_joint_names = data.get("joint_names", joint_order)
    if all(name in articulation.joint_names for name in raw_data_joint_names):
        data_joint_names = raw_data_joint_names
    elif len(raw_data_joint_names) == len(joint_order):
        print(
            "[INFO]: Data joint labels do not match simulator joints; "
            f"using configured joint_order instead: {raw_data_joint_names} -> {joint_order}"
        )
        data_joint_names = joint_order
    else:
        missing = [
            name
            for name in raw_data_joint_names
            if name not in articulation.joint_names
        ]
        raise ValueError(
            "Data joint names not in articulation. "
            f"Missing={missing}, sim_joints={articulation.joint_names}, "
            f"joint_order={joint_order}"
        )

    data_joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in data_joint_names],
        device=env.unwrapped.device,
    )

    # --- env 초기화 + joint pos 강제 세팅 (real 첫 step 과 일치) ---
    env.reset()
    initial_dof_pos = (
        measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    )
    articulation.write_joint_position_to_sim(initial_dof_pos, joint_ids=data_joint_ids)
    articulation.write_joint_velocity_to_sim(
        torch.zeros_like(initial_dof_pos), joint_ids=data_joint_ids
    )
    articulation.write_joint_friction_coefficient_to_sim(
        joint_viscous_friction_coeff=torch.tensor([1.25e-5]),
        joint_friction_coeff=torch.tensor([0.5383]),
    )
    # --- replay 루프 준비 ---
    time_steps = time_data.shape[0]
    sim_dt = env.unwrapped.sim.cfg.dt
    num_data_joints = len(data_joint_names)

    sim_actual_buf = torch.zeros(
        (time_steps, num_data_joints), device=env.unwrapped.device
    )
    previous_sim_dof_pos = articulation.data.joint_pos[:, data_joint_ids].clone()
    unwrapped_sim_dof_pos = initial_dof_pos.clone()

    # 현재 actuator 파라미터 print (sanity check)
    motor = list(articulation.actuators.values())[0]  # 첫 actuator
    print("[INFO] Fixed actuator parameters:")
    print(f"       stiffness = {motor.stiffness}")
    print(f"       damping   = {motor.damping}")
    print(
        f"       armature  = {articulation.data.default_joint_armature[0, data_joint_ids]}"
    )
    print(
        f"       static friction  = {articulation.data.default_joint_friction_coeff[0, data_joint_ids]}"
    )
    print(
        f"       viscous friction = {articulation.data.default_joint_viscous_friction_coeff[0, data_joint_ids]}"
    )
    print(
        f"       dynamic friction = {articulation.data.default_joint_dynamic_friction_coeff[0, data_joint_ids]}"
    )

    print("[INFO] Live PhysX values:")
    print(
        f"       static  (live)  = {articulation.data.joint_friction_coeff[0, data_joint_ids]}"
    )
    print(
        f"       dynamic (live)  = {articulation.data.joint_dynamic_friction_coeff[0, data_joint_ids]}"
    )
    print(
        f"       viscous (live)  = {articulation.data.joint_viscous_friction_coeff[0, data_joint_ids]}"
    )
    print(
        f"[INFO] Replay start: T={time_steps} steps, dt={sim_dt}, "
        f"duration={time_data[-1].item():.2f} s"
    )

    counter = 0
    while simulation_app.is_running() and counter < time_steps:
        with torch.inference_mode():
            current_sim_dof_pos = articulation.data.joint_pos[:, data_joint_ids]
            if counter > 0:
                unwrapped_sim_dof_pos = _unwrap_joint_positions(
                    current_sim_dof_pos, previous_sim_dof_pos, unwrapped_sim_dof_pos
                )
            else:
                unwrapped_sim_dof_pos = current_sim_dof_pos.clone()
            previous_sim_dof_pos = current_sim_dof_pos.clone()

            # env 0 의 값을 기록 (num_envs 가 여러 개여도 모두 같은 cfg)
            sim_actual_buf[counter] = unwrapped_sim_dof_pos[0]

            actions = articulation.data.default_joint_pos.clone()
            actions[:, data_joint_ids] = (
                target_dof_pos[counter, :]
                .unsqueeze(0)
                .repeat(env.unwrapped.num_envs, 1)
            )
            env.step(actions)
            counter += 1
            if counter % 400 == 0:
                print(
                    f"[INFO]: Step {counter * sim_dt:.1f} / {time_data[-1]:.1f} s "
                    f"({counter / time_steps * 100:.1f} %)"
                )

    print("[INFO]: Replay finished.")

    # --- 저장 경로 ---
    if args_cli.save_dir is not None:
        save_dir = Path(args_cli.save_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%y_%m_%d_%H-%M-%S")
        save_dir = (
            project_root() / "logs" / "pace_replay" / env_cfg.sim2real.robot_name / ts
        )
    save_dir.mkdir(parents=True, exist_ok=True)

    # raw tensors 저장 (재plot 용)
    save_payload = {
        "time": time_data.cpu(),
        "des_dof_pos": target_dof_pos.cpu(),
        "measured_dof_pos": measured_dof_pos.cpu(),
        "sim_actual_dof_pos": sim_actual_buf.cpu(),
        "joint_order": data_joint_names,
        "actuator_cfg_snapshot": {
            "stiffness": motor.stiffness.cpu()
            if torch.is_tensor(motor.stiffness)
            else motor.stiffness,
            "damping": motor.damping.cpu()
            if torch.is_tensor(motor.damping)
            else motor.damping,
            "armature": articulation.data.default_joint_armature[
                0, data_joint_ids
            ].cpu(),
            "friction_static": articulation.data.default_joint_friction_coeff[
                0, data_joint_ids
            ].cpu(),
            "friction_viscous": articulation.data.default_joint_viscous_friction_coeff[
                0, data_joint_ids
            ].cpu(),
            "friction_dynamic": articulation.data.default_joint_dynamic_friction_coeff[
                0, data_joint_ids
            ].cpu(),
        },
    }
    torch.save(save_payload, save_dir / "replay_data.pt")
    print(f"[INFO] saved replay tensors -> {save_dir / 'replay_data.pt'}")

    # --- plot ---
    t = time_data.cpu().numpy()
    des = target_dof_pos.cpu().numpy()  # (T, J)
    meas = measured_dof_pos.cpu().numpy()  # (T, J)
    sim = sim_actual_buf.cpu().numpy()  # (T, J)

    n_joints = des.shape[1]
    fig, axes = plt.subplots(
        2, n_joints, figsize=(7 * n_joints, 8), squeeze=False, sharex=True
    )

    for j in range(n_joints):
        ax_pos = axes[0, j]
        ax_err = axes[1, j]

        ax_pos.plot(t, des[:, j], color="C0", lw=1.2, alpha=0.7, label="desired (cmd)")
        ax_pos.plot(t, meas[:, j], color="C2", lw=1.5, label="measured (real)")
        ax_pos.plot(t, sim[:, j], color="C3", lw=1.2, ls="--", label="sim actual")
        ax_pos.set_title(f"{data_joint_names[j]} — position")
        ax_pos.set_ylabel("position [rad]")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(loc="best", fontsize=9)

        err_sim_real = sim[:, j] - meas[:, j]
        err_sim_des = sim[:, j] - des[:, j]
        err_meas_des = meas[:, j] - des[:, j]
        ax_err.plot(t, err_sim_real, color="C3", lw=1.0, label="sim − real")
        ax_err.plot(
            t, err_sim_des, color="C1", lw=0.8, alpha=0.7, label="sim − desired"
        )
        ax_err.plot(
            t, err_meas_des, color="C4", lw=0.8, alpha=0.7, label="real − desired"
        )
        ax_err.axhline(0, color="k", lw=0.5)
        ax_err.set_ylabel("error [rad]")
        ax_err.set_xlabel("time [s]")
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(loc="best", fontsize=9)

        rmse_sim_real = float(np.sqrt(np.mean(err_sim_real**2)))
        mae_sim_real = float(np.mean(np.abs(err_sim_real)))
        ax_err.set_title(
            f"sim vs real:  RMSE = {rmse_sim_real:.4f} rad,  "
            f"MAE = {mae_sim_real:.4f} rad"
        )

    fig.suptitle(
        f"Replay — task: {args_cli.task}\n"
        f"actuator cfg: K={motor.stiffness.flatten()[0].item():.3g}, "
        f"D={motor.damping.flatten()[0].item():.3g}",
        fontsize=12,
    )
    plt.tight_layout()

    plot_path = save_dir / "replay_compare.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] saved plot -> {plot_path}")

    if not args_cli.no_show:
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
