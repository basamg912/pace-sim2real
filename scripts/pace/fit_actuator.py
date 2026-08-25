# Based on fit.py — 3-parameter actuator fitting only (armature, damping, friction)
# No bias or delay optimization.

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import pathlib

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Actuator fitting (armature, damping, friction) for Isaac Lab environments."
)
parser.add_argument(
    "--num_envs", type=int, default=4096, help="Number of environments to simulate."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Pace-Actuator-100Nm",
    help="Name of the task.",
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help=(
        "Path to a YAML batch-config. 지정 시 그 안에 나열된 복수개의 pt 파일을 "
        "한 번의 sim 세션에서 순차적으로 fitting 한다 (반복 실험). "
        "미지정 시 기존처럼 task config 의 data_dir 1개만 fitting."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# --- batch config 를 AppLauncher 기동 전에 먼저 읽어 task/num_envs/headless 결정 ---
batch_cfg = None
if args_cli.config is not None:
    import sys

    import yaml

    with open(args_cli.config, "r") as _f:
        batch_cfg = yaml.safe_load(_f) or {}
    if batch_cfg.get("task"):
        args_cli.task = batch_cfg["task"]
    # CLI 로 명시하지 않았을 때만 yaml 값으로 num_envs 덮어쓰기
    if batch_cfg.get("num_envs") and "--num_envs" not in sys.argv:
        args_cli.num_envs = int(batch_cfg["num_envs"])
    if batch_cfg.get("headless"):
        args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import pace_sim2real.tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg
from pace_sim2real import ActuatorCMAESOptimizer
from pace_sim2real.utils import project_root


def _unwrap_joint_positions(current_pos, previous_pos, unwrapped_pos):
    """Convert wrapped continuous-joint positions into multi-turn positions."""
    delta = current_pos - previous_pos
    # delta 값을 -pi ~ pi 범위로 제한, remainder 연산은 2pi 를 기준으로 나머지 계산하고 Abs 로 반환
    delta = torch.remainder(delta + math.pi, 2.0 * math.pi) - math.pi
    return unwrapped_pos + delta


def resolve_data_files(batch_cfg, env_cfg):
    """fitting 대상 pt 파일 목록을 결정한다 (data/ 기준 상대경로).

    batch_cfg 가 없으면 기존처럼 task config 의 data_dir 1개를 반환.
    batch_cfg 가 있으면 data_files(명시 리스트) + data_glob(패턴) 을 합쳐
    중복 제거 후 정렬된 경로 리스트를 반환.
    """
    data_root = project_root() / "data"
    if batch_cfg is None:
        return [data_root / env_cfg.sim2real.data_dir]

    files = []
    for rel in batch_cfg.get("data_files") or []:
        files.append((data_root / rel).resolve())
    glob_pat = batch_cfg.get("data_glob")
    if glob_pat:
        files.extend(sorted(p.resolve() for p in data_root.glob(glob_pat)))

    seen, ordered = set(), []
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        if not f.exists():
            print(f"[WARN] data file 없음, 건너뜀: {f}")
            continue
        ordered.append(f)

    if not ordered:
        raise RuntimeError(
            f"config 에서 fitting 할 pt 를 찾지 못했습니다 (data_files/data_glob 확인): "
            f"{args_cli.config}"
        )
    return ordered


def run_single_fit(env, data_file, env_cfg, base_log_dir, max_iteration, nest_log):
    """pt 파일 1개에 대해 CMA-ES fitting 을 끝까지 수행한다.

    nest_log=True 면 결과를 base_log_dir/<pt_stem>/ 아래에 분리 저장한다 (batch).
    반환: simulation_app 가 아직 살아있으면 True (다음 파일 진행), 죽었으면 False.
    """
    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    device = env.unwrapped.device

    data = torch.load(data_file)
    time_data = data["time"].to(device)
    target_dof_pos = data["des_dof_pos"].to(device)
    measured_dof_pos = data["dof_pos"].to(device)

    # Use joint_names from data file if they match simulator joint names.
    # Some single-motor datasets store experiment labels such as "100Nm_J1";
    # in that case the column order follows the configured joint_order.
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
            "Data joint names are not present in the simulator articulation and "
            f"cannot be mapped by order. Missing={missing}, "
            f"sim_joints={articulation.joint_names}, joint_order={joint_order}"
        )

    # Map data columns to simulator joint indices
    data_joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in data_joint_names],
        device=device,
    )
    # Filter bounds_params to only the collected joints
    # bounds_params shape: (num_joints*3, 2) — armature, damping, friction
    bounds_params = env_cfg.sim2real.bounds_params.to(device)
    data_joint_order_indices = torch.tensor(
        [joint_order.index(name) for name in data_joint_names],
        device=device,
    )
    num_all_joints = len(joint_order)
    param_indices = []
    for group in range(3):  # armature, damping, frictionm dynamic(x)
        param_indices.append(group * num_all_joints + data_joint_order_indices)
    param_indices = torch.cat(param_indices)
    bounds_params = bounds_params[param_indices, :]

    # batch 반복 시: 직전 파일의 inference_mode 루프가 sim 내부 버퍼(joint_acc 등)를
    # inference tensor 로 만들어, 루프 밖에서 env.reset() 하면 inplace update 에러가 난다.
    # → 초기 reset/update 도 루프와 동일하게 inference_mode 안에서 수행한다.
    with torch.inference_mode():
        env.reset()

    # Regularization: armature만 ground truth 존재
    nominal_armature = articulation.data.default_joint_armature[0, data_joint_ids]
    num_data_joints = len(data_joint_names)
    # nominal_params layout: [armature, damping, friction]
    nominal_params = torch.zeros(3 * num_data_joints, device=device)
    nominal_params[:num_data_joints] = nominal_armature
    print(f"[INFO]: Nominal Armature: {nominal_armature}")
    print(f"[INFO]: Data joint IDs (sim): {data_joint_ids}")

    initial_dof_pos = (
        measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    )

    time_steps = time_data.shape[0]  # 한 세대 마무리 계산용도
    sim_dt = env.unwrapped.sim.cfg.dt

    # batch 모드면 pt 파일명(stem) 하위 폴더로 결과 분리
    log_dir = (
        str(pathlib.Path(base_log_dir) / pathlib.Path(data_file).stem)
        if nest_log
        else base_log_dir
    )

    opt = ActuatorCMAESOptimizer(
        bounds=bounds_params,
        population_size=env.unwrapped.num_envs,
        log_dir=log_dir,
        joint_order=data_joint_names,
        max_iteration=max_iteration,
        data=data,
        device=device,
        epsilon=env_cfg.sim2real.cmaes.epsilon,
        sigma=env_cfg.sim2real.cmaes.sigma,
        save_interval=env_cfg.sim2real.cmaes.save_interval,
        save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process,
        nominal_params=nominal_params,
    )
    # 어떤 pt 로부터 나온 결과인지 추적용 메모
    try:
        with open(
            pathlib.Path(opt.writer.log_dir) / "source.txt", "w"
        ) as _sf:
            _sf.write(str(data_file) + "\n")
    except OSError:
        pass

    with torch.inference_mode():
        env.reset()
        opt.update_simulator(articulation, data_joint_ids, initial_dof_pos)
        previous_sim_dof_pos = articulation.data.joint_pos[:, data_joint_ids].clone()
    unwrapped_sim_dof_pos = initial_dof_pos.clone()

    counter = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            current_sim_dof_pos = env.unwrapped.scene.articulations[
                "robot"
            ].data.joint_pos[:, data_joint_ids]
            if counter > 0:
                # 5/11
                # delta 를 -pi ~ pi 로 범위 조절하고 sim dof pos 누적시키기
                unwrapped_sim_dof_pos = _unwrap_joint_positions(
                    current_sim_dof_pos, previous_sim_dof_pos, unwrapped_sim_dof_pos
                )
            previous_sim_dof_pos = current_sim_dof_pos.clone()

            opt.tell(
                unwrapped_sim_dof_pos,  # 현재 sim 에 누적된 position data (unwrapped)
                measured_dof_pos[counter, :]  # 수집한 데이터에서 뽑아옴
                .unsqueeze(0)
                .repeat(env.unwrapped.num_envs, 1),
            )
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
                    f"[INFO]: Step {counter * sim_dt:.1f} / {time_data[-1]:.1f} seconds ({counter / time_steps * 100:.1f} %)"
                )
            if counter >= time_steps:
                print("[INFO]: Reached the end of the trajectory.")
                counter = 0
                opt.evolve()
                if opt.finished():
                    break
                env.reset()
                opt.update_simulator(
                    env.unwrapped.scene["robot"], data_joint_ids, initial_dof_pos
                )
                previous_sim_dof_pos = articulation.data.joint_pos[
                    :, data_joint_ids
                ].clone()
                unwrapped_sim_dof_pos = initial_dof_pos.clone()

    # close optimizer (이 pt 의 결과 마무리)
    opt.close()
    return simulation_app.is_running()


def main():
    """Actuator parameter fitting with CMA-ES (armature, damping, friction only).

    --config 지정 시 yaml 에 나열된 복수개의 pt 를 한 sim 세션에서 순차 fitting.
    """
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] {env.unwrapped.scene['robot'].cfg.spawn.joint_drive}")
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    base_log_dir = str(
        project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name
    )

    # 반복 실험 대상 pt 목록 결정
    data_files = resolve_data_files(batch_cfg, env_cfg)
    nest_log = batch_cfg is not None
    max_iteration = int(
        (batch_cfg or {}).get("max_iteration", env_cfg.sim2real.cmaes.max_iteration)
    )

    print(f"[INFO]: {len(data_files)} 개 pt 파일 fitting 예정 (task={args_cli.task}):")
    for f in data_files:
        print(f"  - {f}")

    for idx, data_file in enumerate(data_files):
        if not simulation_app.is_running():
            print("[INFO]: simulator 종료됨, 남은 실험 중단.")
            break
        print(
            f"\n===== [{idx + 1}/{len(data_files)}] fitting: "
            f"{pathlib.Path(data_file).name} ====="
        )
        still_running = run_single_fit(
            env, data_file, env_cfg, base_log_dir, max_iteration, nest_log
        )
        if not still_running:
            print("[INFO]: simulator 종료됨, 남은 실험 중단.")
            break

    print("[INFO]: 모든 실험 완료.")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
