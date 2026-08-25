import torch
from isaaclab.utils import configclass

from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg
from pace_sim2real.tasks.manager_based.pace.ACT_CFG import (
    RI60_CFG,
    RI60_INPUT_ENCODER_CFG,
    RO80_INPUT_ENCODER_CFG,
    RO100_CFG,
    RO100_INPUT_ENCODER_CFG,
)

"""
Input-side nominal rotor inertia
RO80 / 100Nm  = 1.60e-4 kg m^2
RO100 / 300Nm = 5.29e-4 kg m^2
RI60 / 20Nm   = 3.305e-6 kg m^2

기존 output-side fit 결과를 motor-side 로 환산하면:
RO80  0.1556 / 23.04^2 ~= 2.93e-4 kg m^2
RO100 0.4512 / 23.04^2 ~= 8.50e-4 kg m^2
RI60  0.0087 / 26^2    ~= 1.29e-5 kg m^2
"""

MOTOR_SPECS: dict = {
    "100Nm": {
        "robot_cfg": RO80_INPUT_ENCODER_CFG,
        "data_dir": "508/100Nm_input_position_data.pt",
        "armature_bounds": (1e-5, 1e-3),
        "damping_bounds": (0.0, 1),
        "friction_bounds": (1e-5, 3.0),  # static >= dynamic 을 만족해야한다.
        # "dynamic_friction_bounds": (1e-5, 2.0),
    },
    # "300Nm": {
    #     "robot_cfg": RO100_CFG,
    #     "data_dir": "508/300Nm_output_position_data.pt",
    #     "armature_bounds": (1e-2, 0.8),
    #     "damping_bounds": (0.1, 8.0),  # viscous Damping : 0.1 - 1.5 Nms/rad
    #     "friction_bounds": (1.0, 4.0),  # static Friction : 2.0 - 5.5 Nm
    #     # "dynamic_friction_bounds": (1e-5, 3.0),  # coulomb friction
    # },
    "300Nm": {
        "robot_cfg": RO100_INPUT_ENCODER_CFG,
        "data_dir": "611/logs/300Nm_merged_input_position_data.pt",
        "armature_bounds": (5e-6, 2e-2),
        "damping_bounds": (1e-9, 1e-2),
        "friction_bounds": (5e-4, 1.0),
        # "dynamic_friction_bounds": (0.0, 1.0),
    },
    "20Nm": {
        "robot_cfg": RI60_CFG,
        "data_dir": "508/20Nm_output_position_data.pt",
        "armature_bounds": (1e-8, 1e-2),
        "damping_bounds": (0.0, 5e-1),
        "friction_bounds": (0.0, 1),
        # "dynamic_friction_bounds": (0.0, 1.0),
    },
}


def _make_env_cfg(motor_type: str) -> type:
    """motor_type 에 대응하는 subclass 를 동적으로 생성."""
    spec = MOTOR_SPECS[motor_type]
    arm_lo, arm_hi = spec["armature_bounds"]
    dmp_lo, dmp_hi = spec["damping_bounds"]
    fri_lo, fri_hi = spec["friction_bounds"]
    # dy_lo, dy_hi = spec["dynamic_friction_bounds"]

    @configclass
    class _PaceCfg(PaceCfg):
        robot_name = "Robot"
        data_dir = spec["data_dir"]
        # 4 * n_joints + 1 (armature, damping, friction, bias, delay)
        bounds_params: torch.Tensor = torch.zeros((3, 2))  # ! lower, upper bound
        joint_order: list[str] = ["J1"]

        def __post_init__(self):
            n = len(self.joint_order)

            # ===== armature: input-side motor rotor inertia =====
            self.bounds_params[:n, 0] = arm_lo
            self.bounds_params[:n, 1] = arm_hi

            # ===== viscous damping: input-side motor scale =====
            self.bounds_params[n : 2 * n, 0] = dmp_lo
            self.bounds_params[n : 2 * n, 1] = dmp_hi

            # ===== friction: input-side motor torque scale =====
            self.bounds_params[2 * n : 3 * n, 0] = fri_lo
            self.bounds_params[2 * n : 3 * n, 1] = fri_hi

            # self.bounds_params[3 * n : 4 * n, 0] = dy_lo
            # self.bounds_params[3 * n : 4 * n, 1] = dy_hi

    @configclass
    class _SceneCfg(PaceSim2realSceneCfg):
        robot = spec["robot_cfg"].replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot.init_state.pos = (0.0, 0.0, 0.5)

    @configclass
    class _EnvCfg(PaceSim2realEnvCfg):
        scene: _SceneCfg = _SceneCfg()
        sim2real: PaceCfg = _PaceCfg()
        sim2real.cmaes.epsilon = 1e-8

        def __post_init__(self):
            super().__post_init__()
            self.sim2real.cmaes.max_iteration = 200
            # robot sim and control settings
            self.sim.dt = 0.0025  # 400Hz simulation
            self.decimation = 1  # 400Hz control

    # 디버깅 시 클래스 식별을 위해 이름 부여
    _EnvCfg.__name__ = f"PaceEnvCfg_{motor_type}"
    _EnvCfg.__qualname__ = _EnvCfg.__name__
    return _EnvCfg


# 모듈 레벨에서 task 별 EnvCfg 자동 생성 (manager_based/__init__.py 가 이 이름들을 참조)
PaceEnvCfg_100Nm = _make_env_cfg("100Nm")
PaceEnvCfg_300Nm = _make_env_cfg("300Nm")
PaceEnvCfg_20Nm = _make_env_cfg("20Nm")
