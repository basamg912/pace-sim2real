import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real.utils.assetloader import LoadUrdfFileCfg

# ELMo Kp : 0.152

GEAR_RATIO_RO80 = 23.04
GEAR_RATIO_RO100 = 23.04
GEAR_RATIO_RI60 = 26.0

ROTOR_INERTIA_RO80 = 1.60e-4
ROTOR_INERTIA_RO100 = 5.29e-4
ROTOR_INERTIA_RI60 = 3.305e-6

ARMATURE_RO80 = (
    ROTOR_INERTIA_RO80 * GEAR_RATIO_RO80**2
)  # 0.084934656, output-side reflected
ARMATURE_RO100 = (
    ROTOR_INERTIA_RO100 * GEAR_RATIO_RO100**2
)  # 0.2808152064, output-side reflected
ARMATURE_RI60 = (
    ROTOR_INERTIA_RI60 * GEAR_RATIO_RI60**2
)  # 0.00223418, output-side reflected

NATURAL_FREQ = 5 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_RO80 = ARMATURE_RO80 * NATURAL_FREQ**2  # 83.8271454615
STIFFNESS_RO100 = ARMATURE_RO100 * NATURAL_FREQ**2  # 277.153499682
STIFFNESS_RI60 = ARMATURE_RI60 * NATURAL_FREQ**2  # 1.73155447344
DAMPING_RO80 = 2.0 * DAMPING_RATIO * ARMATURE_RO80 * NATURAL_FREQ  # 10.6732036527
DAMPING_RO100 = 2.0 * DAMPING_RATIO * ARMATURE_RO100 * NATURAL_FREQ  # 35.2882795767
DAMPING_RI60 = 2.0 * DAMPING_RATIO * ARMATURE_RI60 * NATURAL_FREQ  # 0.220468362951

ELMO_POSITION_KP = 0.152
ELMO_POSITION_KD = 0.0
MOTOR_SIDE_VELOCITY_LIMIT = 999999.0

RO100_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="RO100/RO100_motor.urdf",
        fix_base=True,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness={".*": 0.0},
                damping={".*": 0.0},
            )
        ),
    ),
    actuators={
        "motor": PaceDCMotorCfg(
            joint_names_expr=["J1"],
            armature=0.5171,
            # 출력단(post-gearbox, N=23.04) 환산 한계
            effort_limit=10000,  # 12 Nm × N (peak torque)
            saturation_effort=10000,  # 12 Nm × N (stall torque)
            velocity_limit=10000,  # output-side actuator limit
            stiffness=927.4272,  # 520 데이터 기준 24243.22 / 522 데이터기준 927.4272
            damping=6.9211,  # 520 데이터 기준 192.4065 / 522 데이터 기준 6.9211
            friction=2.5717,
            # dynamic_friction=2.0040,
            viscous_friction=5.2523,
            encoder_bias=[0.0] * 1,
        ),
    },
)

RO100_INPUT_ENCODER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="RO100/RO100_motor.urdf",
        fix_base=True,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    ),
    actuators={
        "motor": PaceDCMotorCfg(
            joint_names_expr=["J1"],
            # Input encoder fit: all quantities are motor-side, pre-gearbox.
            effort_limit=12,  # 12Nm
            saturation_effort=12,
            velocity_limit=MOTOR_SIDE_VELOCITY_LIMIT,
            stiffness=30837.65,
            damping=218.7067,
            friction=0.055,
            armature=0.001309,  # ROTOR_INERTIA_RO100 5.29e-4 / Pace : 0.001155
            encoder_bias=[0.0] * 1,
        ),
    },
)

RO80_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="RO80/RO80_motor.urdf",
        fix_base=True,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    ),
    actuators={
        "motor": PaceDCMotorCfg(
            joint_names_expr=["J1"],
            effort_limit=92.16,  # 4.0 Nm × N (peak torque)
            saturation_effort=92.16,  # 4.0 Nm × N (stall torque)
            velocity_limit=50.0,  # output-side actuator limit
            stiffness=0.152,  # ELMo logging PD Kp
            damping=0.0,  # ELMo logging PD Kd
            friction=0.553,  # 0.024 Nm × N (cogging 출력단 환산)
            armature=ARMATURE_RO80,  # 1.60e-4 × N² (rotor inertia 출력단 반사)
            encoder_bias=[0.0] * 1,
            max_delay=0,
        ),
    },
)

RO80_INPUT_ENCODER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="RO80/RO80_motor.urdf",
        fix_base=True,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    ),
    actuators={
        "motor": PaceDCMotorCfg(
            joint_names_expr=["J1"],
            # Input encoder fit: all quantities are motor-side, pre-gearbox.
            effort_limit=4.0,
            saturation_effort=4.0,
            velocity_limit=MOTOR_SIDE_VELOCITY_LIMIT,
            stiffness=ELMO_POSITION_KP,
            damping=ELMO_POSITION_KD,
            friction=0.024,
            armature=ROTOR_INERTIA_RO80,
            encoder_bias=[0.0] * 1,
            max_delay=0,
        ),
    },
)

RI60_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="RI60/RI60_motor.urdf",
        fix_base=True,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    ),
    actuators={
        "motor": PaceDCMotorCfg(
            joint_names_expr=["J1"],
            # 출력단(post-gearbox, T-Motor RI60 Planetary N=26) 환산 한계
            effort_limit=10000,  # 1.63 Nm × N (peak torque)
            saturation_effort=10000,  # 1.63 Nm × N (stall torque)
            velocity_limit=10000,  # output-side actuator limit
            # PD gain — RI60 J 가 작아 RO 의 Kp/Kd 그대로면 ωₙ 과도 → 별도 산정
            stiffness=86.5,  # ωₙ² × J  (튜닝값 — 갱신된 ARMATURE_RI60=2.234e-3 기준 재산정 가능)
            damping=1.95,  # 2 × ωₙ × J
            friction=0.00,  # cogging 미제공 — fit 시 CMA-ES 가 덮어씀
            armature=ARMATURE_RI60,  # 3.305e-6 × 26² ≈ 0.002234 (rotor inertia 출력단 반사)
            encoder_bias=[0.0] * 1,
            max_delay=0,
        ),
    },
)
# 4/27 실험 결과
# Output 기준 1.286e-5 / Inertia 모터 스펙 기준 3.3e-6
RI60_INPUT_ENCODER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="RI60/RI60_motor.urdf",
        fix_base=True,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    ),
    actuators={
        "motor": PaceDCMotorCfg(
            joint_names_expr=["J1"],
            # Input encoder fit: all quantities are motor-side, pre-gearbox.
            effort_limit=20.0 / GEAR_RATIO_RI60,
            saturation_effort=20.0 / GEAR_RATIO_RI60,
            velocity_limit=MOTOR_SIDE_VELOCITY_LIMIT,
            stiffness=86.44999,  # 5/12 기준 : 86.44999
            damping=1.947072,  # 5/12 기준 : 1.947072
            friction=0.0,
            armature=ROTOR_INERTIA_RI60,
            encoder_bias=[0.0] * 1,
            max_delay=0,
        ),
    },
)

ACTUATOR_300NM = ArticulationCfg(
    spawn=LoadUrdfFileCfg(
        fix_base=True,
        asset_path="/home/kist/work/workspace/pace-humanoid/RO100/300NM.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "motor": PaceDCMotorCfg(
            joint_names_expr=["J1"],
            effort_limit=92.2,
            velocity_limit=11.6,
            stiffness=0.152,
            damping=0.0,
            armature=0.4512,
            saturation_effort=285.8,
            max_delay=10,
            encoder_bias=[0.0] * 1,
        ),
    },
)
