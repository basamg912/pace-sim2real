from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim import UrdfFileCfg
from isaaclab.scene import InteractiveScene
from isaaclab.assets.articulation import Articulation
import torch


@configclass
class LoadUrdfFileCfg(UrdfFileCfg):
    fix_base = False
    replace_cylinders_with_capsules = True
    activate_contact_sensor: bool = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0, damping=0
        )
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
