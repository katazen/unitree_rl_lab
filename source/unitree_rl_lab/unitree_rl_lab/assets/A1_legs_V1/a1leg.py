import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.95


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
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
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


A1leg_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_29dof_rev_1_0.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path="/home/woan/workspace/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/A1_legs_V1/mjcf/A1_legs/A1_legs.usd",
        # usd_path="/home/woan/workspace/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/A1_legs_V1/urdf/A1-legs_V1/A1-legs_V1.usd",
    ),
    articulation_root_prim_path='/base/base',
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*1": -0.1,
            ".*4": 0.2,
            ".*5": -0.1,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "leg": ImplicitActuatorCfg(
            joint_names_expr=[".*1", ".*2", ".*3", ".*4", ".*5", ".*6"],
            effort_limit_sim={
                ".*1": 27.0,
                ".*2": 27.0,
                ".*3": 27.0,
                ".*4": 27.0,
                ".*5": 7.0,
                ".*6": 7.0,
            },
            velocity_limit_sim=14.0,
            stiffness={
                ".*1": 100.0,
                ".*2": 100.0,
                ".*3": 100.0,
                ".*4": 100.0,
                ".*5": 10.0,
                ".*6": 10.0,
            },
            damping={
                ".*1": 1.0,
                ".*2": 1.0,
                ".*3": 1.0,
                ".*4": 1.0,
                ".*5": 1.0,
                ".*6": 1.0,
            },
            armature=0.01,
        ),
    },
    joint_sdk_names=['joint_R1',
                     'joint_R2',
                     'joint_R3',
                     'joint_R4',
                     'joint_R5',
                     'joint_R6',
                     'joint_L1',
                     'joint_L2',
                     'joint_L3',
                     'joint_L4',
                     'joint_L5',
                     'joint_L6'],
)
