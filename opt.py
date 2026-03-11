import numpy as np
import copy
import sim
import goal
import rrt
import utils


########## TODO ##########

import time
from dataclasses import dataclass
import pybullet as p
from pdef import Bounds, ProblemDefinition
import os


@dataclass
class PerformanceLogs:
    path_magnitude: float
    clutter_disturbance: float
    runtime_sec: float
    utility_score: float

def build_optimization_env(simulation_instance, objective_goal):
    """
    Initializes a localized problem definition for path refinement.
    """
    
    env_pdef = ProblemDefinition(simulation_instance)
    s_dim = env_pdef.get_state_dimension()
    c_dim = env_pdef.get_control_dimension()

    # Define State Constraints
    s_limits = Bounds(s_dim)
    for i in range(sim.pandaNumDofs):
        s_limits.set_bounds(i, sim.pandaJointRange[i, 0], sim.pandaJointRange[i, 1])
    
    # Define Workspace and Orientation Bounds
    for i in range(sim.pandaNumDofs, s_dim):
        if (i - sim.pandaNumDofs) % 3 == 2:
            s_limits.set_bounds(i, -np.pi, np.pi)
        else:
            s_limits.set_bounds(i, -0.3, 0.3)
    env_pdef.set_state_bounds(s_limits)

    # Define Command Constraints
    cmd_limits = Bounds(c_dim)
    cmd_limits.set_bounds(0, -0.2, 0.2)
    cmd_limits.set_bounds(1, -0.2, 0.2)
    cmd_limits.set_bounds(2, -1.0, 1.0)
    cmd_limits.set_bounds(3, 0.4, 0.6)
    env_pdef.set_control_bounds(cmd_limits)
    env_pdef.set_goal(objective_goal)
    
    return env_pdef

def fetch_current_robot_context(sim_handle):
    """
    Captures the current joint and object states into a state dictionary.
    """
    angles, _, _ = sim_handle.get_joint_states()
    snapshot = list(angles[0:sim.pandaNumDofs])
    for item in sim_handle.objects:
        loc, rot_quat = sim_handle.bullet_client.getBasePositionAndOrientation(item)
        euler_rot = sim_handle.bullet_client.getEulerFromQuaternion(rot_quat)
        snapshot += [loc[0], loc[1], euler_rot[2]]
    return {"stateID": -1, "stateVec": np.array(snapshot)}

def get_aligned_grasp_angle(target_yaw, current_yaw):
    """
    Calculates the closest 90-degree orthogonal grasp orientation.
    """
    # Uses modulo to handle cube symmetry
    offsets = [target_yaw + n * (np.pi / 2.0) for n in range(4)]
    return min(offsets, key=lambda val: abs(((val - current_yaw) + np.pi) % (2 * np.pi) - np.pi))

def bind_object_to_hand(sim_handle):
    """
    Creates a physical constraint to simulate a firm grasp on the target.
    """
    target_id = sim_handle.objects[0]
    gripper_info = sim_handle.bullet_client.getLinkState(sim_handle.panda, linkIndex=11)
    g_pos, g_quat = gripper_info[4], gripper_info[5]
    t_pos, t_quat = sim_handle.bullet_client.getBasePositionAndOrientation(target_id)

    # Calculate Relative Transform
    inv_g_pos, inv_g_quat = sim_handle.bullet_client.invertTransform(g_pos, g_quat)
    local_pos, local_quat = sim_handle.bullet_client.multiplyTransforms(inv_g_pos, inv_g_quat, t_pos, t_quat)
    
    return sim_handle.bullet_client.createConstraint(
        sim_handle.panda, 11, target_id, -1,
        sim_handle.bullet_client.JOINT_FIXED,
        [0, 0, 0], local_pos, [0, 0, 0], local_quat
    )

def smart_manipulation_routine(log_output="logs/task4_analysis.txt"):
    """
    Executes a high-precision approach, grasp, and relocation sequence.
    """
    if not os.path.exists("logs"): os.makedirs("logs")

    # Headless setup for calculation
    physics_engine = utils.setup_bullet_client(p.DIRECT)
    virtual_sim = sim.PandaSim(physics_engine)
    utils.setup_env(virtual_sim)
    
    final_dest = goal.RelocateGoal()
    capture_goal = goal.GraspGoal()
    env_config = build_optimization_env(virtual_sim, final_dest)

    init_actions, final_actions = [], []
    has_captured, has_relocated = False, False

    with open(log_output, "w") as logger:
        logger.write("--- Optimization Phase Initiated ---\n")

        # PHASE 1: PRECISION APPROACH
        for tick in range(30):
            current_ctx = fetch_current_robot_context(virtual_sim)
            if capture_goal.is_satisfied(current_ctx):
                has_captured = True
                break

            # Controller Math
            ee_pos_raw, ee_quat_raw = virtual_sim.get_ee_pose()
            ee_yaw = physics_engine.getEulerFromQuaternion(ee_quat_raw)[2]
            cube_data = current_ctx["stateVec"][7:10]
            
            target_yaw = get_aligned_grasp_angle(cube_data[2], ee_yaw)
            rot_error = ((target_yaw - ee_yaw) + np.pi) % (2 * np.pi) - np.pi
            pos_error = cube_data[0:2] - np.array(ee_pos_raw[0:2])

            # PD-style Gains
            gain = 1.8 if np.linalg.norm(pos_error) > 0.03 else 1.1
            v_xy = np.clip(gain * pos_error, -0.14, 0.14)
            v_theta = np.clip(2.2 * rot_error, -0.8, 0.8)
            
            move_cmd = np.clip([v_xy[0], v_xy[1], v_theta, 0.4], 
                               env_config.bounds_ctrl.low, env_config.bounds_ctrl.high)

            _, is_legal = virtual_sim.execute(move_cmd)
            init_actions.append(move_cmd)
            if not is_legal: break

        if has_captured:
            virtual_sim.grasp()
            bind_object_to_hand(virtual_sim)
            logger.write("Grasp Secure\n")

            # PHASE 2: RELOCATION
            target_coords = np.array([final_dest.x_g, final_dest.y_g])
            for tick in range(60):
                current_ctx = fetch_current_robot_context(virtual_sim)
                if final_dest.is_satisfied(current_ctx):
                    has_relocated = True
                    break

                obj_loc = current_ctx["stateVec"][7:9]
                ee_loc, _ = virtual_sim.get_ee_pose()
                
                err_to_goal = target_coords - obj_loc
                err_in_hand = obj_loc - np.array(ee_loc[0:2])
                
                v_reloc = (0.8 * err_to_goal) + (0.7 * err_in_hand)
                v_reloc = np.clip(v_reloc, -0.05, 0.05)
                
                reloc_cmd = np.clip([v_reloc[0], v_reloc[1], 0, 0.3], 
                                    env_config.bounds_ctrl.low, env_config.bounds_ctrl.high)
                
                _, is_legal = virtual_sim.execute(reloc_cmd)
                final_actions.append(reloc_cmd)
                if not is_legal: break

    physics_engine.disconnect()
    return {"success": has_relocated, "approach_cmds": init_actions, "deliver_cmds": final_actions}

##########################
