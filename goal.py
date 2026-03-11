import numpy as np
import jac


class Goal(object):
    """
    A trivial goal that is always satisfied.
    """

    def __init__(self):
        pass

    def is_satisfied(self, state):
        """
        Determine if the query state satisfies this goal or not.
        """
        return True


class RelocateGoal(Goal):
    """
    The goal for relocating tasks.
    (i.e., pushing the target object into a circular goal region.)
    """

    def __init__(self, x_g=0.2, y_g=-0.2, r_g=0.1):
        """
        args: x_g: The x-coordinate of the center of the goal region.
              y_g: The y-coordinate of the center of the goal region.
              r_g: The radius of the goal region.
        """
        super(RelocateGoal, self).__init__()
        self.x_g, self.y_g, self.r_g = x_g, y_g, r_g

    def is_satisfied(self, state):
        """
        Check if the state satisfies the RelocateGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        """
        stateVec = state["stateVec"]
        x_tgt, y_tgt = stateVec[7], stateVec[8] # position of the target object
        if np.linalg.norm([x_tgt - self.x_g, y_tgt - self.y_g]) < self.r_g:
            return True
        else:
            return False


class GraspGoal(Goal):
    """
    The goal for grasping tasks.
    (i.e., approaching the end-effector to a pose that can grasp the target object.)
    """

    def __init__(self):
        super(GraspGoal, self).__init__()
        self.jac_solver = jac.JacSolver() # the jacobian solver

    def is_satisfied(self, state):
        """
        Check if the state satisfies the GraspGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        returns: True or False.
        """
        ########## TODO ##########
        # 1. Target cube pose [x, y, theta]
        target_pose = state["stateVec"][7:10]
        
        # 2. Get End-Effector pose via Forward Kinematics
        joint_values = state["stateVec"][:7]
        pos_ee_local, quat_ee = self.jac_solver.forward_kinematics(joint_values)
        
        # Convert EE to World Frame (Base at [-0.4, -0.2, 0])
        ee_x_world = pos_ee_local[0] - 0.4
        ee_y_world = pos_ee_local[1] - 0.2
        ee_theta = self.jac_solver.bullet_client.getEulerFromQuaternion(quat_ee)[2]

        # 3. Calculate relative differences
        dx = target_pose[0] - ee_x_world
        dy = target_pose[1] - ee_y_world

        # 4. Calculate d1 and d2 (Projection onto EE frame)
        d1 = abs(dx * np.cos(ee_theta) + dy * np.sin(ee_theta))
        d2 = abs(-dx * np.sin(ee_theta) + dy * np.cos(ee_theta))

        # 5. Calculate gamma (Symmetric angle difference)
        # Finding the difference relative to the nearest 90-degree axis
        angle_diff = (ee_theta - target_pose[2]) % (np.pi / 2)
        gamma = abs(angle_diff - np.pi / 2) if angle_diff > np.pi / 4 else abs(angle_diff)

        # 6. Check all thresholds
        return d1 < 0.01 and d2 < 0.02 and gamma < 0.2

        ##########################
        