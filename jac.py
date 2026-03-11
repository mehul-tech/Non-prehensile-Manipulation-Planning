import copy
import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc


class JacSolver(object):
    """
    The Jacobian solver for the 7-DoF Franka Panda robot.
    """

    def __init__(self):
        self.bullet_client = bc.BulletClient(connection_mode=p.DIRECT)
        self.bullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    def forward_kinematics(self, joint_values):
        """
        Calculate the Forward Kinematics of the robot given joint angle values.
        args: joint_values: The joint angle values of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:       pos: The position of the end-effector.
                            Type: numpy.ndarray [x, y, z]
                      quat: The orientation of the end-effector represented by quaternion.
                            Type: numpy.ndarray [x, y, z, w]
        """
        for j in range(7):
            self.bullet_client.resetJointState(self.panda, j, joint_values[j])
        ee_state = self.bullet_client.getLinkState(self.panda, linkIndex=11)
        pos, quat = np.array(ee_state[4]), np.array(ee_state[5])
        return pos, quat

    def get_jacobian_matrix(self, joint_values):
        """
        Numerically calculate the Jacobian matrix based on joint angles.
        args: joint_values: The joint angles of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:         J: The calculated Jacobian matrix.
                            Type: numpy.ndarray of shape (6, 7)
        """
        ########## TODO ##########
        # 1. Use the delta_q that worked best for your stability tests
        delta_q = 1e-2
        J = np.zeros(shape=(6, 7))
        
        # 2. Get the nominal (unperturbed) pose
        pos_nominal, quat_nominal = self.forward_kinematics(joint_values)
        
        for i in range(7):
            # 3. Perturb the i-th joint
            joint_perturbed = np.copy(joint_values)
            joint_perturbed[i] += delta_q
            
            # 4. Get the perturbed pose
            pos_perturbed, quat_perturbed = self.forward_kinematics(joint_perturbed)
            
            # 5. Calculate Linear Velocity component (v = dx/dq)
            linear_vel = (pos_perturbed - pos_nominal) / delta_q
            
            # 6. Calculate Angular Velocity component (omega)
            # Use conjugate for inversion: [-x, -y, -z, w]
            q_nominal_inv = np.array([-quat_nominal[0], -quat_nominal[1], -quat_nominal[2], quat_nominal[3]])
            
            # Relative rotation: q_diff = q_perturbed * q_nominal_inv
            _, q_diff = self.bullet_client.multiplyTransforms([0, 0, 0], quat_perturbed, 
                                                               [0, 0, 0], q_nominal_inv)
            
            # Convert to axis-angle representation
            axis, angle = self.bullet_client.getAxisAngleFromQuaternion(q_diff)
            
            # Normalize angle to [-pi, pi] to ensure shortest rotation path
            if angle > np.pi:
                angle -= 2 * np.pi
            elif angle < -np.pi:
                angle += 2 * np.pi
                
            angular_vel = (np.array(axis) * angle) / delta_q
            
            # 7. Fill the i-th column of the Jacobian
            J[0:3, i] = linear_vel
            J[3:6, i] = angular_vel
        ##########################   
        return J