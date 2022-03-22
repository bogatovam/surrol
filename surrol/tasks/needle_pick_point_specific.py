import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle,
    plot_coordinate_frame
)
from surrol.const import ASSET_DIR_PATH


class NeedlePickPointSpecific(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.

    # TODO: grasp is sometimes not stable; check how to fix it

    def _env_setup(self):
        super(NeedlePickPointSpecific, self)._env_setup()
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True

        # object orientation needs to match the goals
        self.goal_orientation_required = False

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1

        # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        #yaw = 0
        
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
                             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

        #self._grasping_point = np.random.randint(1,6)
        self._grasping_point = 3

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        position = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.04 * self.SCALING])

        # Goal orientation in Euler angles
        #orientation = np.array([np.pi / 2 * np.random.uniform(low=-1, high=1) * self.SCALING,
                                #np.pi / 2 * np.random.uniform(low=-1, high=1) * self.SCALING,
                                #np.pi / 2 * np.random.uniform(low=-1, high=1) * self.SCALING])
        
        orientation_euler = [0,-np.pi/2,0]
        orientation_quat = p.getQuaternionFromEuler(orientation_euler)

        goal = np.concatenate((position,orientation_quat))

        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        
        p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], self.goal[:3], self.goal[3:])
        
        plot_coordinate_frame(self.obj_ids['fixed'][0])

        self._waypoints = [None, None, None, None]  # four waypoints

        # Sample mid-point location
        #pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)

        # Sample one out of 5 grasping point poses
        pos_obj, orn_obj = get_link_pose(self.obj_id, self._grasping_point)

        self._waypoint_z_init = pos_obj[2]

        # Euler representation of needle orietation
        orn = p.getEulerFromQuaternion(orn_obj)

        # EE orientation in Euler angles
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)

        # EE yaw angle based on the needle orientation
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        # # for physical deployment only
        # print(" -> Needle pose: {}, {}".format(np.round(pos_obj, 4), np.round(orn_obj, 4)))
        # qs = self.psm1.get_current_joint_position()
        # joint_positions = self.psm1.inverse_kinematics(
        #     (np.array(pos_obj) + np.array([0, 0, (-0.0007 + 0.0102)]) * self.SCALING,
        #      p.getQuaternionFromEuler([-90 / 180 * np.pi, -0 / 180 * np.pi, yaw])),
        #     self.psm1.EEF_LINK_INDEX)
        # self.psm1.reset_joint(joint_positions)
        # print("qs: {}".format(joint_positions))
        # print("Cartesian: {}".format(self.psm1.get_current_position()))
        # self.psm1.reset_joint(qs)

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([self.goal[0], self.goal[1],
                                       self.goal[2] + 0.0102 * self.SCALING, yaw, -0.5])  # lift up

    def _get_obs(self) -> dict:

        # Robot tip pose and gripper state (3+3+1)
        robot_state = self._get_robot_state(idx=0)

        # Base link pose
        baselink_pos, baselink_orn = get_link_pose(self.obj_id, -1)
        needle_baselink_pos = np.array(baselink_pos)
        needle_baselink_rot = np.array(p.getEulerFromQuaternion(baselink_orn))

        # Grasping point pose
        pos, orn = get_link_pose(self.obj_id, self._grasping_point)
        #pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        grasping_point_pos = np.array(pos)
        grasping_point_rot = np.array(orn)

        # Grasping point poition, baselink orientation
        achieved_goal = np.concatenate((grasping_point_pos.copy(),baselink_orn.copy()))
        
        # Grasping point index
        grasp_point = np.array([self._grasping_point])

        # Robot pose, needle base link position, 
        observation = np.concatenate([
            robot_state, 
            needle_baselink_pos, 
            needle_baselink_rot, 
            grasp_point
        ])

        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }

        return obs

    def _step_callback(self):

        plot_coordinate_frame(self.obj_id,0,0.1)

        return super()._step_callback()

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self._grasping_point)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    def get_oracle_action_task_specific(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
                self._waypoints[i] = None
            break

        return action


if __name__ == "__main__":
    env = NeedlePickPointSpecific(render_mode='human')  # create one process and corresponding env

    env.reset()
    for i in range(5):

        env.test()
        
        #robot_state = env._get_robot_state(0)

        #print("Tip position: {}, Orientation: {}".format(robot_state[:3],p.getEulerFromQuaternion(robot_state[3:])))
        
    
    env.close()
    time.sleep(2)
