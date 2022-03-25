import os
import time
import numpy as np
import math

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle
)
from surrol.const import ASSET_DIR_PATH

from surrol.tasks.psm_env import goal_distance


class NeedleGrasp(PsmEnv):
    """
    Refer to Gym FetchReach
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/reach.py
    """
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.

    def _env_setup(self):
        super(NeedleGrasp, self)._env_setup()
        self.has_object = True
        self._waypoint_goal = True

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        self._contact_approx = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(10, 10, 10))
        self.obj_ids['fixed'].append(obj_id)  # 1

        # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        goal = np.array([pos[0], pos[1], pos[2]])

        return goal.copy()


    def _get_obs(self) -> dict:

        robot_state = self._get_robot_state(idx=0)
        # TODO: may need to modify

        # Needle baselink position
        #pos, _ = get_link_pose(self.obj_id, -1)
        #object_pos = np.array(pos)

        # Needle midlink pose
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        waypoint_pos = np.array(pos)
        # rotations
        waypoint_rot = np.array(p.getEulerFromQuaternion(orn))
        
        achieved_goal = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])

        observation = np.concatenate([
            robot_state, waypoint_pos.ravel(),
            waypoint_rot.ravel()  # achieved_goal.copy(),
        ])

        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        return obs

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None]  # four waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
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

    def get_oracle_action_task_specific(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
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

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ All sparse reward.
        The reward is 0 or -1.
        """

        completion_reward = 3.

        if len(achieved_goal.shape) > 1:
            reward = np.zeros(info.shape)
            info[0]['is_success'] = 1.0
            for i,s in enumerate(info):
                if s['is_success']:
                    reward[i] = completion_reward
                else:
                    d_x = abs(achieved_goal[i][0] - desired_goal[i][0])
                    d_y = abs(achieved_goal[i][1] - desired_goal[i][1])
                    d_z = abs(achieved_goal[i][2] - desired_goal[i][2])

                    d = (d_x + d_y + d_z) * 10

                    d = 1/d

                    reward[i] = d

        else:
            
            if info['is_success']:
                    reward = completion_reward
            else:
                d_x = abs(achieved_goal[0] - desired_goal[0])
                d_y = abs(achieved_goal[1] - desired_goal[1])
                d_z = abs(achieved_goal[2] - desired_goal[2])

                d = (d_x + d_y + d_z) * 10

                d = 1/d

                reward = d
                
        return reward



        # d = goal_distance(achieved_goal, desired_goal)
        # return - (d > self.distance_threshold).astype(np.float32)
        


    def _is_success(self, achieved_goal, desired_goal, info=None):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal, desired_goal)

        return (d < self.distance_threshold).astype(np.float32)

if __name__ == "__main__":

    env = NeedleGrasp(render_mode='human',seed=1)  # create one process and corresponding env

    for _ in range(1):
        env.test()
    env.close()
    time.sleep(2)
