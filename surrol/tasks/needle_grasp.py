import os
import time
import numpy as np
import math

import time
from gym import spaces
import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    step,
    get_link_pose,
    wrap_angle,
    plot_coordinate_frame
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

        # Grasping params
        self.distance_threshold = 0.01
        self.grasping_threshold = 0.0021

        self.z_offset = 0.0042

        self.tip_locator = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                                      globalScaling=2)

    def _get_obs(self) -> dict:

        robot_state = self._get_robot_state(idx=0)

        # Needle midlink pose
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        waypoint_pos = np.array(pos)

        # rotations
        waypoint_rot = np.array(p.getEulerFromQuaternion(orn))

        achieved_goal = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])
        achieved_goal[2] += self.z_offset

        observation = np.concatenate([
            robot_state, waypoint_pos.ravel(),
            waypoint_rot.ravel()
        ])

        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

        return obs

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        # Needle mid-point
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        goal = np.array([pos[0], pos[1], pos[2]])

        return goal.copy()

    def _update_goal(self):

        # Needle mid-point
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        goal = np.array([pos[0], pos[1], pos[2]])

        p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], goal, (0, 0, 0, 1))

        self.goal = goal

    def _sample_goal_callback(self):
        """ Define waypoints
        """

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

        pos_obj = self.goal

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
            obs['observation'][2] += self.z_offset
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            cond1 = np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4
            cond2 = np.abs(delta_yaw) < 1e-2
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

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info=None):
        """ All sparse reward.
        The reward is 0 or -1.
        """

        # Identify successful grasp
        success = self._is_success(achieved_goal, desired_goal, info)

        # Initiates all rewards to -1
        reward = np.zeros_like(success) - 1.0

        # Give reward for reaching an approximate area of the grasping point
        d = goal_distance(achieved_goal, desired_goal)

        distance_condition = (d < self.distance_threshold)
        reward += distance_condition

        # Add reward for successfully grasping
        reward += success * 2.0

        return reward.astype(np.float32)

    def _is_success(self, achieved_goal, desired_goal, info):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        jaw_threshold = 0.22
        jaw_state = info['jaw_state']
        jaw_condition = (jaw_state < jaw_threshold)

        # Distance between the grasping point and EE tip
        d = goal_distance(achieved_goal, desired_goal)
        distance_condition = (d < self.grasping_threshold)

        overal_condition = np.logical_and(jaw_condition, distance_condition).astype(np.float32)

        return overal_condition

    def _step_callback(self):
        """ A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """

        # plot_coordinate_frame(self.obj_id,self.obj_link1,0.1)
        # plot_coordinate_frame(self.psm1.body,self.psm1.TIP_LINK_INDEX,lifeTime=0.1,offsets=[0,0,self.z_offset])

        # Update tip position indicator
        EE_tip = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])
        EE_tip[2] += self.z_offset
        p.resetBasePositionAndOrientation(self.tip_locator, EE_tip, (0, 0, 0, 1))

        # Update goal position
        self._update_goal()

        # time.sleep(0.2)

    def _get_info_space(self):
        info_space = dict(
            jaw_state=spaces.Box(-np.inf, np.inf, shape=(1,), dtype='float32'), )

        return info_space

    def get_info(self, obs=None):
        jaw_state = np.array(self.psm1.get_current_jaw_position())

        info = {'jaw_state': jaw_state}
        return {'is_success': self._is_success(obs['achieved_goal'], self.goal, info),
                'jaw_state': jaw_state,
                'goal_distance': goal_distance(obs['achieved_goal'], obs['desired_goal'])}

    @property
    def num_goals(self):
        return 1


if __name__ == "__main__":

    env = NeedleGrasp(render_mode='human')  # create one process and corresponding env

    for _ in range(10):
        env.test()
    env.close()
    time.sleep(2)
