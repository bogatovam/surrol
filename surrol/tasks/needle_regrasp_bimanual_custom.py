import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmsEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
    step
)
from surrol.utils.robotics import get_matrix_from_pose_2d
from surrol.const import ASSET_DIR_PATH

import time

from stable_baselines3.common.env_checker import check_env


class NeedleRegrasp_custom(PsmsEnv):
    ACTION_MODE = 'pitch'
    WORKSPACE_LIMITS1 = ((0.55, 0.6), (0.01, 0.08), (0.695, 0.745))
    WORKSPACE_LIMITS2 = ((0.55, 0.6), (-0.08, -0.01), (0.695, 0.745))
    SCALING = 5.

    def _env_setup(self):
        super(NeedleRegrasp_custom, self)._env_setup()
        self.has_object = True
        self._waypoint_goal = True

        # robot
        for psm, workspace_limits in ((self.psm1, self.workspace_limits1), (self.psm2, self.workspace_limits2)):
            pos = (workspace_limits[0].mean(),
                   workspace_limits[1].mean(),
                   workspace_limits[2].mean())
            # orn = p.getQuaternionFromEuler(np.deg2rad([0, np.random.uniform(-45, -135), -90]))
            orn = p.getQuaternionFromEuler(np.deg2rad([0, -90, -90]))  # reduce difficulty

            # psm.reset_joint(self.QPOS_PSM1)
            joint_positions = psm.inverse_kinematics((pos, orn), psm.EEF_LINK_INDEX)
            psm.reset_joint(joint_positions)

        self.block_gripper = False  # set the constraint
        psm = self.psm1
        workspace_limits = self.workspace_limits1

        # needle
        limits_span = (workspace_limits[:, 1] - workspace_limits[:, 0]) / 3
        sample_space = workspace_limits.copy()
        sample_space[:, 0] += limits_span
        sample_space[:, 1] -= limits_span

        # Load the needle
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (0.01 * self.SCALING, 0, 0),
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        # Needle appearance
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))

        # Make needle rigid
        self.obj_ids['rigid'].append(obj_id)

        self.obj_id, self.obj_link1, self.obj_link2 = self.obj_ids['rigid'][0], 4, 5

        while True:
            # open the jaw
            psm.open_jaw()
            # TODO: strange thing that if we use --num_env=1 with openai baselines, the qs vary before and after step!
            step(0.5)

            # set the position until the psm can grasp it
            pos_needle = np.random.uniform(low=sample_space[:, 0], high=sample_space[:, 1])
            pitch = np.random.uniform(low=-105., high=-75.)  # reduce difficulty
            orn_needle = p.getQuaternionFromEuler(np.deg2rad([-90, pitch, 90]))
            p.resetBasePositionAndOrientation(obj_id, pos_needle, orn_needle)

            # record the needle pose and move the psm to grasp the needle
            pos_waypoint, orn_waypoint = get_link_pose(obj_id, self.obj_link2)  # the right side waypoint
            orn_waypoint = np.rad2deg(p.getEulerFromQuaternion(orn_waypoint))
            p.resetBasePositionAndOrientation(obj_id, (0, 0, 0.01 * self.SCALING), (0, 0, 0, 1))

            # get the eef pose according to the needle pose
            orn_tip = p.getQuaternionFromEuler(np.deg2rad([90, -90 - orn_waypoint[1], 90]))
            pose_tip = [pos_waypoint + np.array([0.0015 * self.SCALING, 0, 0]), orn_tip]
            pose_eef = psm.pose_tip2eef(pose_tip)

            # move the psm
            pose_world = get_matrix_from_pose_2d(pose_eef)
            action_rcm = psm.pose_world2rcm(pose_world)
            success = psm.move(action_rcm)
            if success is False:
                continue
            step(1)
            p.resetBasePositionAndOrientation(obj_id, pos_needle, orn_needle)
            cid = p.createConstraint(obj_id, -1, -1, -1,
                                     p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos_needle,
                                     childFrameOrientation=orn_needle)
            psm.close_jaw()
            step(0.5)
            p.removeConstraint(cid)
            self._activate(0)
            self._step_callback()
            step(1)
            self._step_callback()
            if self._activated >= 0:
                break

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits2
        goal = workspace_limits.mean(axis=1) + np.random.randn(3) * 0.005 * self.SCALING
        goal.clip(workspace_limits[:, 0], workspace_limits[:, 1])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()

        # Initialising 6 waypoints
        self._waypoints = [None, None, None, None, None, None] 

        # Position of one grasping point on the needle
        pos_obj1, _ = get_link_pose(self.obj_id, self.obj_link2)

        # Position of second grasping point on the needle
        pos_obj2, _ = get_link_pose(self.obj_id, self.obj_link1)

        # Making positions ndarrays
        pos_obj1, pos_obj2 = np.array(pos_obj1), np.array(pos_obj2)

        # Distance between the two grasping points
        pos_dis = np.linalg.norm(pos_obj1 - pos_obj2)

        # Same pitch angels for both grasps
        pitch1, pitch2 = np.deg2rad(-30), np.deg2rad(-30)

        # Open gripper 
        jaw = 0.8

        ####################################################################
        ############# MOVING BOTH PSMS TO THE MIDDLE #######################
        ####################################################################

        # Position of the tip for the PSM1 (with needle)
        pos_tip1 = (pos_obj1[0] + 0.002 * self.SCALING, pos_dis / 2, pos_obj1[2])

        # Orientation of the tip for the PSM1 (with needle)
        orn_tip1 = p.getQuaternionFromEuler(np.deg2rad([90, -30, 90]))

        # Pose of the tip for the PSM1 (with needle)
        pose_tip1 = [pos_tip1, orn_tip1]

        # Position of the EE for the PSM1 (with needle)
        pos_eef1, _ = self.psm1.pose_tip2eef(pose_tip1)

        # Position of the tip for the PSM2 (without needle)
        pos_tip2 = (pos_obj1[0] - 0.002 * self.SCALING, - pos_dis / 2, pos_obj1[2])

        # Orientation of the tip for the PSM2 (without needle)
        orn_tip2 = p.getQuaternionFromEuler(np.deg2rad([90, -150, 90]))

        # Pose of the tip for the PSM2 (without needle)
        pose_tip2 = [pos_tip2, orn_tip2]

        # Position of the EE for the PSM2 (without needle)
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)

        # Move both PSMs to the middle
        self._waypoints[0] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, -jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, jaw])  

        ####################################################################
        ################## PSM2 APPROACH NEEDLE ############################
        ####################################################################

        # Move PSM1 a little back
        pose_tip1[0] = (pos_obj1[0], pos_dis / 2, pos_obj1[2])
        pos_eef1, _ = self.psm1.pose_tip2eef(pose_tip1)

        # Move PSM2 forwards towards the needle
        pose_tip2[0] = (pos_obj1[0] + 0.002 * self.SCALING, - pos_dis / 2, pos_obj1[2])
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)

        
        self._waypoints[1] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, -jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, jaw])

        ####################################################################
        ################## PSM2 GRASP, PSM1 RELEASE ########################
        ####################################################################  

        # PSM2 grasp                      
        self._waypoints[2] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, -jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, -jaw]) 

        # PSM1 RELEASE
        self._waypoints[3] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, -jaw])


        ####################################################################
        ############### ADJUST POSES TO AVOID COLLISION ####################
        ####################################################################

        # Move PSM1 a little back and to the right
        pose_tip1[0] = (pos_obj1[0] - 0.005 * self.SCALING, pos_dis / 2 + 0.01 * self.SCALING, pos_obj1[2])
        pos_eef1, _ = self.psm1.pose_tip2eef(pose_tip1)

        # Move PSM2 a little to the front
        pose_tip2[0] = (pos_obj1[0] + 0.005 * self.SCALING, - pos_dis / 2, pos_obj1[2])
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)

        self._waypoints[4] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, -jaw]) 
        
        ####################################################################
        #################### MOVE PSM2 TO RED DOT ##########################
        ####################################################################

        # PSM2 tip position must match goal position
        pose_tip2[0] = (self.goal[0], self.goal[1], self.goal[2])

        # Convert to EE position
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)

        # Place
        self._waypoints[5] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, -jaw])  

    def _meet_contact_constraint_requirement(self):
        """ add a contact constraint to the grasped needle to make it stable
        """
        return True

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # six waypoints executed in sequential order
        action = np.zeros(10)
        action[4], action[9] = 0.8, -0.8
        pitch_scaling = np.deg2rad(15)
        for i, waypoint in enumerate(self._waypoints):
            time.sleep(0.2)
            if waypoint is None:
                continue
            delta_pos1 = (waypoint[0: 3] - obs['observation'][0: 3]) / 0.01 / self.SCALING
            delta_pitch1 = ((waypoint[3] - obs['observation'][4]) / pitch_scaling).clip(-1, 1)
            delta_pos2 = (waypoint[5: 8] - obs['observation'][7: 10]) / 0.01 / self.SCALING
            delta_pitch2 = ((waypoint[8] - obs['observation'][11]) / pitch_scaling).clip(-1, 1)
            if np.abs(delta_pos1).max() > 1:
                delta_pos1 /= np.abs(delta_pos1).max()
            if np.abs(delta_pos2).max() > 1:
                delta_pos2 /= np.abs(delta_pos2).max()
            scale_factor = 0.5
            delta_pos1 *= scale_factor
            delta_pos2 *= scale_factor
            action = np.array([delta_pos1[0], delta_pos1[1], delta_pos1[2], delta_pitch1, waypoint[4],
                               delta_pos2[0], delta_pos2[1], delta_pos2[2], delta_pitch2, waypoint[9]])
            if np.linalg.norm(delta_pos1) * 0.01 / scale_factor < 1e-4 and np.abs(delta_pitch1) < 2. \
                    and np.linalg.norm(delta_pos2) * 0.01 / scale_factor < 1e-4 and np.abs(delta_pitch2) < 2.:
                self._waypoints[i] = None
            break


        return action


if __name__ == "__main__":
    env = NeedleRegrasp_custom(render_mode='human')  # create one process and corresponding env
    obs = env._get_obs()
    time.sleep(10)
    env.test()
    env.close()
    time.sleep(2)

    #env = NeedleRegrasp_custom()
    # If the environment don't follow the interface, an error will be thrown
    #check_env(env, warn=True)

    