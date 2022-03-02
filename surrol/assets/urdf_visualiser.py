import os
import pybullet as p
import pybullet_data
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle,
    reset_camera
)
import time
import numpy as np

from surrol.const import ROOT_DIR_PATH, ASSET_DIR_PATH

cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))
p.setGravity(0, 0, -9.81)
reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * 1,
                         target=(-0.05 * 1, 0, 0.36 * 1))

p.loadURDF("plane.urdf", (0, 0, -0.001))

p.loadURDF(os.path.join(ASSET_DIR_PATH, 'table/table.urdf'),
                   np.array((0.5, 0, 0.001)) * 1,
                   p.getQuaternionFromEuler((0, 0, 0)),
                   globalScaling=1)

tray_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array((0.55, 0, 0.6751)) * 1,
                            p.getQuaternionFromEuler((0, 0, 0)),
                            globalScaling=1)


sphere1_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                            globalScaling=1)
sphere2_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                            globalScaling=1)

needle_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (0.55,  # TODO: scaling
                             0.0,
                             0.695),
                            p.getQuaternionFromEuler((0, 0, 0)),
                            useFixedBase=False,
                            globalScaling=1)

link_position1 = p.getLinkState(needle_id,4)
link_position2 = p.getLinkState(needle_id,0)

print("Link position: {}, orientation: {}".format(link_position1[0],np.degrees(p.getEulerFromQuaternion(link_position1[1]))))

p.resetBasePositionAndOrientation(sphere1_id, link_position1[0], (0, 0, 0, 1))

p.resetBasePositionAndOrientation(sphere2_id, link_position2[0], (0, 0, 0, 1))

time.sleep(30)
p.disconnect()