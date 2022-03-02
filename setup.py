from setuptools import setup

if __name__ == '__main__':
    setup(
        name='surrol',
        version='0.1.0',
        description='SurRoL: An Open-source Reinforcement Learning Centered and '
                    'dVRK Compatible Platform for Surgical Robot Learning',
        author='Med-AIR@CUHK',
        keywords='simulation, medical robotics, dVRK, reinforcement learning',
        packages=[
            'surrol','surrol.tasks','surrol.gym','surrol.utils','surrol.robots','surrol.algorithms'
        ],
        package_dir={
             'surrol' : 'surrol',
             'surrol.tasks' : 'surrol/tasks',
             'surrol.gym' : 'surrol/gym',
             'surrol.utils' : 'surrol/utils',
             'surrol.robots' : 'surrol/robots',
             'surrol.algorithms' : 'surrol/algorithms'

         },
        install_requires=[
            "pybullet",
            "imageio",
            "imageio-ffmpeg",
            "opencv-python",
            "roboticstoolbox-python",
            "sympy",
            "wandb"
        ],
        extras_require={
        'stable-baselines3': [
                            "stable-baselines3[extra]"
                            ],
        }
    )
