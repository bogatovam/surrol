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
            'surrol','surrol.tasks','surrol.gym','surrol.utils','surrol.robots' 
        ],
        package_dir={
             'surrol' : 'surrol',
             'surrol.tasks' : 'surrol/tasks',
             'surrol.gym' : 'surrol/gym',
             'surrol.utils' : 'surrol/utils',
             'surrol.robots' : 'surrol/robots',

         },
        install_requires=[
            "tensorflow-gpu==1.14",
            "gym==0.15.6",
            "pybullet>=3.0.7",
            "numpy>=1.21.1",
            "scipy",
            "pandas",
            "imageio",
            "imageio-ffmpeg",
            "opencv-python",
            "roboticstoolbox-python",
            "mpi4py",
            "sympy"
        ]
    )
