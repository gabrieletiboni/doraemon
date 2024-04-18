# Franka Emika Panda Description (MJCF)

Requires MuJoCo 2.3.3 or later.

Franka Panda Mujoco setup taken from [here](https://github.com/deepmind/mujoco_menagerie/tree/main/franka_emika_panda) and modified manually to:
- add custom fingers
- remove gripper joints and actuators
- remove tendons and equality constraints
- add table and box for pushing setup
- custom damping and frictionloss on the joints with SysId from real panda env