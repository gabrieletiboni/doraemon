import pdb

import numpy as np
try:
    import mujoco
except ImportError:
    print('Warning! Unable to import mujoco.')
    pass


class Controller():
    """
    The base class for all controllers
    """
    def __init__(self, env):
        """
        :param env: the environment to control
        """
        self.env = env

    def get_control(self, command):
        """
        :description: The base function for calculating the control signal given
            command. This base class simply acts as a 'passthrough'
        :param command: the command signal
        :return: the control signal
        """
        return command

    def reset(self):
        """
        :description: the reset function is called at the end of each episode.
            Reset, e.g., your integral terms and previous values for derivative
            calculation here.
        """

class TorqueController(Controller):
    """
    Joint position controller with feedforward-term with fixed given desired acceleration.

    This controller uses a feedforward term with fixed given desired acceleration
    and a PD control term for corrections. This controller can also accept a target
    desired velocity to track, it doesn't just track steady states.
    """
    def __init__(self,
                 env):
        """
        :param env: the environment with the robot to control
        :param clip_position: whether to clip positions to joint limits
        :param clip_acceleration: whether to clip accelerations to robot limits
        """
        super().__init__(env)

        self.ctrl_format = 'action'


    def get_control(self, command):
        """
            command: torque to be applied
        """
        return command + self.gravity_compensation()

    def gravity_compensation(self):
        """
        :description: Calculate compensation terms
        :return: compensation terms
        """
        return self.env.data.qfrc_bias[self.env.arm_joint_index]

class FFJointPositionController(Controller):
    """
    Joint position controller with feedforward-term with fixed given desired acceleration.

    This controller uses a feedforward term with fixed given desired acceleration
    and a PD control term for corrections. This controller can also accept a target
    desired velocity to track, it doesn't just track steady states.
    """
    def __init__(self,
                 env,
                 clip_position=False,
                 clip_acceleration=False,
                 velocity_noise=False,
                 scale_kp: float = 1.,
                 scale_kd: float = 1.):
        """
        :param env: the environment with the robot to control
        :param clip_position: whether to clip positions to joint limits
        :param clip_acceleration: whether to clip accelerations to robot limits
        """
        super().__init__(env)
        # kp = np.array([1e3, 1e3, 1e3, 1e3, 1e2, 1e2, 1e2])
        # ki = np.array([1e4, 1e4, 1e4, 1e4, 1e3, 1e3, 1e3])
        # kd = np.array([1e3, 1e3, 1e3, 1e3, 1e2, 1e2, 1e2])
        kp = np.array([600, 600, 600, 600, 250, 150, 50]) * scale_kp
        kd = np.array([50, 50, 50, 20, 20, 20, 10]) * scale_kd
        self.kp, self.kd = kp, kd
        self.clip_position = clip_position
        self.clip_acceleration = clip_acceleration
        self.velocity_noise = velocity_noise

        self.ctrl_format = 'pos-vel-acc'


    def get_control(self, command):
        """
            command: (des_jpos, des_jvel, des_jacc)
        """
        des_jpos, des_jvel, des_jacc = command

        if self.clip_position:
            margin = 3 * np.pi / 180
            low = self.env.joint_qpos_min + margin - self.env.joint_qpos_shift
            high = self.env.joint_qpos_max - margin - self.env.joint_qpos_shift
            des_jpos = np.clip(des_jpos, low, high)
        
        current_pos = self.env.joint_pos
        current_vel = self.env.joint_vel
        if self.velocity_noise:
            current_vel += np.random.randn(7)*0.0011

        # Feedforward term
        m_q = self.get_m_q()
        feedforward = m_q @ des_jacc

        # PID control term
        pid_term = (des_jpos - current_pos) * self.kp + (des_jvel - current_vel) * self.kd

        # The control torque is the sum of those terms
        control = feedforward + pid_term

        if self.clip_acceleration:
            # Project torque to accelerations using the inverse of M_q
            ctrl_acc = np.linalg.inv(m_q) @ control

            # check inverse is correct
            assert np.all(np.isclose(m_q @ ctrl_acc, control))

            # Clamp the resulting accelerations to robot limits
            acc_clamped = np.clip(ctrl_acc, -.99*self.env.joint_qacc_max, .99*self.env.joint_qacc_max)

            # Project clamped accelerations to torques with M_q
            control = m_q @ acc_clamped

        # Add the compensation terms
        return control + self.gravity_compensation()

    def get_m_q(self):
        """
        :return: the mass-inertia matrix of the robot
        """
        model = self.env.model
        data = self.env.data
        model_nv = model.nv

        # nv x nv joint-space inertia matrix
        full_m_buffer = np.ndarray((model_nv,model_nv))
        """
            data.qM is inertia matrix in a custom sparse format that the user
            should not attempt to manipulate directly.
            One can convert qM to a dense matrix with mj_fullM, but this is
            slower because it no longer benefits from sparsity.
        """
        mujoco.mj_fullM(model, full_m_buffer, data.qM)
        full_m = full_m_buffer[0:7, 0:7]
        return full_m

    def gravity_compensation(self):
        """
        :description: Calculate compensation terms
        :return: compensation terms
        """
        return self.env.data.qfrc_bias[self.env.arm_joint_index]