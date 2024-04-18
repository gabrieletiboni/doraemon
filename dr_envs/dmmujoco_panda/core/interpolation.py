import pdb

import numpy as np



class Repeater():
    def __init__(self, num):
        self.num = num

    def __call__(self, action):
        for _ in range(self.num):
            yield action

    def reset(self):
        return

class AccelerationIntegrator(Repeater):
    """
        Given a desired acceleration, compute the integrated
        jpos and jvel in a span of 20ms in the future.
    """
    def __init__(self, num, dt, velocity_noise=False):
        super().__init__(num)
        self._dt = dt
        self.velocity_noise = velocity_noise

    def __call__(self, cmd):
        cur_pos, cur_vel, acc = cmd
        # if self.velocity_noise:
        #     cur_vel += np.random.randn(7)*0.0011
        dt = self._dt
        for n in range(self.num):
            t = n * dt
            target_pos = cur_pos + t * cur_vel + 0.5 * acc * t ** 2
            target_vel = cur_vel + t * acc
            yield (target_pos, target_vel, acc)