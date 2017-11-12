import time
import threading

from tensorpack.utils.fs import mkdir_p
from tensorpack.utils.stats import StatCounter
from tensorpack.RL.envbase import RLEnvironment, DiscreteActionSpace
import numpy as np
from unityagents import UnityEnvironment

import cv2

__all__ = ['GymEnv']

ACTION_SCALE = 1.0

class Unity3DPlayer(RLEnvironment):
    '''
    ACTION_TABLE = [(0.5, 0.0), # Forward
                    (-0.5, 0.0), # Backward
                    (0.5, 1.0), # Forward-Right
                    (-0.5, 1.0), # Backward-Right
                    (0.5, -1.0), # Forward-Left
                    (-0.5, -1.0) ] # Backward-Left 
    '''
    ACTION_TABLE = [(1.0 * ACTION_SCALE, 0.0 * ACTION_SCALE),
                    (1.0 * ACTION_SCALE, 1.0 * ACTION_SCALE),
                    (1.0 * ACTION_SCALE, -1.0 * ACTION_SCALE)]

    def __init__(self, env_name, base_port, worker_id, mode, skip=1, dumpdir=None, viz=False, auto_restart=True):
        self.gymenv = UnityEnvironment(file_name=env_name, base_port=base_port, worker_id=worker_id)
        print(str(self.gymenv))
        self.skip = skip
        self.brain_idx = self.gymenv.brain_names[0]
        self.mode = mode
        self.reset_stat()
        self.rwd_counter = StatCounter()
        # Wait unity env ready
        time.sleep(5.0)        
        self.restart_episode()
        self.auto_restart = auto_restart
        self.viz = viz
        self.worker_id = worker_id

    def _process_state(self, s):
        s = (s * 255.0).astype(np.uint8)
        #s = cv2.cvtColor(s, cv2.COLOR_RGB2BGR)
        return s

    def restart_episode(self):
        self.rwd_counter.reset()
        self.rwd_counter.feed(0)
        env_info = self.gymenv.reset(train_mode=self.mode)[self.brain_idx]
        self._ob = self._process_state(env_info.observations[0][0])
        return self._ob

    def finish_episode(self):
        self.stats['score'].append(self.rwd_counter.sum)

    def current_state(self):
        #cv2.imwrite('state_%02d.png' % self.worker_id, self._ob)
        return self._ob

    def action(self, act):
        env_act = self.ACTION_TABLE[act]
        for i in range(self.skip):
            env_info = self.gymenv.step(np.asarray([env_act]))[self.brain_idx]
            self._ob = self._process_state(env_info.observations[0][0])
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            
            if done:
                if reward > 0:
                    reward = 1.0 - self.rwd_counter.sum
                break            
        self.rwd_counter.feed(reward)
        if done:
            self.finish_episode()
            if self.auto_restart:
                self.restart_episode()
        return reward, done

    def get_action_space(self):
        return DiscreteActionSpace(len(self.ACTION_TABLE))

    def close(self):
        self.gymenv.close()

if __name__ == '__main__':
    import sys
    from tqdm import *
    p = Unity3DPlayer(env_name='Follow-train', base_port=9000, worker_id=0, mode=False)
    p.restart_episode()
    try:
        while True:
            for i in tqdm(range(1000)):
                act = p.get_action_space().sample()
                r, done = p.action(act)
                obs = p.current_state()
                if done:
                    print (done)
            
    finally:
        p.close()

