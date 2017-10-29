import time
import threading

from tensorpack.utils.fs import mkdir_p
from tensorpack.utils.stats import StatCounter
from tensorpack.RL.envbase import RLEnvironment, DiscreteActionSpace

from unity3d_env import Unity3DEnvironment

__all__ = ['GymEnv']
_ENV_LOCK = threading.Lock()

class Unity3DPlayer(RLEnvironment):
    '''
    ACTION_TABLE = [(0.5, 0.0), # Forward
                    (-0.5, 0.0), # Backward
                    (0.5, 1.0), # Forward-Right
                    (-0.5, 1.0), # Backward-Right
                    (0.5, -1.0), # Forward-Left
                    (-0.5, -1.0) ] # Backward-Left 
    '''
    ACTION_TABLE = [(2.7, 0.0),
                    (2.7, 1.5),
                    (2.7, -1.5)]

    def __init__(self, connection, skip=1, dumpdir=None, viz=False, auto_restart=True):
        if connection != None:
            with _ENV_LOCK:
                self.gymenv = Unity3DEnvironment(server_address=connection)
            self.use_dir = dumpdir
            self.skip = skip
            self.reset_stat()
            self.rwd_counter = StatCounter()
            self.restart_episode()
            self.auto_restart = auto_restart
            self.viz = viz

    def restart_episode(self):
        self.rwd_counter.reset()
        self._ob = self.gymenv.reset()

    def finish_episode(self):
        self.stats['score'].append(self.rwd_counter.sum)

    def current_state(self):
        if self.viz:
            self.gymenv.render()
            time.sleep(self.viz)
        return self._ob

    def action(self, act):
        env_act = self.ACTION_TABLE[act]
        for i in range(self.skip):
            self._ob, r, isOver, info = self.gymenv.step(env_act)
            if r < 0.0:
                isOver = True
            if isOver:
                break            
        self.rwd_counter.feed(r)
        if isOver:
            self.finish_episode()
            if self.auto_restart:
                self.restart_episode()
        return r, isOver

    def get_action_space(self):
        return DiscreteActionSpace(len(self.ACTION_TABLE))

    def close(self):
        self.gymenv.close()

if __name__ == '__main__':
    import sys
    ip = sys.argv[1]
    port = int(sys.argv[2])
    p = Unity3DPlayer(connection=(ip, port))
    p.restart_episode()
    try:
        for i in range(100000):
            key = input()
            act = 0
            print(key)
            if key == 'w':
                act = 0
            elif key == 's':
                act = 1
            elif key == 'e':
                act = 2
            elif key == 'c':
                act = 3
            elif key == 'q':
                act = 4
            elif key == 'z':
                act = 5
            r, done = p.action(act)
            print(r)
    except:
        p.close()

