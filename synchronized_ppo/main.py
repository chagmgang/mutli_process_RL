from multi_pipe import SubprocVecEnv
import numpy as np
import time
from trans import trans_data, check_done, get_action
def train():  
    num_process = 2
    sub = SubprocVecEnv(num_process, False)
    for i in range(100):
        obs_s = sub.reset()
        terminal, each_terminal = False, [False] * num_process
        global_step = 0
        while not terminal:
            global_step += 1
            a = get_action(each_terminal, num_process)
            info = sub.step(a, obs_s, [global_step]*num_process)
            obs_s, state, action, reward, done = trans_data(info, num_process)
            each_terminal, terminal = check_done(info, num_process)
        print(i)
    sub.close()

if __name__=='__main__':
    train()