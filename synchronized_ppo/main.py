from multi_pipe import SubprocVecEnv
import numpy as np
import time
from trans import trans_data, check_done, get_action
from policy_net import Policy_net
from ppo import PPOTrain
import tensorflow as tf

def train():  
    num_process = 2
    sub = SubprocVecEnv(num_process, False)
    Policy = Policy_net('policy', 16*16*2, 4)
    Old_Policy = Policy_net('old_policy', 16*16*2, 4)
    PPO = PPOTrain(Policy, Old_Policy, gamma=0.95)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            info = sub.reset()
            terminal, each_terminal = False, [False] * num_process
            global_step = 0
            obs_s, state, action, reward, done = trans_data(info, num_process)
            while not terminal:
                global_step += 1
                a, v_pred = get_action(Policy, each_terminal, num_process, state)
                info = sub.step(a, obs_s, [global_step]*num_process)
                obs_s, next_state, action, reward, done = trans_data(info, num_process)
                each_terminal, terminal = check_done(info, num_process)

                state = next_state
        sub.close()

if __name__=='__main__':
    train()