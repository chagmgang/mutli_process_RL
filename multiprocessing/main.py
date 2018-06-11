from multi_pipe import SubprocVecEnv
from trans import check_done, check_state, check_reward, get_action, hot_action, memory_stack
from pg import PG
import tensorflow as tf
import numpy as np

def train():
    num_process = 3
    state_space = 4
    action_space = 2
    sess = tf.Session()
    pg = PG(sess, state_space, action_space, 0.0001)
    sess.run(tf.global_variables_initializer())
    sub = SubprocVecEnv(num_process, render=False)
    for i in range(1):
        each_terminal = [False] * num_process
        terminal = False
        state = sub.reset()
        memory = []
        global_step = 0
        while not terminal:
            global_step += 1
            actions = get_action(pg, each_terminal, num_process, state)
            info = sub.step(actions)

            one_hot_action = hot_action(actions, num_process, action_space)

            each_terminal, terminal = check_done(info, num_process)
            next_state = check_state(info, num_process)
            reward = check_reward(info, num_process)
            
            memory.append([state, one_hot_action, reward])
            if terminal:
                a = (memory_stack(memory, num_process))
            state = next_state
    
    sub.close()

if __name__=='__main__':
    train()