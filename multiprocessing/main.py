from multi_pipe import SubprocVecEnv
from trans import check_done, check_state, check_reward, get_action
from pg import PG
import tensorflow as tf

def train():
    num_process = 3

    sess = tf.Session()
    pg = PG(sess, 4, 2, 0.0001)
    sess.run(tf.global_variables_initializer())

    sub = SubprocVecEnv(num_process, render=True)
    for i in range(3):
        each_terminal = [False] * num_process
        terminal = False
        state = sub.reset()
        while not terminal:
            actions = get_action(pg, each_terminal, num_process, state)
            info = sub.step(actions)

            each_terminal, terminal = check_done(info, num_process)
            next_state = check_state(info, num_process)
            reward = check_reward(info, num_process)
            print(reward)
            #print(each_terminal, terminal)
            #print(next_state)
            #print(state)
            #print(terminal)
            #print(reward)

            state = next_state

    sub.close()

if __name__=='__main__':
    train()