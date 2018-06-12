from multi_pipe import SubprocVecEnv
from trans import check_done, check_state, check_reward, get_action, hot_action, memory_stack
from pg import PG
import tensorflow as tf
import numpy as np

def train():
    sess = tf.Session()
    r = tf.placeholder(tf.float32)
    rr = tf.summary.scalar('score', r)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./board/process_6', sess.graph)
    num_process = 6
    state_space = 4
    action_space = 2
    pg = PG(sess, state_space, action_space, 0.0)
    sess.run(tf.global_variables_initializer())
    sub = SubprocVecEnv(num_process, render=False)
    for i in range(200):
        each_terminal = [False] * num_process
        terminal = False
        state = sub.reset()
        memory = []
        global_step = 0
        score = 0
        while not terminal:
            global_step += 1
            actions = get_action(pg, each_terminal, num_process, state)
            info = sub.step(actions)

            one_hot_action = hot_action(actions, num_process, action_space)

            each_terminal, terminal = check_done(info, num_process)
            next_state = check_state(info, num_process)
            reward = check_reward(info, num_process)
            score += np.mean(reward)  
            memory.append([state, one_hot_action, reward])
            if terminal:
                state, action, discounted_reward = memory_stack(memory, num_process, state_space, action_space)
                _ = sess.run(pg.atrain, feed_dict={pg.X: state, pg.Y: action, pg.advantages: discounted_reward})
                summary = sess.run(merged, feed_dict={r: score})
                writer.add_summary(summary, i)
            state = next_state
    
    sub.close()

if __name__=='__main__':
    train()