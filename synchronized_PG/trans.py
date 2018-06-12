import numpy as np
from pg import discount_rewards

def memory_stack(memory, num_process, state_space, action_space):
    memory = np.array(memory)
    state, action, reward = np.empty(shape=[0, state_space]), np.empty(shape=[0, action_space]), np.empty(shape=[0, 1])
    for i in range(num_process):
        state_stack, action_stack, reward_stack = np.empty(shape=[0, state_space]), np.empty(shape=[0, action_space]), np.empty(shape=[0, 1])
        for j in range(memory.shape[0]):
            if type(memory[j][1][i]) != str:
                state_stack = np.vstack([state_stack, memory[j][0][i]])
                action_stack = np.vstack([action_stack, memory[j][1][i]])
                reward_stack = np.vstack([reward_stack, memory[j][2][i]])
        discounted_stack = discount_rewards(reward_stack)
        state = np.vstack([state, state_stack])
        action = np.vstack([action, action_stack])
        reward = np.vstack([reward, discounted_stack])
    return state, action, reward

def hot_action(actions, num_process, action_space):
    action_list = []
    for action in actions:
        if action == 'done':
            a = 'done'
        else:
            a = np.zeros(action_space)
            a[action] = 1
        action_list.append(a)
    return action_list

def get_action(pg, each_terminal, num_process, state):
    actions = []
    for i in range(num_process):
        if not each_terminal[i]:
            actions.append(pg.choose_action([state[i]]))
        else:
            actions.append('done')
    return actions

def check_reward(info, num_process):
    data, reward = [], []
    for i in info:
        data.append(list(i))
    for d in data:
        reward.append(d[1])
    return reward

def check_state(info, num_process):
    data, state = [], []
    for i in info:
        data.append(list(i))
    for d in data:
        state.append(d[0])
    return state

def check_done(info, num_process):
    data, done, all_done = [], [], False
    for i in info:
        data.append(list(i))
    for d in data:
        done.append(d[2])
    if sum(list(map(int, done))) == num_process:
        all_done = True
    return done, all_done