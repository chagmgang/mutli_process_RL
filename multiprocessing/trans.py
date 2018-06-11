import numpy as np

def memory_stack(memory, num_process):
    memory = np.array(memory)
    state_inp = np.zeros((num_process, memory.shape[0]))
    reward_inp = np.zeros((num_process, memory.shape[0]))
    action_inp = np.zeros((num_process, memory.shape[0]))
    for i in range(num_process):
        for j in range(memory.shape[0]):
            if memory[j][1][i] != 'done':
                print(memory[j][0][i])      #state
                print(memory[j][1][i])      #action
                print(memory[j][2][i])      #reward
                
    return state_inp, reward_inp, action_inp

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