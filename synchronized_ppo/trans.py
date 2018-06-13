import numpy as np

def trans_data(info, num_process):
    data, obs, state, action, reward, done = [], [], [], [], [], []
    for i in info:
        data.append(list(i))
    for d in data:
        obs.append(d[0])
        state.append(d[1])
        action.append(d[2])
        reward.append(d[3])
        done.append(d[4])
    return obs, state, action, reward, done

def check_done(info, num_process):
    data, done, all_done = [], [], False
    for i in info:
        data.append(list(i))
    for d in data:
        done.append(d[4])
    if sum(list(map(int, done))) == num_process:
        all_done = True
    return done, all_done

def get_action(each_terminal, num_process):
    actions = []
    for i in range(num_process):
        if not each_terminal[i]:
            actions.append(0)
        else:
            actions.append('done')
    return actions