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