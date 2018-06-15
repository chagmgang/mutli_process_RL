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

def get_action(Policy, each_terminal, num_process, state):
    actions, v_preds = [], []
    
    for i in range(num_process):
        if not each_terminal[i]:
            s = np.stack([state[i]]).astype(dtype=np.float32)
            act, v_pred = Policy.act(obs=s, stochastic=True)
            act, v_pred = np.asscalar(act), np.asscalar(v_pred)
            actions.append(act)
            v_preds.append(v_pred)
        else:
            actions.append('done')
            v_preds.append('done')
    return actions, v_preds