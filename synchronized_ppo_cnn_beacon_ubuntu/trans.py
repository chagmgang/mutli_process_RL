import numpy as np

def memory_stack(memory, num_process, state_space, PPO):
    memory = np.array(memory)
    state_, action_, reward_, v_preds_next_, gaes_ = np.empty(shape=[0, state_space]), np.array([]), np.array([]), np.array([]), np.array([])
    for i in range(num_process):
        state_stack, action_stack, reward_stack, v_pred_stack = np.empty(shape=[0, state_space]), np.array([]), np.array([]), np.array([])
        for j in range(memory.shape[0]):
            if type(memory[j][1][i]) != str:
                state_stack = np.vstack([state_stack, memory[j][0][i]])
                action_stack = np.append(action_stack, memory[j][1][i])
                reward_stack = np.append(reward_stack, memory[j][2][i])
                v_pred_stack = np.append(v_pred_stack, memory[j][3][i])

        v_preds_next_stack = list(v_pred_stack)[1:] + [0]
        gaes = PPO.get_gaes(rewards=reward_stack, v_preds=v_pred_stack,
                            v_preds_next=v_preds_next_stack)
        gaes = np.array(gaes).astype(dtype=np.float32)
        gaes = (gaes)
        action_stack = np.array(action_stack).astype(dtype=np.int32)
        reward_stack = np.array(reward_stack).astype(dtype=np.float32)
        v_preds_next_stack = np.array(v_preds_next_stack).astype(dtype=np.float32)

        state_ = np.concatenate((state_, state_stack), axis=0)
        action_ = np.append(action_, action_stack)
        reward_ = np.append(reward_, reward_stack)
        v_preds_next_ = np.append(v_preds_next_, v_preds_next_stack)
        gaes_ = np.append(gaes_, gaes)
        
    return state_, action_, reward_, v_preds_next_, gaes_


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