from multiprocessing import Process, Pipe
import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib.pyplot as plt
from action_group import actAgent2Pysc2, no_operation
from state_group import obs2state, obs2distance
import numpy as np

FLAGS = flags.FLAGS
FLAGS(sys.argv)
_NO_OP = actions.FUNCTIONS.no_op.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

def worker(remote, visualize):
    env =  sc2_env.SC2Env(
                        map_name='MoveToBeacon',
                        agent_interface_format=sc2_env.parse_agent_interface_format(
                            feature_screen=32,
                            feature_minimap=32,
                            rgb_screen=None,
                            rgb_minimap=None,
                            action_space=None,
                            use_feature_units=False),
                        step_mul=4,
                        game_steps_per_episode=None,
                        disable_fog=False,
                    visualize=False)
    done = False
    while True:
        cmd, action, obs, global_step = remote.recv()
        end_step = 400
        if cmd == 'step':
            if not action == 'done':
                available_action, state = obs
                a = actAgent2Pysc2(action, state)
                obs = env.step(actions=[a])
                distance = obs2distance(obs)
                state = obs2state(obs)
                obs = (obs[0].observation.available_actions, state)
                
                #reward shaping

                reward = -0.1
                if distance < 0.03 or global_step == end_step - 1:
                    if distance < 0.03:
                        reward = 1
                    if global_step == end_step - 1:
                        reward = -1
                    done = True
                remote.send((obs, state, action, reward, done))
            else:
                remote.send((0, 0, 0, 0, True))

        if cmd == 'reset':
            pre_num_mineral = 20
            done = False
            obs = env.reset()          #env 초기화
            while not 331 in obs[0].observation.available_actions:   #마린을 선택하기
                actions = actAgent2Pysc2(100, obs)
                obs = env.step(actions=[actions])
            state = obs2state(obs)
            obs = (obs[0].observation.available_actions, state)
            remote.send((obs, state, 0, 0, False))

        if cmd == 'close':
            remote.close()
            break

class SubprocVecEnv:
    def __init__(self, n_proc, visualize):
        self.visualize = visualize
        self.n_proc = n_proc
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_proc)])
        self.ps = []

        for i, (work_remote,) in enumerate(zip(self.work_remotes, )):
            self.ps.append(
                Process(target=worker, args=(work_remote, self.visualize))
            )
        for p in self.ps:
            p.start()

    def step(self, actions, obs_s, global_steps):
        for remote, action, obs, global_step in zip(self.remotes, actions, obs_s, global_steps):
            remote.send(('step', action, obs, global_step))

        results = [remote.recv() for remote in self.remotes]
        return results

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', 0, 0, 0))
        
        results = [remote.recv() for remote in self.remotes]
        return results

    def close(self):
        for remote in self.remotes:
            remote.send(('close', 0, 0, 0))

        for p in self.ps:
            p.join()