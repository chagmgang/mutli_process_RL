from multiprocessing import Process, Pipe
import gym


def worker(remote, render):
    env = gym.make('CartPole-v0')
    while True:
        if render:
            env.render()
        cmd, action = remote.recv()
        if cmd == 'step':
            if not action == 'done':
                next_state, reward, done, _ = env.step(action)
                if done:
                    reward = 0
                remote.send((next_state, reward, done))
            else:
                remote.send((0, 0, True))

        if cmd == 'reset':
            state = env.reset()
            remote.send(state)

        if cmd == 'wait':
            print('wait')

        if cmd == 'close':
            remote.close()
            break

class SubprocVecEnv:
    def __init__(self, n_proc, render):
        self.render = render
        self.n_proc = n_proc
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_proc)])
        self.ps = []

        for i, (work_remote,) in enumerate(zip(self.work_remotes, )):
            self.ps.append(
                Process(target=worker, args=(work_remote, render))
            )
        for p in self.ps:
            p.start()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        results = [remote.recv() for remote in self.remotes]
        return results

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', 0))
        results = [remote.recv() for remote in self.remotes]
        return results

    def close(self):
        for remote in self.remotes:
            remote.send(('close',0))

        for p in self.ps:
            p.join()