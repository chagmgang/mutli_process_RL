import sys
from absl import flags
import tensorflow as tf
import threading
from pg import PG
import numpy as np
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib.pyplot as plt

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
def MyLoop(coord, index, pg, results):
    while not coord.should_stop():
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)
        with sc2_env.SC2Env(map_name="MoveToBeacon", step_mul=4, visualize=False,
                                    screen_size_px=(16, 16), minimap_size_px=(16, 16)) as env:
            obs = env.reset()
            state = (obs[0].observation['screen'][_PLAYER_RELATIVE])
            action = pg.choose_action([[1,1,1,1]])
            results[index] = state
            coord.request_stop()

def train():
        sess = tf.Session()
        coord = tf.train.Coordinator()
        pg = PG(sess, 4 , 2, 0.001)
        sess.run(tf.global_variables_initializer())
        threads = []
        results = [None] * 2
        indexs = np.arange(2)
        print(len(indexs))
        for index in indexs:
            threads.append(threading.Thread(target=MyLoop,
                            args=(coord, index, pg, results)))

        for t in threads:
            t.start()

        coord.join(threads)
        print(results)
        plt.imshow(results[1])
        plt.show()
        plt.imshow(results[0])
        plt.show()


if __name__ == '__main__':
    train()