import tensorflow as tf
import numpy as np
import gym

class PG:
    def __init__(self, sess, state_size, action_size, exp_rate):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.exp_rate = exp_rate

        self.X = tf.placeholder(tf.float32, [None, self.state_size])
        self.advantages = tf.placeholder(tf.float32, [None, 1])
        self.Y = tf.placeholder(tf.float32, [None, self.action_size])
        self.r = tf.placeholder(tf.float32, [None, 1])
        
        self.network = self._build_network()
        self.atrain = self.train()

    def _build_network(self):
        net = tf.layers.dense(self.X, 24, activation=tf.nn.relu)
        net = tf.layers.dense(net ,24, activation=tf.nn.relu)
        action_prob = tf.layers.dense(net, self.action_size, activation=tf.nn.softmax)
        return action_prob

    def train(self):
        log_lik = self.Y*tf.log(self.network)
        log_lik_adv = log_lik * self.advantages
        obj_func = tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis = 1))
        entropy = -tf.reduce_sum(self.network * tf.log(self.network))
        loss = -(obj_func + self.exp_rate*entropy)
        train = tf.train.AdamOptimizer(0.01).minimize(loss)
        return train

    def choose_action(self, s):
        act_prob = self.sess.run(self.network, feed_dict={self.X: s})
        action = np.random.choice(self.action_size, p=act_prob[0])
        return action

def discount_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for i in reversed(range(len(r))):
        running_add = running_add * 0.99 + r[i]
        discounted_r[i] = running_add
    discounted_r = (discounted_r - discounted_r.mean())/(discounted_r.std() + 1e-7)
    return discounted_r
