import os
from multiprocessing import Process, current_process, Manager
import gym
from pg import PG, discount_rewards
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')

def doubler(main_parameter, name, return_dict):
    sess = tf.Session()
    sub_pg = PG(sess, 4, 2, 0.001, name)
    sess.run(tf.global_variables_initializer())
    sub_parameter, two = sub_pg.get_parameters()
    print(two, name, 'before assignment')

    sub_pg.set_parameters(main_parameter)
    sub_parameter, two = sub_pg.get_parameters()
    print(two, name, 'after assignment')

    proc_name = os.getpid()
    return_dict[proc_name] = 1

def main():
    sess = tf.Session()
    main_pg = PG(sess, 4, 2, 0.001, 'main')
    sess.run(tf.global_variables_initializer())
    
    main_parameter, one = main_pg.get_parameters()
    print(one, 'main parameter')

    manager = Manager()
    return_dict = manager.dict()
    model_name = ['1', '2']
    procs = []

    for index in range(2):
        proc = Process(target=doubler, args=(main_parameter, model_name[index], return_dict))
        procs.append(proc)
        proc.start()
    
    for proc in procs:
        proc.join()
    

if __name__ == '__main__':
    main()
