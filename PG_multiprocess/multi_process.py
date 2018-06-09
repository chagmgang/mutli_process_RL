import os
from multiprocessing import Process, current_process, Manager
import gym
from pg import PG, discount_rewards
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')

def doubler(index, return_dict):

    proc_name = os.getpid()
    return_dict[proc_name] = index

def main():
    sess = tf.Session()
    main_pg = PG(sess, 4, 2, 0.001, 'main')
    sub_pg = PG(sess, 4, 2, 0.001, 'sub')
    sess.run(tf.global_variables_initializer())
    
    main_parameter, one = main_pg.get_parameters()
    print(one, 'main parameter')
    sub_parameter, two = sub_pg.get_parameters()
    print(two, 'before assignment')
    sub_pg.set_parameters(main_parameter)
    sub_parameter, three = sub_pg.get_parameters()
    print(three, 'after assignment')

    manager = Manager()
    return_dict = manager.dict()
    model_name = ['1', '2']
    procs = []

    for index in range(2):
        proc = Process(target=doubler, args=(index, return_dict))
        procs.append(proc)
        proc.start()
    
    for proc in procs:
        proc.join()
    print(return_dict)
    

if __name__ == '__main__':
    main()