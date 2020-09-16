from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network, q_rnn_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import random
from collections import deque
import gym
import pandas as pd
import os
import time
import pickle
import sys
from argparse import ArgumentParser

from enviroment.order_side import OrderSide
from enviroment.agent_utils.ui import UI
import gym_enviroment
from enviroment.feature_type import FeatureType


parser = ArgumentParser()
parser.add_argument("-s", help="side name", dest="side_name", default="buy")
parser.add_argument("-v", help="version", dest="version", default="paper")
parser.add_argument("-d", help="data trend", dest="trend", default="dn")
parser.add_argument("-f", help="feature type", dest="feature_type", default="order")
parser.add_argument("-m", help="model type", dest="model_type", default="MLP")
parser.add_argument("-r", help="reward type", dest="reward_type", default='profit')
parser.add_argument("--tui", help="target update iter", dest="target_update_iter", default=400)
parser.add_argument("--bf", help="buffer size", dest="buffer_size", default=100000)
parser.add_argument("--it", help="numbers of iteration", dest="iterations", default=500000)
parser.add_argument("-gn", help="gpu number", dest="gpu_number", default=1)
parser.add_argument("-b", help="beta for sensitive q learning", dest="beta", default=0.0, type=float)
parser.add_argument("-l", help="lamda for sensitive q learning", dest="lamda", default=1.0, type=float)
parser.add_argument("--ev", help="evaluation episodes", dest="evps", default=100, type=int)
parser.add_argument("-p", help="preference", dest="preference", default="[1.0,0.0]")
parser.add_argument("-fv", help="file version", dest="file_version", default="v2")

args = parser.parse_args()
side_name = args.side_name
version = args.version
trend = args.trend
feature_type = args.feature_type
model_type = args.model_type
preference = eval(args.preference)

# model type, buffer size, iterations, target_update_iter
if args.beta != 0.0:
    logs_dir = './TFA_Sensitive_DDQN_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.file_version, args.beta, model_type, args.buffer_size, args.iterations, args.target_update_iter, args.evps, preference, args.reward_type)
elif args.lamda != 1.0:
    logs_dir = './TFA_Averse_DDQN_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.file_version, args.lamda, model_type, args.buffer_size, args.iterations, args.target_update_iter, args.evps, preference, args.reward_type)
else:
    logs_dir = './TFA_DDQN_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.file_version, model_type, args.buffer_size, args.iterations, args.target_update_iter, args.evps, preference, args.reward_type)

print('logs_dir',logs_dir)


if not os.path.isdir(logs_dir):
    os.mkdir(logs_dir)
    
if version == 'paper':
    from enviroment.orderbook import Orderbook
else:
    from enviroment.orderbook_new import Orderbook

def load_customized_gym(env_name, orderbook, side, featureType, reward_type, preference=[1.0,0.0]):
    max_episode_steps = None
    gym_kwargs = None
    
    gym_kwargs = gym_kwargs if gym_kwargs else {}
    gym_spec = gym.spec(env_name)
    gym_env = gym_spec.make(**gym_kwargs)
    gym_env.configure(orderbook, side=side, featureType=featureType, reward_type=reward_type, preference=preference)
    
    if max_episode_steps is None and gym_spec.max_episode_steps is not None:
        max_episode_steps = gym_spec.max_episode_steps
    #print(max_episode_steps)
    env = suite_gym.wrap_env(
        gym_env,
        discount=0.99,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=(),
        env_wrappers=(),
        spec_dtype_map=None)
    return env

def compute_avg_return(environment, policy, num_episodes=10, policy_state=()):

    total_return = 0.0
    return_list = []
    volatility_list = []
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        volatility_list.append(np.array(environment.get_info()[0]))
        
        while not time_step.is_last():
            action_step = policy.action(time_step,policy_state)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
        return_list.append(episode_return)
    return_list = np.array(return_list)
    volatility_list = np.array(volatility_list)
    avg_return = total_return / num_episodes
    
    # return profit_vol, variance, profit
    if args.reward_type == 'profit_vol':
        profit_array = return_list*volatility_list
        return avg_return.numpy()[0], np.var(return_list), np.mean(profit_array), np.var(profit_array)
    elif args.reward_type == 'profit':
        profit_vol_list = return_list/volatility_list
        return np.mean(profit_vol_list), np.var(profit_vol_list), avg_return.numpy()[0], np.var(return_list)


def collect_step(environment, policy, buffer, policy_state=()):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step,policy_state)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)
        
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the spcific GPU
        try:
            pass
            tf.config.experimental.set_visible_devices(gpus[int(args.gpu_number)], 'GPU') # specify gpu
            tf.config.experimental.set_memory_growth(gpus[int(args.gpu_number)], True) # set growth of memory
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
            
    # Load orderbook
    if trend == 'dn':
        orderbook = Orderbook()
        orderbook.loadFromEvents('data/events/ob-train.tsv')
        orderbook_test = Orderbook()
        orderbook_test.loadFromEvents('data/events/ob_dnTrend_test.tsv')
    else:
        orderbook = Orderbook()
        orderbook.loadFromEvents('data/events/ob_upTrend_train.tsv')
        orderbook_test = Orderbook()
        orderbook_test.loadFromEvents('data/events/ob_upTrend_test.tsv')
    
    if side_name == 'buy':
        side = OrderSide.BUY
    else:
        side = OrderSide.SELL
    
    if feature_type == 'order':
        featureType = FeatureType.ORDERS 
    else:
        featureType=FeatureType.TRADES
    
    # hyperparameters
    num_iterations = int(args.iterations) # @param {type:"integer"}

    initial_collect_steps = 1000  # @param {type:"integer"} 
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = int(args.buffer_size) # @param {type:"integer"}
    target_update_period = int(args.target_update_iter) # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}
    num_eval_episodes = args.evps  # @param {type:"integer"}
    
    # Prepare the enviroment
    train_py_env = load_customized_gym("enviroment-v0",orderbook, side, featureType, args.reward_type, preference=preference)
    eval_py_env = load_customized_gym("enviroment-v0",orderbook, side, featureType, args.reward_type, preference=preference)
    test_py_env = load_customized_gym("enviroment-v0",orderbook_test, side, featureType, args.reward_type, preference=preference)
    
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    test_env = tf_py_environment.TFPyEnvironment(test_py_env)
    
    # build model
    if 'RNN' not in args.model_type:
        
        if args.model_type == 'CNN':
            if feature_type == 'order':
                conv_layer_params = ((16,(1,2),(1,2)),(16,(4,1),1),(16,(4,1),1),(16,(1,2),(1,2)),(16,(4,1),1),(16,(4,1),1),)
            else:
                conv_layer_params = ((16,(1,4),(1,4)),(16,(1,4),(1,4)),)
        else:
            conv_layer_params = None
        fc_layer_params = (100,100,)
        q_net = q_network.QNetwork(
            train_env.observation_spec(), # input
            train_env.action_spec(), # output
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params)

        q_target_net = q_network.QNetwork(
            train_env.observation_spec(), # input
            train_env.action_spec(), # output
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params)
    else:
        
        if args.model_type == 'CNNRNN':
            conv_layer_params = ((16,(1,2),(1,2)),(16,(4,1),1),(16,(4,1),1),(16,(1,2),(1,2)),(16,(4,1),1),(16,(4,1),1),)
        else:
            conv_layer_params = None
        lstm_size=(200,) # rank of this means the depth of the LSTM model
        fc_layer_params = (lstm_size[0],) #for fully connected

        q_net = q_rnn_network.QRnnNetwork(
            train_env.observation_spec(), # input
            train_env.action_spec(), # output
            input_fc_layer_params=None,
            conv_layer_params=conv_layer_params,
            lstm_size=lstm_size,
            output_fc_layer_params=fc_layer_params)

        q_target_net = q_rnn_network.QRnnNetwork(
            train_env.observation_spec(), # input
            train_env.action_spec(), # output
            input_fc_layer_params=None,
            conv_layer_params=conv_layer_params,
            lstm_size=lstm_size,
            output_fc_layer_params=fc_layer_params)
        
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)
    
    # bulid agents
    agent = dqn_agent.DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        target_q_network=q_target_net,
        target_update_period = target_update_period,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        beta=args.beta,
        lamda=args.lamda)


    agent.initialize()
   
    # bulid policy
    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())
    # bulid replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)
    
    # collect some data first
    collect_data(train_env, random_policy, replay_buffer, steps=initial_collect_steps)
   
    # bulid dataset
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=batch_size, 
        num_steps=2).prefetch(3)
    iterator = iter(dataset)
    
    # start training
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)
   
    # Evaluate the agent's policy once before training.
    avg_profit_vol, var_profit_vol, avg_profit, var_profit = compute_avg_return(eval_env, agent.policy, num_eval_episodes, policy_state=())
    var_profit_vol_list = [var_profit_vol]
    avg_profit_vol_list = [avg_profit_vol]
    avg_profit_list = [avg_profit]
    var_profit_list = [var_profit]
    
    test_avg_profit_vol, test_var_profit_vol, test_avg_profit, test_var_profit = compute_avg_return(test_env, agent.policy, num_eval_episodes, policy_state=())
    test_var_profit_vol_list = [test_var_profit_vol]
    test_avg_profit_vol_list = [test_avg_profit_vol]
    test_avg_profit_list = [test_avg_profit]
    test_var_profit_list = [test_var_profit]
    
    losses = []
    
    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer, policy_state=())

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss
            losses.append(train_loss)
            step = agent.train_step_counter.numpy()

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))
    
                if step % eval_interval == 0:
                    avg_profit_vol, var_profit_vol, avg_profit, var_profit = compute_avg_return(eval_env, agent.policy, num_eval_episodes, policy_state=())
                    print('step = {0}: Average profit_vol = {1} Var of profit_vol = {2} Average profit={3}'.format(step, avg_profit_vol, var_profit_vol, avg_profit))
                    avg_profit_vol_list.append(avg_profit_vol)
                    var_profit_vol_list.append(var_profit_vol)
                    avg_profit_list.append(avg_profit)
                    var_profit_list.append(var_profit)
                    
                if step % eval_interval == 0:
                    test_avg_profit_vol, test_var_profit_vol, test_avg_profit, test_var_profit = compute_avg_return(test_env, agent.policy, num_eval_episodes, policy_state=())
                    print('step = {0}: Average test_profit_vol = {1} Var of test_profit_vol = {2} Average test_profit={3}'.format(step, test_avg_profit_vol, test_var_profit_vol, test_avg_profit))
                    test_avg_profit_vol_list.append(test_avg_profit_vol)
                    test_var_profit_vol_list.append(test_var_profit_vol)
                    test_avg_profit_list.append(test_avg_profit)
                    test_var_profit_list.append(test_var_profit)
                    
    
    with open('./{}/Profit_vol_{}_{}_{}_{}.pkl'.format(logs_dir,side_name,version, trend, feature_type),'wb') as f:
        pickle.dump(avg_profit_vol_list,f)
    with open('./{}/Var_PV_{}_{}_{}_{}.pkl'.format(logs_dir,side_name,version, trend, feature_type),'wb') as f:
        pickle.dump(var_profit_vol_list,f)
    with open('./{}/Profit_{}_{}_{}_{}.pkl'.format(logs_dir,side_name,version, trend, feature_type),'wb') as f:
        pickle.dump(avg_profit_list,f)
    with open('./{}/Var_Profit_{}_{}_{}_{}.pkl'.format(logs_dir,side_name,version, trend, feature_type),'wb') as f:
        pickle.dump(var_profit_list,f)
    with open('./{}/Loss_{}_{}_{}_{}.pkl'.format(logs_dir,side_name,version, trend, feature_type),'wb') as f:
        pickle.dump(losses,f)
        
   
    with open('./{}/Test_Profit_vol_{}_{}_{}_{}.pkl'.format(logs_dir,side_name,version, trend, feature_type),'wb') as f:
        pickle.dump(test_avg_profit_vol_list,f)
    with open('./{}/Test_Var_PV_{}_{}_{}_{}.pkl'.format(logs_dir,side_name,version, trend, feature_type),'wb') as f:
        pickle.dump(test_var_profit_vol_list,f)
    with open('./{}/Test_Profit_{}_{}_{}_{}.pkl'.format(logs_dir,side_name,version, trend, feature_type),'wb') as f:
        pickle.dump(test_avg_profit_list,f)
    with open('./{}/Test_Var_Profit_{}_{}_{}_{}.pkl'.format(logs_dir,side_name,version, trend, feature_type),'wb') as f:
        pickle.dump(test_var_profit_list,f)
