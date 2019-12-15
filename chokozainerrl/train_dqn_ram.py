"""An example of training DQN against OpenAI Gym Envs.
This script is changed from chainerrl examples. 
This script is an example of training a DQN agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported. For continuous action
spaces, A NAF (Normalized Advantage Function) is used to approximate Q-values.

To solve CartPole-v0, run:
    python train_dqn_gym.py --env CartPole-v0

To solve Pendulum-v0, run:
    python train_dqn_gym.py --env Pendulum-v0
"""


from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import argparse
import os
import sys

from chainer import optimizers
import gym
from gym import spaces
import numpy as np

import chainerrl
from chokozainerrl.agents.dqn import chokoDQN
from chokozainerrl import experiments
from chokozainerrl import tools
from chainerrl import links
from chainerrl import misc
from chainerrl import q_functions
from chainerrl import replay_buffer

from chainerrl.wrappers import atari_wrappers


def make_args(argstr):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='check')
    parser.add_argument('--outdir', type=str, default='result')
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--load-agent', type=str, default=None)
    parser.add_argument('--steps', type=int, default=10 ** 5)
    parser.add_argument('--step-offset', type=int, default=0)
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--max-frames', type=int, default=30 * 60 * 60)
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 4)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--prioritized-replay', action='store_true')
    parser.add_argument('--replay-start-size', type=int, default=1000)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 2)
    parser.add_argument('--target-update-method', type=str, default='hard')
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=10 ** 4)
    parser.add_argument('--n-hidden-channels', type=int, default=100)
    parser.add_argument('--n-hidden-layers', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--minibatch-size', type=int, default=None)
    parser.add_argument('--reward-scale-factor', type=float, default=1.0)
    parser.add_argument('--log-type',type=str,default="full_stream")
    parser.add_argument('--save-mp4',type=str,default="test.mp4")
    myargs = parser.parse_args(argstr)
    return myargs

def main(args):
    import logging
    logging.basicConfig(level=logging.INFO, filename='log')

    if(type(args) is list):
        args=make_args(args)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    def clip_action_filter(a):
        return np.clip(a, action_space.low, action_space.high)

    def make_env(test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        # atari
        #env = atari_wrappers.wrap_deepmind(env,episode_life=not test,clip_rewards=not test)
        env = atari_wrappers.FireResetEnv(env)
        env = atari_wrappers.EpisodicLifeEnv(env)
        if isinstance(env.action_space, spaces.Box):
            misc.env_modifiers.make_action_filtered(env, clip_action_filter)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        print(env.action_space)
        return env

    env = make_env(test=False)

    agent = chokoDQN(env, args=args)

    # Draw the computational graph and save it in the output directory.
    #chainerrl.misc.draw_computational_graph(
    #    [agent.q_func(np.zeros_like(agent.obs_space.low, dtype=np.float32)[None])],
    #    os.path.join(args.outdir, 'model'))

    if args.load_agent:
        agent.load(args.load_agent)

    eval_env = make_env(test=True)

    if (args.mode=='train'):
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_env=eval_env,
            step_offset=args.step_offset,
            checkpoint_freq=args.checkpoint_freq,
            train_max_episode_len=args.max_frames,
            eval_max_episode_len=args.max_frames,
            log_type=args.log_type
            )
    elif (args.mode=='check'):
        return tools.make_video.check(env=env,agent=agent,save_mp4=args.save_mp4)

    elif (args.mode=='growth'):
        return tools.make_video.growth(env=env,agent=agent,outdir=args.outdir,max_num=args.max_frames,save_mp4=args.save_mp4)
