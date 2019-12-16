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

import chainer
from chainer import functions as F
from chainer import links as L
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

#from chokozainerrl import wrappers
#from chainerrl.wrappers import atari_wrappers
from chokozainerrl.wrappers import atari_wrappers


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
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--prioritized-replay', action='store_true')
    parser.add_argument('--replay-start-size', type=int, default=1000)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 2)
    parser.add_argument('--target-update-method', type=str, default='hard')
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=10 ** 4)
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
        env = atari_wrappers.FireResetEnvAuto(env)
        print("Set FireResetEnvAuto")
        if isinstance(env.action_space, spaces.Box):
            misc.env_modifiers.make_action_filtered(env, clip_action_filter)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        print(env.action_space)
        return env

    env = make_env(test=False)

    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space
    n_actions = action_space.n
    
    q_func = chainerrl.links.Sequence(
            chainerrl.links.NatureDQNHead(),
            L.Linear(512, n_actions),
            chainerrl.action_value.DiscreteActionValue)

    # Use the same hyper parameters as the Nature paper's
    optimizer = chainer.optimizers.RMSpropGraves(lr=2.5e-4, alpha=0.95, momentum=0.0, eps=1e-2)
    optimizer.setup(q_func)

    rbuf = chainerrl.replay_buffer.ReplayBuffer(10 ** 6)

    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        1.0, 0.1,
        10 ** 6,
        lambda: np.random.randint(n_actions))


    def dqn_phi(screens):
        assert len(screens) == 4
        assert screens[0].dtype == np.uint8
        raw_values = np.asarray(screens, dtype=np.float32)
        # [0,255] -> [0, 1]
        raw_values /= 255.0
        return raw_values

    agent = chainerrl.agents.DQN(q_func, optimizer, rbuf, gpu=0, gamma=0.99,
                explorer=explorer, replay_start_size=5 * 10 ** 4,
                target_update_interval=10 ** 4,
                clip_delta=True,
                update_interval=4,
                batch_accumulator='sum', phi=dqn_phi)

    if args.load_agent:
        agent.load(args.load_agent)

    eval_env = make_env(test=True)

    if (args.mode=='train'):
        def step_hook(env, agent, step):
            sys.stdout.write("\r{} / {} steps.".format(step, STEPS))
            sys.stdout.flush()

        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_env=eval_env,
            step_offset=args.step_offset,
            checkpoint_freq=args.checkpoint_freq,
            train_max_episode_len=args.max_frames,
            eval_max_episode_len=args.max_frames,
            log_type=args.log_type, step_hooks=[step_hook]
            )
    elif (args.mode=='check'):
        return tools.make_video.check(env=env,agent=agent,max_num=args.max_frames,save_mp4=args.save_mp4)

    elif (args.mode=='growth'):
        return tools.make_video.growth(env=env,agent=agent,outdir=args.outdir,max_num=args.max_frames,save_mp4=args.save_mp4)
