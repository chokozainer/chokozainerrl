"""An example of training a Deep Recurrent Q-Network (DRQN).

DRQN is a DQN with a recurrent Q-network, described in
https://arxiv.org/abs/1507.06527.

To train DRQN for 50M timesteps on Breakout, run:
    python train_drqn_ale.py --recurrent

To train DQRN using a recurrent model on flickering 1-frame Breakout, run:
    python train_drqn_ale.py --recurrent --flicker --no-frame-stack
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import functools
import os
import sys

import chainer
from chainer import functions as F
from chainer import links as L
import gym
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chokozainerrl import experiments
from chokozainerrl import tools
from chainerrl import explorers
from chainerrl import misc
from chainerrl import replay_buffer

from chainerrl.wrappers import atari_wrappers


def make_args(argstr):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='check')
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-ram-v4')
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load-agent', type=str, default=None)
    parser.add_argument('--log-type',type=str,default="full_stream")
    parser.add_argument('--save-mp4',type=str,default="test.mp4")

    parser.add_argument('--steps', type=int, default=5 * 10 ** 7)
    parser.add_argument('--step-offset', type=int, default=0)
    parser.add_argument('--checkpoint-frequency', type=int,default=None)
    parser.add_argument('--max-frames', type=int,default=30 * 60 * 60)  # 30 minutes with 60 fps
    parser.add_argument('--eval-interval', type=int, default=250000)
    parser.add_argument('--eval-n-steps', type=int, default=125000)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--final-exploration-frames',type=int, default=10 ** 6)
    parser.add_argument('--final-epsilon', type=float, default=0.01)
    parser.add_argument('--eval-epsilon', type=float, default=0.001)
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-interval',type=int, default=10 ** 4)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--monitor', action='store_true', default=False)

    parser.add_argument('--recurrent', action='store_true', default=False)
    parser.add_argument('--flicker', action='store_true', default=False)
    parser.add_argument('--no-frame-stack', action='store_true', default=False)
    parser.add_argument('--episodic-update-len', type=int, default=10)

    myargs = parser.parse_args(argstr)
    return myargs

def main(args):
    import logging
    logging.basicConfig(level=logging.INFO, filename='log')

    if(type(args) is list):
        args=make_args(args)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = gym.make(args.env)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = chainerrl.wrappers.RandomizeAction(env, args.eval_epsilon)
        if args.monitor:
            env = gym.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)
    print('Observation space', env.observation_space)
    print('Action space', env.action_space)

    n_actions = env.action_space.n
    if args.recurrent:
        # Q-network with LSTM
        q_func = chainerrl.links.StatelessRecurrentSequential(
            L.Convolution2D(None, 32, 8, stride=4),
            F.relu,
            L.Convolution2D(None, 64, 4, stride=2),
            F.relu,
            L.Convolution2D(None, 64, 3, stride=1),
            functools.partial(F.reshape, shape=(-1, 3136)),
            F.relu,
            L.NStepLSTM(1, 3136, 512, 0),
            L.Linear(None, n_actions),
            DiscreteActionValue,
        )
        # Replay buffer that stores whole episodes
        rbuf = replay_buffer.EpisodicReplayBuffer(10 ** 6)
    else:
        # Q-network without LSTM
        q_func = chainer.Sequential(
            L.Convolution2D(None, 32, 8, stride=4),
            F.relu,
            L.Convolution2D(None, 64, 4, stride=2),
            F.relu,
            L.Convolution2D(None, 64, 3, stride=1),
            functools.partial(F.reshape, shape=(-1, 3136)),
            L.Linear(None, 512),
            F.relu,
            L.Linear(None, n_actions),
            DiscreteActionValue,
        )
        # Replay buffer that stores transitions separately
        rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    # Draw the computational graph and save it in the output directory.
    fake_obss = np.zeros(env.observation_space.shape, dtype=np.float32)[None]
    if args.recurrent:
        fake_out, _ = q_func(fake_obss, None)
    else:
        fake_out = q_func(fake_obss)
    chainerrl.misc.draw_computational_graph(
        [fake_out], os.path.join(args.outdir, 'model'))

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    opt = chainer.optimizers.Adam(1e-4, eps=1e-4)
    opt.setup(q_func)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = chainerrl.agents.DoubleDQN(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        batch_accumulator='mean',
        phi=phi,
        minibatch_size=args.batch_size,
        episodic_update_len=args.episodic_update_len,
        recurrent=args.recurrent,
    )

    if args.load_agent:
        agent.load(args.load_agent)

    if (args.mode=='train'):
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            checkpoint_freq=args.checkpoint_frequency,
            step_offset=args.step_offset,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            eval_env=eval_env,
            log_type=args.log_type
        )
    elif (args.mode=='check'):
        return tools.make_video.check(env=env,agent=agent,save_mp4=args.save_mp4)

    elif (args.mode=='growth'):
        return tools.make_video.growth(env=env,agent=agent,outdir=args.outdir,max_num=args.max_frames,save_mp4=args.save_mp4)
