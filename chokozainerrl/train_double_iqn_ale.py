"""An example of training DQN against OpenAI Gym Atari Envs.
This script is changed from chainerrl examples. 
This script is an example of training a DQN agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported. For continuous action
spaces, A NAF (Normalized Advantage Function) is used to approximate Q-values.

Caution: Double IQN work only new chainerRL > 7.0?

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
import json
import os
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import gym
import numpy as np

import chainerrl
from chokozainerrl import experiments
from chokozainerrl import tools
from chainerrl import explorers
from chainerrl import misc
from chainerrl import replay_buffer
from chainerrl.wrappers import atari_wrappers


def parse_agent(agent):
    return {'IQN': chainerrl.agents.IQN,
            'DoubleIQN': chainerrl.agents.DoubleIQN}[agent]

def make_args(argstr):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='check')
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load-agent', type=str, default=None)
    parser.add_argument('--log-type',type=str,default="full_stream")
    parser.add_argument('--save-mp4',type=str,default="test.mp4")
    parser.add_argument('--agent', type=str, default='IQN',choices=['IQN', 'DoubleIQN'])

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
    parser.add_argument('--prioritized', action='store_true', default=False)
    parser.add_argument('--num-step-return', type=int, default=1)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-accumulator', type=str, default='mean',choices=['mean', 'sum'])
    parser.add_argument('--quantile-thresholds-N', type=int, default=64)
    parser.add_argument('--quantile-thresholds-N-prime', type=int, default=64)
    parser.add_argument('--quantile-thresholds-K', type=int, default=32)
    parser.add_argument('--n-best-episodes', type=int, default=200)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--monitor', action='store_true', default=False)

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
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test)
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
    n_actions = env.action_space.n

    q_func = chainerrl.agents.iqn.ImplicitQuantileQFunction(
        psi=chainerrl.links.Sequence(
            L.Convolution2D(None, 32, 8, stride=4),
            F.relu,
            L.Convolution2D(None, 64, 4, stride=2),
            F.relu,
            L.Convolution2D(None, 64, 3, stride=1),
            F.relu,
            functools.partial(F.reshape, shape=(-1, 3136)),
        ),
        phi=chainerrl.links.Sequence(
            chainerrl.agents.iqn.CosineBasisLinear(64, 3136),
            F.relu,
        ),
        f=chainerrl.links.Sequence(
            L.Linear(None, 512),
            F.relu,
            L.Linear(None, n_actions),
        ),
    )

    # Draw the computational graph and save it in the output directory.
    fake_obss = np.zeros((4, 84, 84), dtype=np.float32)[None]
    fake_taus = np.zeros(32, dtype=np.float32)[None]
    chainerrl.misc.draw_computational_graph(
        [q_func(fake_obss)(fake_taus)],
        os.path.join(args.outdir, 'model'))

    # Use the same hyper parameters as https://arxiv.org/abs/1710.10044
    opt = chainer.optimizers.Adam(5e-5, eps=1e-2 / args.batch_size)
    opt.setup(q_func)

    if args.prioritized:
        betasteps = args.steps / args.update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            10 ** 6, alpha=0.5, beta0=0.4, betasteps=betasteps,
            num_steps=args.num_step_return)
    else:
        rbuf = replay_buffer.ReplayBuffer(
            10 ** 6,
            num_steps=args.num_step_return)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    Agent = parse_agent(args.agent)
    agent = Agent(
        q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
        explorer=explorer, replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        batch_accumulator=args.batch_accumulator,
        phi=phi,
        quantile_thresholds_N=args.quantile_thresholds_N,
        quantile_thresholds_N_prime=args.quantile_thresholds_N_prime,
        quantile_thresholds_K=args.quantile_thresholds_K,
    )

    if args.load_agent:
        agent.load(args.load_agent)

    if (args.mode=='train'):
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            checkpoint_freq=args.checkpoint_frequency,
            step_offset=args.step_offset,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
            log_type=args.log_type
        )

        dir_of_best_network = os.path.join(args.outdir, "best")
        agent.load(dir_of_best_network)

        # run 200 evaluation episodes, each capped at 30 mins of play
        stats = experiments.evaluator.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.n_best_episodes,
            max_episode_len=args.max_frames / 4,
            logger=None)
        with open(os.path.join(args.outdir, 'bestscores.json'), 'w') as f:
            # temporary hack to handle python 2/3 support issues.
            # json dumps does not support non-string literal dict keys
            json_stats = json.dumps(stats)
            print(str(json_stats), file=f)
        print("The results of the best scoring network:")
        for stat in stats:
            print(str(stat) + ":" + str(stats[stat]))
    elif (args.mode=='check'):
        return tools.make_video.check(env=env,agent=agent,save_mp4=args.save_mp4)

    elif (args.mode=='growth'):
        return tools.make_video.growth(env=env,agent=agent,outdir=args.outdir,max_num=args.max_frames,save_mp4=args.save_mp4)

