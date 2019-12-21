from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import argparse
import functools
import logging
import os

import chainer
import numpy as np

import chainerrl
from chainerrl.agents import a2c
from chokozainerrl import experiments
from chokozainerrl import tools
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policy
from chainerrl import v_function

from chainerrl.wrappers import atari_wrappers


def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32) / 255


class A2CFF(chainer.ChainList, a2c.A2CModel):

    def __init__(self, n_actions):
        self.action_space = 1

        self.head = links.NIPSDQNHead()
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        super().__init__(self.head, self.pi, self.v)

    def pi_and_v(self, state):
        out = self.head(state)
        return self.pi(out), self.v(out)

def make_args(argstr):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='check')
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--load-agent', type=str, default=None)
    parser.add_argument('--log-type',type=str,default="full_stream")
    parser.add_argument('--save-mp4',type=str,default="test.mp4")
    parser.add_argument('--num-envs', type=int, default=1)
    
    parser.add_argument('--steps', type=int, default=5 * 10 ** 7)
    parser.add_argument('--step-offset', type=int, default=0)
    parser.add_argument('--checkpoint-frequency', type=int,default=None)
    parser.add_argument('--max-frames', type=int,default=30 * 60 * 60)  # 30 minutes with 60 fps
    parser.add_argument('--update-steps', type=int, default=5)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)


    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-5)
    parser.add_argument('--use-gae', action='store_true', default=False)
    parser.add_argument('--tau', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=40)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--monitor', action='store_true', default=False)
    parser.set_defaults(use_lstm=False)

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
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    misc.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 31

    def make_env(process_idx, test):
        # Use different random seeds for train and test envs
        process_seed = process_seeds[process_idx]
        env_seed = 2 ** 31 - 1 - process_seed if test else process_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test)
        env.seed(int(env_seed))
        if args.monitor:
            env = chainerrl.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env
    def make_env_check():
        # Use different random seeds for train and test envs
        env_seed = args.seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=True,
            clip_rewards=True)
        env.seed(int(env_seed))
        return env

    def make_batch_env(test):
        return chainerrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_env, idx, test)
             for idx, env in enumerate(range(args.num_envs))])

    sample_env = make_env(0, test=False)

    n_actions = sample_env.action_space.n

    model = A2CFF(n_actions)
    optimizer = rmsprop_async.RMSpropAsync(lr=args.lr,
                                           eps=args.rmsprop_epsilon,
                                           alpha=args.alpha)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.max_grad_norm))
    if args.weight_decay > 0:
        optimizer.add_hook(NonbiasWeightDecay(args.weight_decay))
    agent = a2c.A2C(
        model, optimizer, gamma=args.gamma,
        gpu=args.gpu,
        num_processes=args.num_envs,
        update_steps=args.update_steps,
        phi=phi,
        use_gae=args.use_gae,
        tau=args.tau,
    )

    if args.load_agent:
        agent.load(args.load_agent)

    if (args.mode=='train'):
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(test=False),
            eval_env=make_batch_env(test=True),
            steps=args.steps,
            step_offset=args.step_offset,
            checkpoint_freq=args.checkpoint_frequency,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=False,
            log_interval=1000,
            log_type=args.log_type
        )
    elif (args.mode=='check'):
        return tools.make_video.check(env=make_env_check(),agent=agent,save_mp4=args.save_mp4)

    elif (args.mode=='growth'):
        return tools.make_video.growth(env=make_env_check(),agent=agent,outdir=args.outdir,max_num=args.max_frames,save_mp4=args.save_mp4)
