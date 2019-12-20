"""An example of training PPO against OpenAI Gym Atari Envs.

This script is an example of training a PPO agent on Atari envs.

To train PPO for 10M timesteps on Breakout, run:
    python train_ppo_ale.py

To train PPO using a recurrent model on a flickering Atari env, run:
    python train_ppo_ale.py --recurrent --flicker --no-frame-stack
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

import chainer
from chainer import functions as F
from chainer import links as L
import numpy as np

import chainerrl
from chainerrl.agents import PPO
from chokozainerrl import experiments
from chokozainerrl import tools
from chainerrl import misc
from chainerrl.wrappers import atari_wrappers


def make_args(argstr):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='check')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load-agent', type=str, default=None)
    parser.add_argument('--log-type',type=str,default="full_stream")
    parser.add_argument('--save-mp4',type=str,default="test.mp4")

    parser.add_argument('--steps', type=int, default=5 * 10 ** 7)
    parser.add_argument('--step-offset', type=int, default=0)
    parser.add_argument('--checkpoint-frequency', type=int,default=None)
    parser.add_argument('--max-frames', type=int,default=30 * 60 * 60)  # 30 minutes with 60 fps
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)


    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--update-interval', type=int, default=128 * 8)
    parser.add_argument('--batchsize', type=int, default=32 * 8)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--log-interval', type=int, default=10000)
    parser.add_argument('--recurrent', action='store_true', default=False)
    parser.add_argument('--flicker', action='store_true', default=False)
    parser.add_argument('--no-frame-stack', action='store_true', default=False)
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

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def make_env(idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test,
            flicker=args.flicker,
            frame_stack=not args.no_frame_stack,
        )
        env.seed(env_seed)
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
            [(lambda: make_env(idx, test))
             for idx, env in enumerate(range(args.num_envs))])

    sample_env = make_env(0, test=False)
    print('Observation space', sample_env.observation_space)
    print('Action space', sample_env.action_space)
    n_actions = sample_env.action_space.n

    winit_last = chainer.initializers.LeCunNormal(1e-2)
    if args.recurrent:
        model = chainerrl.links.StatelessRecurrentSequential(
            L.Convolution2D(None, 32, 8, stride=4),
            F.relu,
            L.Convolution2D(None, 64, 4, stride=2),
            F.relu,
            L.Convolution2D(None, 64, 3, stride=1),
            F.relu,
            L.Linear(None, 512),
            F.relu,
            L.NStepGRU(1, 512, 512, 0),
            chainerrl.links.Branched(
                chainer.Sequential(
                    L.Linear(None, n_actions, initialW=winit_last),
                    chainerrl.distribution.SoftmaxDistribution,
                ),
                L.Linear(None, 1),
            )
        )
    else:
        model = chainer.Sequential(
            L.Convolution2D(None, 32, 8, stride=4),
            F.relu,
            L.Convolution2D(None, 64, 4, stride=2),
            F.relu,
            L.Convolution2D(None, 64, 3, stride=1),
            F.relu,
            L.Linear(None, 512),
            F.relu,
            chainerrl.links.Branched(
                chainer.Sequential(
                    L.Linear(None, n_actions, initialW=winit_last),
                    chainerrl.distribution.SoftmaxDistribution,
                ),
                L.Linear(None, 1),
            )
        )

    # Draw the computational graph and save it in the output directory.
    fake_obss = np.zeros(
        sample_env.observation_space.shape, dtype=np.float32)[None]
    if args.recurrent:
        fake_out, _ = model(fake_obss, None)
    else:
        fake_out = model(fake_obss)
    chainerrl.misc.draw_computational_graph(
        [fake_out], os.path.join(args.outdir, 'model'))

    opt = chainer.optimizers.Adam(alpha=args.lr, eps=1e-5)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(0.5))

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = PPO(
        model,
        opt,
        gpu=args.gpu,
        phi=phi,
        update_interval=args.update_interval,
        minibatch_size=args.batchsize,
        epochs=args.epochs,
        clip_eps=0.1,
        clip_eps_vf=None,
        standardize_advantages=True,
        entropy_coef=1e-2,
        recurrent=args.recurrent,
    )

    if args.load_agent:
        agent.load(args.load_agent)

    if (args.mode=='train'):
        step_hooks = []
        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            agent.optimizer.alpha = value

        step_hooks.append(
            experiments.LinearInterpolationHook(
                args.steps, args.lr, 0, lr_setter))

        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            step_offset=args.step_offset,
            checkpoint_freq=args.checkpoint_frequency,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            save_best_so_far_agent=False,
            step_hooks=step_hooks,
            log_type=args.log_type
        )
    elif (args.mode=='check'):
        return tools.make_video.check(env=make_env_check(),agent=agent,save_mp4=args.save_mp4)

    elif (args.mode=='growth'):
        return tools.make_video.growth(env=make_env_check(),agent=agent,outdir=args.outdir,max_num=args.max_frames,save_mp4=args.save_mp4)
