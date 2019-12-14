"""An example of training a REINFORCE agent against OpenAI Gym envs.
This script is changed from chainerrl examples. 
This script is an example of training a REINFORCE agent against OpenAI Gym
envs. Both discrete and continuous action spaces are supported.

To solve CartPole-v0, run:
    python train_reinforce_gym.py

To solve InvertedPendulum-v1, run:
    python train_reinforce_gym.py --env InvertedPendulum-v1
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os
import sys

import chainer
import gym
import gym.spaces
import numpy as np

import chainerrl
from chokozainerrl import experiments
from chainerrl import misc

def make_args(argstr):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='check')
    parser.add_argument('--outdir', type=str, default='result',
                        help='Directory path to save output files.'
                            ' If it does not exist, it will be created.')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--load-agent', type=str, default=None)
    parser.add_argument('--steps', type=int, default=10 ** 5)
    parser.add_argument('--step-offset', type=int, default=0)
    parser.add_argument('--checkpoint-freq', type=int, default=10000)    
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--beta', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--eval-interval', type=int, default=10 ** 4)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--monitor', action='store_true')
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

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    def make_env(test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = chainerrl.wrappers.Monitor(env, args.outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if args.render and not test:
            env = chainerrl.wrappers.Render(env)
        return env

    train_env = make_env(test=False)
    timestep_limit = train_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = train_env.observation_space
    action_space = train_env.action_space

    # Switch policy types accordingly to action space types
    if isinstance(action_space, gym.spaces.Box):
        model = chainerrl.policies.FCGaussianPolicyWithFixedCovariance(
            obs_space.low.size,
            action_space.low.size,
            var=0.1,
            n_hidden_channels=200,
            n_hidden_layers=2,
            nonlinearity=chainer.functions.leaky_relu,
        )
    else:
        model = chainerrl.policies.FCSoftmaxPolicy(
            obs_space.low.size,
            action_space.n,
            n_hidden_channels=200,
            n_hidden_layers=2,
            nonlinearity=chainer.functions.leaky_relu,
        )

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [model(np.zeros_like(obs_space.low, dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    opt = chainer.optimizers.Adam(alpha=args.lr)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(1))

    agent = chainerrl.agents.REINFORCE(
        model, opt, beta=args.beta, batchsize=args.batchsize)

    eval_env = make_env(test=True)

    if args.load_agent:
        agent.load(args.load_agent)

    if (args.mode=='train'):
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=train_env,
            eval_env=eval_env,
            outdir=args.outdir,
            steps=args.steps,
            step_offset=args.step_offset,
            checkpoint_freq=args.checkpoint_freq,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            train_max_episode_len=timestep_limit,
            log_type=args.log_type
            )
    elif (args.mode=='check'):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        env=make_env(test=True)
        
        frames = []
        for i in range(3):
            obs = env.reset()
            done = False
            R = 0
            t = 0
            while not done and t < 200:
                frames.append(env.render(mode = 'rgb_array'))
                action = agent.act(obs)
                obs, r, done, _ = env.step(action)
                R += r
                t += 1
            print('test episode:', i, 'R:', R)
            agent.stop_episode()
        env.close()

        from IPython.display import HTML
        plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off') 
        def animate(i):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),interval=50)
        anim.save(args.save_mp4)
        return anim
