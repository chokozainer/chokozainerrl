"""An example of training Categorical DQN against OpenAI Gym Envs.
This script is changed from chainerrl examples. 
This script is an example of training an IQN agent against OpenAI
Gym envs. Only discrete spaces are supported.

To solve CartPole-v0, run:
    python train_categorical_dqn_gym.py --env CartPole-v0
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

import chainer.functions as F
import chainer.links as L
from chainer import optimizers
import gym

import chainerrl
from chokozainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import replay_buffer

def make_args(argstr):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='check')
    parser.add_argument('--outdir', type=str, default='result',
                        help='Directory path to save output files.'
                            ' If it does not exist, it will be created.')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--load-agent', type=str, default=None)
    parser.add_argument('--steps', type=int, default=10 ** 8)
    parser.add_argument('--step-offset', type=int, default=0)
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--save-agent', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=1000)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--replay-start-size', type=int, default=50)
    parser.add_argument('--target-update-interval', type=int, default=100)
    parser.add_argument('--target-update-method', type=str, default='hard')
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--n-hidden-channels', type=int, default=12)
    parser.add_argument('--n-hidden-layers', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--minibatch-size', type=int, default=32)
    parser.add_argument('--render-train', action='store_true')
    parser.add_argument('--render-eval', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--reward-scale-factor',type=float, default=1.0)
    parser.add_argument('--log-type',type=str,default="full_stream")
    parser.add_argument('--save-mp4',type=str,default="test.mp4")
    myargs = parser.parse_args(argstr)
                    
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

    def make_env(test):
        env = gym.make(args.env)
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = chainerrl.wrappers.Monitor(env, args.outdir)
        if not test:
            misc.env_modifiers.make_reward_filtered(
                env, lambda x: x * args.reward_scale_factor)
        if ((args.render_eval and test) or
                (args.render_train and not test)):
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_size = env.observation_space.low.size
    action_space = env.action_space

    hidden_size = 64
    q_func = chainerrl.agents.iqn.ImplicitQuantileQFunction(
        psi=chainerrl.links.Sequence(
            L.Linear(obs_size, hidden_size),
            F.relu,
        ),
        phi=chainerrl.links.Sequence(
            chainerrl.agents.iqn.CosineBasisLinear(64, hidden_size),
            F.relu,
        ),
        f=L.Linear(hidden_size, env.action_space.n),
    )
    # Use epsilon-greedy for exploration
    explorer = explorers.LinearDecayEpsilonGreedy(
        args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
        action_space.sample)

    opt = optimizers.Adam(1e-3)
    opt.setup(q_func)

    rbuf_capacity = 50000  # 5 * 10 ** 5
    rbuf = replay_buffer.ReplayBuffer(rbuf_capacity)

    agent = chainerrl.agents.IQN(
        q_func, opt, rbuf, gpu=args.gpu, gamma=args.gamma,
        explorer=explorer, replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        minibatch_size=args.minibatch_size,
    )

    eval_env = make_env(test=True)

    if args.load_agent:
        agent.load(args.load_agent)

    if (args.mode=='train'):
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            eval_env=eval_env,
            step_offset=args.step_offset,
            checkpoint_freq=args.checkpoint_freq,
            train_max_episode_len=timestep_limit,
            log_type=args.log_type
            )

    elif (args.mode=='check'):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        
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
