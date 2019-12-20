from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import logging
import os

from chokozainerrl.experiments.evaluator import Evaluator
from chokozainerrl.experiments.evaluator import save_agent
from chainerrl.misc.ask_yes_no import ask_yes_no
from tqdm import tqdm_notebook as tqdm


def save_agent_replay_buffer(agent, t, outdir, suffix='', logger=None):
    logger = logger or logging.getLogger(__name__)
    filename = os.path.join(outdir, '{}{}.replay.pkl'.format(t, suffix))
    agent.replay_buffer.save(filename)
    logger.info('Saved the current replay buffer to %s', filename)


def ask_and_save_agent_replay_buffer(agent, t, outdir, suffix=''):
    if hasattr(agent, 'replay_buffer') and \
            ask_yes_no('Replay buffer has {} transitions. Do you save them to a file?'.format(len(agent.replay_buffer))):  # NOQA
        save_agent_replay_buffer(agent, t, outdir, suffix=suffix)


def train_agent(agent, env, steps, outdir, checkpoint_freq=None,
                max_episode_len=None, step_offset=0, evaluator=None,
                successful_score=None, step_hooks=(), logger=None,log_type=None):

    logger = logger or logging.getLogger(__name__)
    log_type = log_type or 'full_stream'
    if log_type=='pbar' :
        pbar= tqdm(total=steps,position=0)
        if step_offset>0:
            pbar.update(step_offset)
    if checkpoint_freq:
        check_n = step_offset + checkpoint_freq

    #pbar_steps=step_offset
    episode_r = 0
    episode_idx = 0

    # o_0, r_0
    obs = env.reset()
    r = 0

    t = step_offset
    if hasattr(agent, 't'):
        agent.t = step_offset

    episode_len = 0
    try:
        while t < steps:

            # a_t
            action = agent.act_and_train(obs, r)
            # o_{t+1}, r_{t+1}
            obs, r, done, info = env.step(action)
            t += 1
            episode_r += r
            episode_len += 1

            for hook in step_hooks:
                hook(env, agent, t)

            reset = (episode_len == max_episode_len
                     or info.get('needs_reset', False))
            if done or reset or t == steps:
                agent.stop_episode_and_train(obs, r, done=done)
                logger.info('outdir:%s step:%s episode:%s R:%s',
                            outdir, t, episode_idx, episode_r)
                logger.info('statistics:%s', agent.get_statistics())
                # add choko
                if(log_type=='full_stream'):
                    print('outdir:%s step:%s episode:%s R:%s' %(outdir, t, episode_idx, episode_r))
                    print('statistics:%s' % agent.get_statistics())
                if(log_type=='pbar'):
                    if(t-pbar.n)>= 1000:
                        pbar.update(t-pbar.n)
                if checkpoint_freq and t >= check_n :
                    save_agent(agent, t, outdir, logger, suffix='_checkpoint')
                    check_n += checkpoint_freq
                if evaluator is not None:
                    evaluator.evaluate_if_necessary(
                        t=t, episodes=episode_idx + 1)
                    if (successful_score is not None and
                            evaluator.max_score >= successful_score):
                        break
                if t == steps:
                    break
                # Start a new episode
                episode_r = 0
                episode_idx += 1
                episode_len = 0
                obs = env.reset()
                r = 0
            
    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix='_except')
        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix='_finish')

    if log_type=='pbar' :
        pbar.close()


def train_agent_with_evaluation(agent,
                                env,
                                steps,
                                eval_n_steps,
                                eval_n_episodes,
                                eval_interval,
                                outdir,
                                checkpoint_freq=None,
                                train_max_episode_len=None,
                                step_offset=0,
                                eval_max_episode_len=None,
                                eval_env=None,
                                successful_score=None,
                                step_hooks=(),
                                save_best_so_far_agent=True,
                                log_type=None,
                                logger=None,
                                ):
    """Train an agent while periodically evaluating it.

    Args:
        agent: A chainerrl.agent.Agent
        env: Environment train the agent against.
        steps (int): Total number of timesteps for training.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_episodes (int): Number of episodes at each evaluation phase.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output data.
        checkpoint_freq (int): frequency at which agents are stored.
        train_max_episode_len (int): Maximum episode length during training.
        step_offset (int): Time step from which training starts.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If None, train_max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            than or equal to this value if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See chainerrl.experiments.hooks.
        save_best_so_far_agent (bool): If set to True, after each evaluation
            phase, if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)
    log_type = log_type or 'full_stream'
    print("Start")

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = train_max_episode_len

    evaluator = Evaluator(agent=agent,
                          n_steps=eval_n_steps,
                          n_episodes=eval_n_episodes,
                          eval_interval=eval_interval, outdir=outdir,
                          max_episode_len=eval_max_episode_len,
                          env=eval_env,
                          step_offset=step_offset,
                          save_best_so_far_agent=save_best_so_far_agent,
                          logger=logger,
                          log_type=log_type
                          )

    train_agent(
        agent, env, steps, outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=train_max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        step_hooks=step_hooks,
        logger=logger,
        log_type=log_type
        )
