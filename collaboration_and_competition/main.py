import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from collaboration_and_competition.ddpg_agent import Agent

AGENT_1_ACTOR_CHECKPOINT_PATH = str(Path() / 'collaboration_and_competition' / 'checkpoint_actor1.pth')
AGENT_2_ACTOR_CHECKPOINT_PATH = str(Path() / 'collaboration_and_competition' / 'checkpoint_actor2.pth')
CRITIC_CHECKPOINT_PATH = str(Path() / 'collaboration_and_competition' / 'checkpoint_critic.pth')


def not_trained_mode(agent1, agent2, env, brain_name):
    print_every = 100
    scores_deque = deque(maxlen=print_every)
    scores = []
    episode = 0
    best_episode_score = 0

    while True:
        episode += 1
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state1 = env_info.vector_observations[0]
        state2 = env_info.vector_observations[1]
        agent1.reset()
        agent2.reset()
        score1 = 0
        score2 = 0
        while True:
            actions1 = agent1.act(state1)
            actions2 = agent2.act(state2)
            all_actions = np.concatenate((actions1, actions2))
            env_info = env.step(all_actions)[brain_name]
            next_state1 = env_info.vector_observations[0]
            next_state2 = env_info.vector_observations[1]
            rewards = env_info.rewards
            dones = env_info.local_done
            agent1.step(state1, actions1, rewards[0], next_state1, dones[0])
            agent2.step(state2, actions2, rewards[1], next_state2, dones[1])
            state1 = next_state1
            state2 = next_state2
            score1 += rewards[0]
            score2 += rewards[1]
            if dones[0]:
                break
        max_score = max(score1, score2)
        scores_deque.append(max_score)
        scores.append(max_score)
        if max_score > best_episode_score:
            best_episode_score = max_score
            torch.save(agent1.actor_local.state_dict(), AGENT_1_ACTOR_CHECKPOINT_PATH)
            torch.save(agent2.actor_local.state_dict(), AGENT_2_ACTOR_CHECKPOINT_PATH)
            torch.save(agent1.critic_local.state_dict(), CRITIC_CHECKPOINT_PATH)
            print(f'\nsaved with a score of {max_score}')

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)), end='')
        if episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 0.5:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
            if episode > 100 and best_episode_score > 0.5:
                break

    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def trained_mode(agent1, agent2, env, brain_name):
    agent1.actor_local.load_state_dict(torch.load(AGENT_1_ACTOR_CHECKPOINT_PATH))
    agent1.critic_local.load_state_dict(torch.load(CRITIC_CHECKPOINT_PATH))
    agent2.actor_local.load_state_dict(torch.load(AGENT_2_ACTOR_CHECKPOINT_PATH))
    agent2.critic_local.load_state_dict(torch.load(CRITIC_CHECKPOINT_PATH))

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state1 = env_info.vector_observations[0]
    state2 = env_info.vector_observations[1]
    score1 = 0  # initialize the score
    score2 = 0  # initialize the score
    while True:
        actions1 = agent1.act(state1, add_noise=False)
        actions2 = agent2.act(state2, add_noise=False)
        all_actions = np.concatenate((actions1, actions2))
        env_info = env.step(all_actions)[brain_name]
        next_state1 = env_info.vector_observations[0]
        next_state2 = env_info.vector_observations[1]
        rewards = env_info.rewards  # get the rewards
        dones = env_info.local_done  # see if episode has finished
        state1 = next_state1
        state2 = next_state2
        score1 += rewards[0]
        score2 += rewards[1]
        if dones[0]:  # exit loop if episode finished
            break
    print("Score: {}".format(np.mean(max(score1, score2))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained', help='Load a trained agent', action='store_true')
    parser.add_argument('--environment', help='Path to the environment', default="Tennis_Linux/Tennis.x86_64")
    args = parser.parse_args()

    env = UnityEnvironment(file_name=args.environment)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # size of each action
    action_size = brain.vector_action_space_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=2)
    agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=2)

    if args.trained:
        trained_mode(agent1, agent2, env, brain_name)
    else:
        not_trained_mode(agent1, agent2, env, brain_name)
    env.close()


if __name__ == '__main__':
    main()
