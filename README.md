## Introduction and objectives
This project is part of the [Udacity deep reinforcement learning nanodegree](http://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). The goal is to apply the knowledge obtained during the course to train two agents to collaborate on a task.

## Background
Collaboration and competition of agents is a field of the reinforcement learning that can be approached using multiple solutions. For this project, the selected option is a variation of [DDPG](https://arxiv.org/abs/1509.02971). In the paper the algorithm is introduced as an ["Actor-Critic" method](https://cs.wmich.edu/~trenary/files/cs5300/RLBook/node66.html). Though, some researchers think DDPG is best classified as a [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) method for continuous action spaces, so it is worth it to read and understand how actor-critic methods and DQN methods works.

## Project Details
This project uses the Tennis environment.

![tennis](docs/tennis.png "Tennis")

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically, after each episode, the reward of the episode if the maximum reward of the two agents.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started
This project requires Python 3.6+, pytorch, torchvision, matplotlib and numpy to work. It's recommended to use a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/) to run the project with the appropriate requirements.

For this project, you will need to download the environment from one of the links below. You need only select the environment that matches your operating system:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the root folder, and unzip (or decompress) the file.

## Instructions
To run the project navigate in your terminal to the root folder and execute `python3.6 -m collaboration_and_competition.main --environment path_to_your_environment`. For example, if you use Linux and placed the environment in the root folder following the instructions, the concrete instruction would be `python3.6 -m collaboration_and_competition.main --environment Tennis_Linux/Tennis.x86_64`. To use a trained model, include the option `--trained`: `python3.6 -m collaboration_and_competition.main --trained --environment Tennis_Linux/Tennis.x86_64`