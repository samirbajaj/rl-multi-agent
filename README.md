[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project 3: Collaboration and Competition

### Introduction

For this project, I worked with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

Download the environment from one of the links below that matches your operating system:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

The solution was developed in Python 3.6.9 on a Linux machine running Ubuntu 18.04. Instructions for setting up the required Python modules are available [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). 

### Training the Agents

The training code was added to the Jupyter notebook `Tennis.ipynb`, which was provided in skeletal form as part of the starter code template for this project.

To train the agent, simply run the code cells in the notebook in sequence all the way to the one containing the function `train_maddpg()`. The following two cells initialize the agents and start the training loop that runs until the desired minimum average score is recorded, or 2,000 training episodes have been completed. If an average score of +0.5 over 100 consecutive episodes is obtained, training stops and model parameters are written to disk. The parameters for the Actor network are written to `checkpoint_actor_x.pth`, and those for the Critic network are written to `checkpoint_critic_x.pth` for each agent, which are indexed as `0` and `1`.

These saved parameters are available in the repository.

### Testing the Agents

The Jupyter notebook also includes code that loads the checkpointed weights following a successful training loop and runs the agents in test mode (i.e., with `train_mode=False`) for 10 episodes.

Testing code is implemented in the correspondingly-named function `test_maddpg()`.