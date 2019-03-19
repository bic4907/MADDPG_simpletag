# MADDPG_simpletag
MADDPG environment to solve openai's 'simple_tag' environment.  
Three(default) predators chase a preyer for reward(10), this environment was shaped by distance of predators and preyers.
Three predator choose action with MADDPG algorithms and the preyer acts with uniform distribution from -1. to 1.


## Dependency
* pytorch==1.0.1  
* tensorboardX  
* Use my environment on [envs](/envs) or...
  * Install the [OpenAI's environment](https://github.com/openai/multiagent-particle-envs) and edit some codes

``` python
# environment.py L29

# self.discrete_action_space = True
self.discrete_action_space = False
```
``` python
# simple_tag.py L92
def agent_reward(self, agent, world):
    # Agents are negatively rewarded if caught by adversaries
    rew = 0
    # shape = False
    shape = True

# simple_tag.py L118
def adversary_reward(self, agent, world):
    # Adversaries are rewarded for collisions with agents
    rew = 0
    # shape = False
    shape = True

```

## Getting started
### Train
```
python train.py --tensorboard
```

### Result
![simple_tag](/screenshot/simple_tag.png)

## Acknowledgement
* [shariqiqbal2810's Pytorch implementation](https://github.com/shariqiqbal2810/maddpg-pytorch) (Most motivated)
* [Kostrikov's Pytorch implementation](https://github.com/ikostrikov/pytorch-ddpg-naf)
