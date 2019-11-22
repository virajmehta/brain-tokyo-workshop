import gym
from cartpole_swingup import *
from bipedal_walker import *

env = BipedalWalker()
env.reset()
env.set_weight(1.0)
for _ in range(1000):
    act = env.action_space.sample()
    print(env.step(act))

env.close()
