import numpy as np

from unityagents import UnityEnvironment

env_name = "Navigation" # Name of the Unity environment binary to launch
train_mode = False # Whether to run the environment in training or inference mode

env = UnityEnvironment(file_name=env_name)

# Examine environment parameters
print(str(env))

# Set the default brain to work with
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

# Reset the environment
env_info = env.reset(train_mode=train_mode)[default_brain]

# Examine the state space for the default brain
print("Agent state looks like: \n{}".format(env_info.states[0]))

import cv2
# for observation in env_info.observations:
#     print("Agent observations look like:")
#     cv2.imshow('img', observation[0, :, :, :])
#     cv2.waitKey(0)

for episode in range(10):
    env_info = env.reset(train_mode=train_mode)[default_brain]
    done = False
    episode_rewards = 0
    while not done:
        if brain.action_space_type == 'continuous':
            act = np.random.randn(len(env_info.agents), brain.action_space_size)
            # act = np.array([[0.1, 0.0]])
            env_info = env.step(act)[default_brain]
        else:
            env_info = env.step(np.random.randint(0, brain.action_space_size, 
                                                  size=(len(env_info.agents))))[default_brain]
        obs = env_info.observations[0][0]
        cv2.imshow('img', obs)
        cv2.waitKey(1)
        episode_rewards += env_info.rewards[0]
        done = env_info.local_done[0]
    print("Total reward this episode: {}".format(episode_rewards))

env.close()
