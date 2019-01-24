from gym.envs.registration import register

register(id='light-v0',
         entry_point='gym_light.envs:SingleLightEnv')
