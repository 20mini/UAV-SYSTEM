import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

class MultiAgent_DroneManager(gym.Env):
    def __init__(self):
        # Unity 환경 설정
        # engine_configuration_channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(file_name="C:/Users/jiyong/yeongmin/envs/MultiAgent_DroneManager", side_channels=[]) # engine_configuration_channel
        self.env = UnityToGymWrapper(unity_env)

        # Gym 환경 설정
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        return state

    def render(self, mode='human'):
        self.env.render()

    def close(self):
        self.env.close()
