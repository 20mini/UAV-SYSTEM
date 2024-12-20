import numpy as np
import sys
sys.path.append('./')

from marllib import marl
from gym.spaces import Box, Dict as GymDict, Discrete, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


# 전역 Unity 환경 인스턴스
unity_env_instance = None

policy_mapping_dict = {
    "multi_agent_mlagent": {
        "description": "three team cooperate",
        "team_prefix": ("agent_0", 'agent_1', 'agent_2'),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    }
}

class MultiAgentMLAgentEnv(MultiAgentEnv):

    def __init__(self, env_config):
        global unity_env_instance
        self.env_path = env_config.get("env_path", None)
        self.no_graphics = env_config.get("no_graphics", True)
        self.worker_id = env_config.get("worker_id", 0)
        self.base_port = env_config.get("base_port", 5005)
        
        self.engine_configuration_channel = EngineConfigurationChannel()
        self.engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
        
        if unity_env_instance is None:
            unity_env_instance = UnityEnvironment(
                file_name=self.env_path,
                no_graphics=self.no_graphics,
                worker_id=self.worker_id,
                base_port=self.base_port,
                side_channels=[self.engine_configuration_channel]
            )
            unity_env_instance.reset()
        
        self.env = unity_env_instance
        self.behavior_names = list(self.env.behavior_specs.keys())
        self.behavior_specs = {name: self.env.behavior_specs[name] for name in self.behavior_names}

        self.agents = [f"agent_{i}" for i in range(len(self.behavior_names))]
        self.num_agents = len(self.agents)

        # self.action_space  = Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape=(self.env.behavior_specs[self.behavior_names[0]].action_spec.continuous_size,),
        #     dtype=np.float32
        # )

        # discrete_action_space = Discrete(31)

        self.action_space = Tuple([
            Box(
                low= -sys.float_info.max,
                high= sys.float_info.max,
                shape=(self.env.behavior_specs[self.behavior_names[0]].action_spec.continuous_size,),
                dtype=np.float32
            ),
            Discrete(self.env.behavior_specs[self.behavior_names[0]].action_spec.discrete_branches[0])
        ])
        
        self.observation_space = GymDict({"obs": Box(
                low=-np.inf, 
                high=np.inf, 
                shape=self.behavior_specs[self.behavior_names[0]].observation_specs[0].shape,
                dtype=np.dtype("float32"))})
    

    def reset(self):
        self.env.reset()
        obs = {}

        for i, behavior_name in enumerate(self.behavior_names):
            decision_steps, terminal_steps = self.env.get_steps(behavior_name)
            obs[self.agents[i]] = {"obs": np.array(decision_steps.obs[0][0])}
        return obs

    def step(self, action_dict):
        for i, behavior_name in enumerate(self.behavior_names):
            agent = self.agents[i]
            action = action_dict[agent]
            action_tuple = ActionTuple()
            action_tuple.add_continuous(np.array(action))
            action_tuple.add_discrete(np.array(action))
            self.env.set_actions(behavior_name, action_tuple)
        
        self.env.step()
        
        obs, rewards, dones = {}, {}, {}
        for i, behavior_name in enumerate(self.behavior_names):
            agent = self.agents[i]
            decision_steps, terminal_steps = self.env.get_steps(behavior_name)
            dones[agent] = len(terminal_steps.agent_id) > 0
            obs[agent] = {"obs": np.array(decision_steps.obs[0][0] if not dones[agent] else terminal_steps.obs[0][0])}
            rewards[agent] = decision_steps.reward[0]
        
        dones["__all__"] = any(dones.values())
        return obs, rewards, dones, {}

    def close(self):
        self.env.close()

    def render(self, mode=None):
        pass  # Unity 환경은 자체적으로 렌더링을 관리

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 5000,
            "policy_mapping_info": policy_mapping_dict,
        }
        return env_info


if __name__ == '__main__':

    # Register new env
    ENV_REGISTRY["multi_agent_mlagent"] = MultiAgentMLAgentEnv
    
    # Initialize env
    env = marl.make_env(environment_name="multi_agent_mlagent", map_name="multi_agent_mlagent")
  
    # Pick mappo algorithms for CTDE
    ippo = marl.algos.ippo(hyperparam_source="common")
    
    # Customize model
    model = marl.build_model(env, ippo, {"core_arch": "mlp", 
                                        "encode_layer": "128-256",
                                        "fc_layer": 3,
                                        "hidden_state_size": 256,
                                        "out_dim_fc_0": 128,
                                        "out_dim_fc_1": 64,})
    
    # Start learning with CTDE structure
    ippo.fit(
        env,  
        model, 
        stop={'episode_reward_mean': 10000000, 'timesteps_total': 1e+7},
        local_mode=True,

        num_gpus=1,
        num_workers=1, 
        share_policy='individual',
        checkpoint_freq=10,
        # restore_path=
        # {
        #     'params_path': r'C:\Users\jiyong\Desktop\custom_MARLlib\exp_results\ippo_mlp_multi_agent_mlagent\ksi_3\params.json',
        #     'model_path': r'C:\Users\jiyong\Desktop\custom_MARLlib\exp_results\ippo_mlp_multi_agent_mlagent\ksi_3\checkpoint_000040\checkpoint-40',
        # }
    )

    # mappo.render(
    #     env,  
    #     model, 
    #     stop={'episode_reward_mean': 10000000, 'timesteps_total': 1e+6},
    #     local_mode=True,
    #     # num_gpus=1,
    #     # num_workers=1, 
    #     share_policy='all',
    #     restore_path=
    #     {
    #         'params_path': r'C:\Users\jiyong\Desktop\custom_MARLlib\exp_results\ippo_mlp_multi_agent_mlagent\47번\params.json',
    #         'model_path': r'C:\Users\jiyong\Desktop\custom_MARLlib\exp_results\ippo_mlp_multi_agent_mlagent\47번\checkpoint_000050\checkpoint-50',
    #     }
    # )