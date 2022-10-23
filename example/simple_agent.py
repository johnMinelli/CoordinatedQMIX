import gym
import macad_gym

# Single agent, multi actor, default registered env without planner, actors just go straight

class SimpleAgent(object):
    def __init__(self, actor_configs, env_configs):
        """A simple, deterministic agent able to control n actors
        Args:
            actor_configs: Actor config dict
        """
        self.actor_configs = actor_configs
        self.env_configs = env_configs
        self.action_dict = {}

    def get_action(self, obs):
        """ Returns `action_dict` containing actions for each agent in the env
        """
        for actor_id in self.actor_configs.keys():
            # ... Process obs of each agent and generate action ...
            if self.env_configs["discrete_actions"]:
                self.action_dict[actor_id] = 3  # Drive forward
            else:
                self.action_dict[actor_id] = [1, 0]  # Full-throttle
        return self.action_dict

if __name__ == "__main__": 
    # standard registered env
    env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")
    configs = env.configs
    agent = SimpleAgent(configs["actors"], configs["env"])  # Plug-in your agent or use MACAD-Agents
    for ep in range(2):
        obs = env.reset()
        done = {"__all__": False}
        step = 0
        while not done["__all__"]:
            obs, reward, done, info = env.step(agent.get_action(obs))
            print(f"Step#:{step}  Rew:{reward}  Done:{done}")
            step += 1
    env.close()