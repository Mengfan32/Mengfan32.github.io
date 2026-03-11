from world.base_world import BaseWorld
import gymnasium as gym


class ContinualSpaceGeneralWorld(BaseWorld):
    def __init__(
        self,
        gym_env_name,
        render_mode,
        max_traj_length=1000,
        env_kwargs=None,
    ):
        super().__init__(gym_env_name)
        assert render_mode in ["human", "rgb_array", None]
        self.env_kwargs = env_kwargs or {}

        if gym_env_name == "gym_navigation:NavigationTrack-v0":
            self.env = gym.make(
                gym_env_name,
                render_mode=render_mode,
                track_id=1,
                **self.env_kwargs,
            )
        elif gym_env_name == "maze-sample-3x3-v0":
            self.env = gym.make(
                gym_env_name,
                enable_render=render_mode,
                **self.env_kwargs,
            )
        else:
            self.env = gym.make(gym_env_name, render_mode=render_mode, **self.env_kwargs)
        self.gym_env_name = gym_env_name
        self.render_mode = render_mode
        self.steps = 0
        self.accu_reward = 0
        self.max_traj_length = max_traj_length
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.discretize = True
        else:
            self.discretize = False

    def reset(self, new_reward=False):
        del self.env
        if not new_reward:
            self.env = gym.make(self.gym_env_name, render_mode=self.render_mode, **self.env_kwargs)
        else:
            self.env = gym.make(
                self.gym_env_name,
                render_mode=self.render_mode,
                healthy_reward=0,
                **self.env_kwargs,
            )

        state, _ = self.env.reset()
        self.steps = 0
        self.accu_reward = 0
        return state

    def step(self, action):
        self.steps += 1
        action = action[0]
        state, reward, terminated, truncated, _ = self.env.step(action)
        self.accu_reward += reward

        if self.steps >= self.max_traj_length or terminated or truncated:
            done = True
        else:
            done = False

        return state, reward, done

    def get_accu_reward(self):
        return self.accu_reward
