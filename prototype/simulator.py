from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import mediapy

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
import numpy as np
import mujoco
import re
import time
from stable_baselines3.common.vec_env import VecMonitor
import pandas as pd
import matplotlib.pyplot as plt

from morphlib.bodies.robogen.gecko import gecko
from morphlib.bodies.robogen.gecko_long import gecko_long
from morphlib.bodies.robogen.snake import snake
from morphlib.bodies.robogen.gecko_bookmark import gecko_bookmark
from morphlib.bodies.robogen.gecko_halfbookmark import gecko_halfbookmark

from morphlib.bodies.robogen.modules.active_joint import ActiveJoint
from morphlib.brain._make_cpg_network_structure_neighbor import active_hinges_to_cpg_network_structure_neighbor
from morphlib.tools.build_file import build_mjcf
from pyrr import Quaternion
from morphlib.terrains.mujoco_plane import mujoco_plane
from morphlib.terrains.hill_and_valleys import hill_and_valleys
from morphlib.terrains.tennis_balls import tennis_balls

from morphlib.tools.mj_default_sim_setup import mujoco_setup_sim
from functools import partial


class CustomMujocoEnv(gym.Env):
    # TODO: add headless?
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, seed=None, generate_terrain=False):
        """
        Custom MuJoCo environment.

        Args:
            xml_path (str): Path to the MuJoCo XML model file.
            render_mode (str): Either "human" or "rgb_array" for rendering options.
        """
        super().__init__()
        # TODO: set body on initialization
        body = gecko_halfbookmark()

        # TODO: what the fuck is this
        # hill_and_valleys_seeded = partial(hill_and_valleys, seed=seed, generate_terrain=generate_terrain)
        xml = build_mjcf(bodies=[body], body_poss=[[0, 0, 0.1]], body_oris=[Quaternion()], terrain_builder=tennis_balls,
                         sim_setup=mujoco_setup_sim, ts=0.001)

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Rendering attributes
        self.render_mode = render_mode
        self.viewer = False

        self.frame_skip = 10

        pattern = r'[^"]*jointy_0[^"]*'
        matches = re.findall(pattern, xml)
        joints_ys = []
        for string in matches:
            joints_y = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{string[:-1]}{i}") for i in range(3)
                        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{string[:-1]}{i}") != -1]
            joints_ys.append(joints_y)

        self.joints_ys = np.array(joints_ys)

        pattern = r'[^"]*jointx_0[^"]*'
        matches = re.findall(pattern, xml)
        joints_xs = []
        for string in matches:
            joints_x = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{string[:-1]}{i}") for i in range(3)
                        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{string[:-1]}{i}") != -1]
            joints_xs.append(joints_x)

        self.joints_xs = np.array(joints_xs)

        self.num_hinges = len(self.data.ctrl)

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_hinges,), dtype=np.float64
        )

        num_obs = self.num_hinges * 2 + 7 + self.joints_ys.flatten().shape[0] * 2 + self.joints_xs.flatten().shape[
            0] * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64
            # Core pos & ori + hinge pos & vel + joint pos & vel
        )

    def step(self, action):
        """
        Step the simulation forward.
        """
        for _ in range(self.frame_skip):
            # Clip action to the valid range
            action = np.clip(action, self.action_space.low, self.action_space.high)

            # Apply the action
            self.data.ctrl[:] = action

            # Simulate one step
            mujoco.mj_step(self.model, self.data)

        # Collect observation
        observation = self._get_obs()

        # Compute reward
        reward = self._compute_reward()

        # Check termination condition
        terminated = self._check_termination()

        # Additional debug info
        info = {}

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)

        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set the initial state
        for joints_y in self.joints_ys:
            self.data.qpos[np.array(joints_y) + 6] = np.arange(0.4, 0.9, 0.6 / len(joints_y))

        core_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "core")
        self.prev_core_pos = self.data.geom_xpos[core_id].copy()

        observation = self._get_obs()

        return observation, {}

    def render(self, camera="track"):
        """
        Render the simulation.
        """
        if self.render_mode == "human":
            pass
        elif self.render_mode == "rgb_array":
            if not self.viewer:
                self._init_rendering()

            self.renderer.update_scene(self.data, scene_option=self.scene_option, camera=camera)
            pixels = self.renderer.render()
            return pixels

    def close(self):
        """
        Clean up resources.
        """
        # if not self.viewer:
        #     glfw.destroy_window(self.window)
        #     glfw.terminate()
        #     self.viewer = False
        pass

    def _get_obs(self):
        """
        Collect observations from the simulation.
        """
        # Get core position and orientation
        core_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "core")
        core_pos = self.data.geom_xpos[core_id].copy()
        core_matrix = self.data.geom_xmat[core_id].copy()
        core_quat = np.array([0, 0, 0, 0], dtype=np.float64)
        mujoco.mju_mat2Quat(core_quat, core_matrix)

        # Get joint positions and velocities
        hinge_pos = self.data.qpos[7:7 + self.num_hinges]
        hinge_vel = self.data.qvel[6:6 + self.num_hinges]

        if self.joints_ys.size == 0:
            joint_pos_y = np.array([])
            joint_vel_y = np.array([])
        else:
            joint_pos_y = self.data.qpos[self.joints_ys.flatten() + 6]
            joint_vel_y = self.data.qpos[self.joints_ys.flatten() + 5]

        if self.joints_xs.size == 0:
            joint_pos_x = np.array([])
            joint_vel_x = np.array([])
        else:
            joint_pos_x = self.data.qpos[self.joints_xs.flatten() + 6]
            joint_vel_x = self.data.qpos[self.joints_xs.flatten() + 5]

        return np.concatenate(
            [core_pos, core_quat, hinge_pos, hinge_vel, joint_pos_y, joint_vel_y, joint_pos_x, joint_vel_x])

    def _compute_reward(self):
        """
        Compute the reward for the current step.
        """
        # Compute core velocity
        core_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "core")
        core_pos = self.data.geom_xpos[core_id].copy()

        core_vel = core_pos - self.prev_core_pos

        self.prev_core_pos = core_pos.copy()

        return core_vel[0] - np.abs(0.05 * core_vel[1])

    def _check_termination(self):
        """
        Check if the episode is terminated.
        """
        # Example termination: If the agent falls below a threshold
        if self.data.time > 60.0:
            return True

        core_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "core")
        cube_xmat = self.data.geom_xmat[core_id].reshape(3, 3)

        # Local +z axis in the world frame (third column of the rotation matrix)
        local_z_world = cube_xmat[:, 0]

        # World +x axis
        world_x = np.array([1, 0, 0])

        # Dot product to check alignment
        alignment = np.dot(local_z_world, world_x)

        # If alignment is close to 1, it's facing the x-direction
        if np.isclose(alignment, 1.0, atol=0.9):
            return False
        else:
            return True

    def _randomize_state(self):
        """
        Optionally randomize the initial state of the simulation.
        """
        self.data.qpos[:] = np.random.uniform(-0.1, 0.1, self.model.nq)
        self.data.qvel[:] = np.random.uniform(-0.1, 0.1, self.model.nv)

    def _init_rendering(self):
        """
        Initialize rendering resources.
        """
        self.renderer = mujoco.Renderer(self.model)
        self.scene_option = mujoco.MjvOption()
        self.viewer = True


register(
    id="CustomMujoco-v0",
    entry_point=CustomMujocoEnv,
    kwargs={"render_mode": "rgb_array", "seed": 7, "generate_terrain": False},
)


# Create multiple parallel environments
def make_env():
    def _init():
        return gym.make("CustomMujoco-v0")

    return _init


def test_policy(model_path):
    PPO_model = PPO.load(model_path)

    env = gym.make("CustomMujoco-v0")
    frames1 = []
    frames2 = []
    frames3 = []
    ts = 0
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = PPO_model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
            ts = 0

        if ts < env.env.env.data.time * env.metadata["render_fps"]:
            # env.env.env.camera = "trackbirdseye"
            pixels1 = env.env.env.render(camera="trackbirdseye")
            # env.env.env.camera = "trackcom2"
            pixels2 = env.env.env.render(camera="trackcom2")
            # env.env.env.camera = "trackside"
            pixels3 = env.env.env.render(camera="trackside")
            frames1.append(pixels1)
            frames2.append(pixels2)
            frames3.append(pixels3)

            ts += 1

    mediapy.write_video("./test_video1.mp4", frames1, fps=env.metadata["render_fps"])
    mediapy.write_video("./test_video2.mp4", frames2, fps=env.metadata["render_fps"])
    mediapy.write_video("./test_video3.mp4", frames3, fps=env.metadata["render_fps"])

    print(obs)


if __name__ == '__main__':
    # test_policy("./ppo_gecko.zip")
    # Number of parallel environments
    start = time.time()
    num_envs = 10
    model_path = None
    replay_buffer_path = None

    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    env = VecMonitor(env, filename="./logs/")  # Save logs to the "./logs/" directory

    # Train a PPO model
    if model_path is not None:
        PPO_model = PPO.load(model_path, env=env, verbose=1)
    else:
        PPO_model = PPO("MlpPolicy", env, verbose=1)

    if replay_buffer_path is not None:
        PPO_model.load_replay_buffer(replay_buffer_path)

    PPO_model.learn(total_timesteps=30_000_000, reset_num_timesteps=True)  #
    PPO_model.save("ppo_gecko")

    print(f"Training time: {time.time() - start:.2f} s")

    test_policy("./ppo_gecko.zip")

    # Plot the training logs
    log_file = "./logs/monitor.csv"
    data = pd.read_csv(log_file, skiprows=1)  # Skip the first row (comments)

    episode_rewards = data["r"]  # Rewards per episode
    time_steps = data["t"]  # Timesteps at each episode

    # time_steps_per_env = int(len(time_steps)/num_envs)
    # time_steps = np.array(time_steps).reshape((time_steps_per_env, num_envs))
    # episode_rewards = np.array(episode_rewards).reshape((int(len(episode_rewards)/num_envs), num_envs))

    # episode_rewards_mean = np.mean(episode_rewards, axis=1)
    # episode_rewards_std = np.std(episode_rewards, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, episode_rewards, label="Episode Reward")
    # plt.fill_between(time_steps[:,0], episode_rewards_mean - episode_rewards_std, episode_rewards_mean + episode_rewards_std, alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.legend()
    plt.savefig("figure.png")