from multiprocessing import Value

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition
from yarr.agents.agent import ActResult

import numpy as np
import open3d as o3d
import os


def load_and_visualize_point_cloud(filename):
    pcd = o3d.io.read_point_cloud(filename) 
    o3d.visualization.draw_geometries([pcd]) 


class RolloutGenerator(object):

    def __init__(self, env_device = 'cuda:0'):
        self._env_device = env_device

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False,
                  ):

        if eval:
            # print(eval_demo_seed)
            obs = env.reset_to_demo(eval_demo_seed)
            # get ground-truth action sequence
            if replay_ground_truth:
                actions = env.get_ground_truth_action(eval_demo_seed)
        else:
            obs = env.reset()
        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        for step in range(episode_length):

            prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
            lang_goal = env._lang_goal
            prepped_data["language_goal"] = [[[lang_goal]]]  # to match the input of the agent
            prepped_data["lang_goal_tokens"] = prepped_data["lang_goal_tokens"].squeeze(1)

            prepped_data["low_dim_state"] = prepped_data["low_dim_state"].squeeze(1)
            # print(prepped_data["low_dim_state"])
            for n in ["front", "left_shoulder", "right_shoulder", "wrist"]:
                prepped_data["%s_rgb" % n] = prepped_data["%s_rgb" % n].squeeze(1)
                prepped_data["%s_point_cloud" % n] = prepped_data["%s_point_cloud" % n].squeeze(1)
            # print(prepped_data["lang_goal_tokens"].shape)
            if not replay_ground_truth:
                act_result = agent.act(step_signal.value, prepped_data)
                act_result.action = act_result.action[:-1]
                # print(f"Agent action: {act_result.action}") 
                #debug 
                # act_result=ActResult(np.array([0,0,0,0,0,0,0,0,0]))

            else:
                if step >= len(actions):
                    return
                act_result = ActResult(actions[step])

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            transition = env.step(act_result)
            # print(f"Step result - reward: {transition.reward}, terminal: {transition.terminal}, timeout: {step == episode_length - 1}")
            # print(f"Info: {transition.info}")
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)
            # print("transition.terminal:",transition.terminal)
            # print("timeout", timeout)
            # if "needs_reset" in transition.info:
            #     print("奇怪！")
            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)

            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
        # return 0





    def save_pcd(self,visualize_pcd,visualize_rgb,save_path):
        pcd_flat = visualize_pcd.reshape(-1, 3)  
        rgb_flat = visualize_rgb.reshape(-1, 3) 

        def save_point_cloud_with_color(save_path, points, colors):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)  
            pcd.colors = o3d.utility.Vector3dVector(colors)  
            o3d.io.write_point_cloud(save_path, pcd)

        save_point_cloud_with_color(save_path, pcd_flat, rgb_flat)
        print("Point clouds saved in {}", save_path)


    def generator_visualize(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False,
                  visualize_save_dir="",
                  visualize=True,
                  ):

        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
            # get ground-truth action sequence
            if replay_ground_truth:
                actions = env.get_ground_truth_action(eval_demo_seed)
        else:
            obs = env.reset()
        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        visualize_save_dir=os.path.join(visualize_save_dir,env._lang_goal)
        if not os.path.exists(visualize_save_dir):
            os.makedirs(visualize_save_dir)
        for step in range(episode_length):

            prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
            lang_goal = env._lang_goal
            prepped_data["language_goal"] = [[[lang_goal]]]  

            if not replay_ground_truth:
                act_result = agent.act(step, prepped_data,
                                    deterministic=eval,visualize_save_dir=visualize_save_dir,visualize=visualize)

            else:
                if step >= len(actions):
                    return
                act_result = ActResult(actions[step])

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval,visualize_save_dir=visualize_save_dir,visualize=visualize)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)

            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return

