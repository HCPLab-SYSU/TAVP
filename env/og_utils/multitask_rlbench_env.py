# Copy from https://github.com/robot-colosseum/rvt_colosseum/blob/main/rvt/utils/multitask_rlbench_env.py
from typing import Type, List, Dict, Any
import os
import json
from omegaconf import DictConfig, OmegaConf
import numpy as np
try:
    from rlbench import ObservationConfig, Environment
except (ModuleNotFoundError, ImportError) as e:
    print("You need to install RLBench: 'https://github.com/stepjam/RLBench'")
    raise e
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from clip import tokenize
from yarr.envs.env import Env, MultiTaskEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition
from yarr.utils.process_str import change_case
from yarr.envs.rlbench_env import (
    ROBOT_STATE_KEYS,
    _extract_obs, 
    _get_cam_observation_elements, 
    _observation_elements
)
from colosseum import (
    ASSETS_CONFIGS_FOLDER,
    ASSETS_JSON_FOLDER,
    TASKS_PY_FOLDER,
    TASKS_TTM_FOLDER,
)
from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import (
    ObservationConfigExt,
    check_and_make,
    name_to_class,
    save_demo,
)
from colosseum.variations.utils import safeGetValue

def get_spreadsheet_config(
    base_cfg: DictConfig, collection_cfg: Dict[str, Any], spreadsheet_idx: int
) -> DictConfig:
    """
    Creates a new config object based on a base configuration, updated with
    entries to match the options from the data collection strategy in JSON
    format for the given spreadsheet index.

    Parameters
    ----------
        base_cfg : DictConfig
            The base configuration for the current task
        collection_cfg : Dict[str, Any]
            The data collection strategy parsed from the JSON strategy file
        spreadsheet_idx : int
            The index in the spreadsheet to use for the current task variation

    Returns
    -------
        DictConfig
            The new configuration object with the updated options for this
            variation
    """
    spreadsheet_cfg = base_cfg.copy()

    collections_variation_cfg = collection_cfg["strategy"][spreadsheet_idx][
        "variations"
    ]
    for collection_var_cfg in collections_variation_cfg:
        var_type = collection_var_cfg["type"]
        var_name = collection_var_cfg["name"]
        var_enabled = collection_var_cfg["enabled"]
        for variation_cfg in spreadsheet_cfg.env.scene.factors:
            if variation_cfg.variation != var_type:
                continue
            else:
                if var_name == "any" or (
                    "name" in variation_cfg and variation_cfg.name == var_name
                ):
                    variation_cfg.enabled = var_enabled

    return spreadsheet_cfg

def get_variation_name(
    collection_cfg: Dict[str, Any], spreadsheet_idx: int
) -> bool:
    return collection_cfg["strategy"][spreadsheet_idx]["variation_name"]



class MultiTaskRLBenchEnv(MultiTaskEnv):

    def __init__(self,
                 task_classes: List[Type[Task]],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 channels_last=False,
                 headless=True,
                 swap_task_every: int = 1,
                 include_lang_goal_in_obs=False,
                 base_cfg_name=None,
                 task_class_variation_idx=None):
        super(MultiTaskRLBenchEnv, self).__init__()

        self._task_classes = task_classes
        self._observation_config = observation_config
        self._channels_last = channels_last
        self._include_lang_goal_in_obs = include_lang_goal_in_obs
        # self._rlbench_env = Environment(
        #     action_mode=action_mode, obs_config=observation_config,
        #     dataset_root=dataset_root, headless=headless)
        self._task = None
        self._task_name = ''
        self._lang_goal = 'unknown goal'
        self._swap_task_every = swap_task_every
        
        self._episodes_this_task = 0
        self._active_task_id = -1

        self._task_name_to_idx = {change_case(tc.__name__):i for i, tc in enumerate(self._task_classes)}
        self._base_cfg_name = base_cfg_name
        self._task_class_variation_idx = task_class_variation_idx
        self._action_mode = action_mode
        self._observation_config = observation_config
        self._dataset_root = dataset_root
        self._headless = headless

    def _set_new_task(self, shuffle=False):
        if shuffle:
            self._active_task_id = np.random.randint(0, len(self._task_classes))
        else:
            self._active_task_id = (self._active_task_id + 1) % len(self._task_classes)
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

    def set_task(self, task_name: str):
        self._active_task_id = self._task_name_to_idx[task_name]
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

        descriptions, _ = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant

    def extract_obs(self, obs: Observation):
        extracted_obs = _extract_obs(obs, self._channels_last, self._observation_config)
        if self._include_lang_goal_in_obs:
            extracted_obs['lang_goal_tokens'] = tokenize([self._lang_goal])[0].numpy()
        return extracted_obs

    def launch(self):
        base_cfg_path = os.path.join(ASSETS_CONFIGS_FOLDER, f"{self._base_cfg_name[self._active_task_id]}.yaml")
        if os.path.exists(base_cfg_path):
            with open(base_cfg_path, 'r') as f:
                base_cfg = OmegaConf.load(f)

        collection_cfg_path: str = (
        os.path.join(ASSETS_JSON_FOLDER, base_cfg.env.task_name) + ".json"
        )
        collection_cfg: Optional[Any] = None
        with open(collection_cfg_path, "r") as fh:
            collection_cfg = json.load(fh)

        if collection_cfg is None:
            return 1

        if "strategy" not in collection_cfg:
            return 1

        num_spreadsheet_idx = len(collection_cfg["strategy"])
        
        if self._task_class_variation_idx != None:
            full_config = get_spreadsheet_config(
                        base_cfg,
                        collection_cfg,
                        self._task_class_variation_idx[self._active_task_id],
                    )
            _, env_cfg = full_config.data, full_config.env  
        else:
            env_cfg = None

        self._rlbench_env = EnvironmentExt(
            action_mode=self._action_mode, obs_config=self._observation_config, 
            path_task_ttms=TASKS_TTM_FOLDER,
            dataset_root=self._dataset_root, headless=self._headless, env_config=env_cfg,)
        self._rlbench_env

        self._rlbench_env.launch()
        self._set_new_task()

    def shutdown(self):
        self._rlbench_env.shutdown()

    def reset(self) -> dict:
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        descriptions, obs = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant
        extracted_obs = self.extract_obs(obs)

        return extracted_obs

    def step(self, action: np.ndarray) -> Transition:
        obs, reward, terminal = self._task.step(action)
        obs = self.extract_obs(obs)
        return Transition(obs, reward, terminal)

    @property
    def observation_elements(self) -> List[ObservationElement]:
        return _observation_elements(self._observation_config, self._channels_last)

    @property
    def action_shape(self):
        return (self._rlbench_env.action_size, )

    @property
    def env(self) -> Environment:
        return self._rlbench_env

    @property
    def num_tasks(self) -> int:
        return len(self._task_classes)



class RLBenchEnv(Env):

    def __init__(self, task_class: Type[Task],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 channels_last=False,
                 headless=True,
                 include_lang_goal_in_obs=False):
        super(RLBenchEnv, self).__init__()
        self._task_class = task_class
        self._observation_config = observation_config
        self._channels_last = channels_last
        self._include_lang_goal_in_obs = include_lang_goal_in_obs
        self._rlbench_env = Environment(
            action_mode=action_mode, obs_config=observation_config,
            dataset_root=dataset_root, headless=headless)
        self._task = None
        self._lang_goal = 'unknown goal'

    def extract_obs(self, obs: Observation):
        extracted_obs = _extract_obs(obs, self._channels_last, self._observation_config)
        if self._include_lang_goal_in_obs:
            extracted_obs['lang_goal_tokens'] = tokenize([self._lang_goal])[0].numpy()
        return extracted_obs

    def launch(self):
        self._rlbench_env.launch()
        self._task = self._rlbench_env.get_task(self._task_class)

    def shutdown(self):
        self._rlbench_env.shutdown()

    def reset(self) -> dict:
        descriptions, obs = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant
        extracted_obs = self.extract_obs(obs)
        return extracted_obs

    def step(self, action: np.ndarray) -> Transition:
        obs, reward, terminal = self._task.step(action)
        obs = self.extract_obs(obs)
        return Transition(obs, reward, terminal)

    @property
    def observation_elements(self) -> List[ObservationElement]:
        return _observation_elements(self._observation_config, self._channels_last)

    @property
    def action_shape(self):
        return (self._rlbench_env.action_size, )

    @property
    def env(self) -> Environment:
        return self._rlbench_env
