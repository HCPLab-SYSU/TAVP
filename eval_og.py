import os
import yaml
import csv
import torch
import cv2
import numpy as np
from omegaconf import OmegaConf
from multiprocessing import Value
from copy import deepcopy
from rlbench.backend import task as rlbench_task
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.agents.agent import VideoSummary
from peract.helpers import utils
from env.og_utils.custom_rlbench_env import (
    CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv
)
from env.og_utils.peract_utils_colosseum import (
    CAMERAS,
    IMAGE_SIZE,
)
from env.og_utils.rlbench_planning import EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning
from utils.rvt_utils import (
    TensorboardManager,
    get_eval_parser,
    COLOSSEUM_TASKS,
)
from colosseum import (
    TASKS_PY_FOLDER,
)
from colosseum.rlbench.utils import (
    name_to_class,
)
from omegaconf import OmegaConf, DictConfig
from termcolor import cprint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

def load_config(config_path: str) -> DictConfig:
    try:
        cfg = OmegaConf.load(config_path)
        return cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    except FileNotFoundError:
        raise FileNotFoundError(f"config not found: {config_path}")
    except Exception as e:
        raise RuntimeError(f"load config failed: {str(e)}")


def load_agent(
    model_path=None,
    exp_cfg_path=None,
    mvt_cfg_path=None,
    eval_log_dir="",
    device=0,
    use_input_place_with_mean=False):

    cfg = load_config(os.path.join(exp_cfg_path, ".hydra", "config.yaml"))
    assert model_path is not None
    model_full_path = os.path.join(exp_cfg_path, model_path)

    py_module = cfg.py_module
    from importlib import import_module
    MOD = import_module(py_module)
    cprint(f"[Info]: Model: {py_module}", "red")
    Policy, PolicyNetwork = MOD.Policy, MOD.PolicyNetwork

    net = PolicyNetwork(cfg.model.hp, cfg.env, render_device=f"cuda:{device}").to(device)
    agent = Policy(net, cfg.model.hp, log_dir=exp_cfg_path)
    agent.build(training=False, device=device)
    agent.load(model_full_path)
    agent.eval()
    if hasattr(agent, 'load_clip'):
        agent.load_clip()
    return agent


@torch.no_grad()
def eval(
    agent,
    tasks,
    eval_datafolder,
    start_episode=0,
    eval_episodes=25,
    episode_length=25,
    replay_ground_truth=False,
    device=0,
    headless=True,
    logging=False,
    log_dir=None,
    verbose=True,
    save_video=False,
):
    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    task_classes = []
    if tasks[0] == "all":
        tasks = COLOSSEUM_TASKS
        if verbose:
            print(f"evaluate on {len(tasks)} tasks: ", tasks)

    
    task_class_variation_idx = []
    task_class_base = []
    for task in tasks:
        task_class_base.append('_'.join(task.split('_')[:-1]))
        if task_class_base[-1] not in task_files:
            raise ValueError('Task %s not recognised!.' % task)
        task_class = name_to_class(task_class_base[-1], TASKS_PY_FOLDER) # task_file_to_task_class(task_class_base)
        task_class_variation_idx.append(int(task.split('_')[-1]))
        task_classes.append(task_class)

    eval_env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=eval_datafolder,
        episode_length=episode_length,
        headless=headless,
        swap_task_every=eval_episodes,
        include_lang_goal_in_obs=True,
        time_in_state=False,
        record_every_n=1 if save_video else -1,
        base_cfg_name=task_class_base,
        task_class_variation_idx=task_class_variation_idx,
    )

    eval_env.eval = True

    device = f"cuda:{device}"

    if logging:
        assert log_dir is not None
        # create metric saving writer
        csv_file = "eval_results.csv"
        if not os.path.exists(os.path.join(log_dir, csv_file)):
            with open(os.path.join(log_dir, csv_file), "w") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_writer.writeheader()

    # evaluate agent
    rollout_generator = RolloutGenerator(device)
    stats_accumulator = SimpleAccumulator(eval_video_fps=30)
    eval_env.launch()

    current_task_id = -1
    num_tasks = len(tasks)
    step_signal = Value("i", -1)

    scores = []
    for task_id in range(num_tasks):
        task_rewards = []
        language_goals=[]
        total_count=0
        for ep in range(start_episode, start_episode + eval_episodes):
            episode_rollout = []
            generator = rollout_generator.generator(
                step_signal=step_signal,
                env=eval_env,
                agent=agent,
                episode_length=episode_length,
                timesteps=1,
                eval=True,
                eval_demo_seed=ep,
                record_enabled=False,
                replay_ground_truth=replay_ground_truth,
            )
            try:
                for replay_transition in generator:
                    episode_rollout.append(replay_transition)
                total_count+=1
            except StopIteration as e:
                assert False
                continue
            except Exception as e:
                eval_env.shutdown()
                raise e

            for transition in episode_rollout:
                stats_accumulator.step(transition, True)
                current_task_id = transition.info["active_task_id"]
                assert current_task_id == task_id

            task_name = tasks[task_id]
            reward = episode_rollout[-1].reward
            task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            language_goals.append(lang_goal)
            if verbose:
                print(
                    f"Evaluating {task_name} | Episode {ep} | Total Episode {total_count} |Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal}"
                )

        # report summaries
        summaries = []
        summaries.extend(stats_accumulator.pop())
        task_name = tasks[task_id]
        if logging:
            # writer csv first
            with open(os.path.join(log_dir, csv_file), "a") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_results = {"task": task_name}
                for s in summaries:
                    if s.name == "eval_envs/return":
                        csv_results["success rate"] = s.value
                    elif s.name == "eval_envs/length":
                        csv_results["length"] = s.value
                    elif s.name == "eval_envs/total_transitions":
                        csv_results["total_transitions"] = s.value
                    if "eval" in s.name:
                        s.name = "%s/%s" % (s.name, task_name)
                csv_writer.writerow(csv_results)
        else:
            for s in summaries:
                if "eval" in s.name:
                    s.name = "%s/%s" % (s.name, task_name)

        if len(summaries) > 0:
            task_score = [
                s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
            ][0]
        else:
            task_score = "unknown"

        print(f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")

        scores.append(task_score)

        if save_video:
            video_image_folder = f"./tmp/{task_name}"
            palette_image_folder = f"./tmp/palette_folder"
            os.makedirs(video_image_folder, exist_ok=True)
            os.makedirs(palette_image_folder, exist_ok=True)
            palette_image_path=os.path.join(palette_image_folder,"palette.png")
            num_succ_video = 25
            num_fail_video = 25
            record_fps = 25
            record_folder = os.path.join(log_dir, "videos")
            os.makedirs(record_folder, exist_ok=True)
            video_success_cnt = 0
            video_fail_cnt = 0
            video_cnt = 0
            for summary in summaries:
                if isinstance(summary, VideoSummary):
                    lang_goal = language_goals.pop(0)
                    lang_goal = lang_goal.replace(" ", "_")
                    video = deepcopy(summary.value)
                    video = np.transpose(video, (0, 2, 3, 1))
                    video = video[:, :, :, ::-1]
                    
                    if (task_rewards[video_cnt] > 99 and video_success_cnt < num_succ_video) or \
                    (not task_rewards[video_cnt] > 99 and video_fail_cnt < num_fail_video):
                        
                        if task_rewards[video_cnt] > 99:
                            video_path = os.path.join(
                                record_folder,
                                f"{task_name}_{lang_goal}_success_{video_success_cnt}.mp4",
                            )
                            video_success_cnt += 1
                        else:
                            video_path = os.path.join(
                                record_folder, f"{task_name}_{lang_goal}_fail_{video_fail_cnt}.mp4"
                            )
                            video_fail_cnt += 1
                        video_cnt += 1
                        if len(video) > 0:
                            height, width = video[0].shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 编码
                            out = cv2.VideoWriter(video_path, fourcc, record_fps, (width, height))
                            for frame in video:
                                out.write(frame)
                            out.release()
                            print(f'video saved - {video_path}')
    eval_env.shutdown()
    if logging:
        csv_fp.close()
    agent.train()
    return scores


def get_model_index(filename):
    """
    :param filenam: path of file of format /.../model_idx.pth
    :return: idx or None
    """
    if len(filename) >= 9 and filename[-4:] == ".pth":
        try:
            index = int(filename[:-4].split("_")[-1])
        except:
            index = None
    else:
        index = None
    return index


def _eval(args):
    assert args.model_name is not None
    model_path=args.model_name
    tb = TensorboardManager(args.eval_log_dir)
    tasks_to_eval = deepcopy(args.tasks)

    model_idx = get_model_index(model_path)
    if model_idx is None:
        model_idx = 0

    agent = load_agent(
        model_path=model_path,
        exp_cfg_path=args.model_folder,
        mvt_cfg_path=args.mvt_cfg_path,
        eval_log_dir=args.eval_log_dir,
        device=args.device,
        use_input_place_with_mean=args.use_input_place_with_mean
    )

    agent_eval_log_dir = os.path.join(
        args.eval_log_dir, os.path.basename(model_path).split(".")[0]
    )

    os.makedirs(agent_eval_log_dir, exist_ok=True)
    scores = eval(
        agent=agent,
        tasks=tasks_to_eval,
        eval_datafolder=args.eval_datafolder,
        start_episode=args.start_episode,
        eval_episodes=args.eval_episodes,
        episode_length=args.episode_length,
        replay_ground_truth=False,
        device=args.device,
        headless=args.headless,
        logging=True,
        log_dir=agent_eval_log_dir,
        verbose=True,
        save_video=False,
        # save_video=True,
    )
    print(f"model {model_path}, scores {scores}")
    task_scores = {}
    for i in range(len(tasks_to_eval)):
        task_scores[tasks_to_eval[i]] = scores[i]
    print("avg score: ", task_scores)
    tb.update("eval", model_idx, task_scores)
    tb.writer.flush()
    tb.close()


if __name__ == "__main__":
    parser = get_eval_parser()
    args = parser.parse_args()
    if args.log_name is None:
        args.log_name = "none"
    args.eval_log_dir = os.path.join(args.model_folder, args.log_name)
    os.makedirs(args.eval_log_dir, exist_ok=True)
    # save the arguments for future reference
    with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
        yaml.dump(args.__dict__, fp)
    _eval(args)
