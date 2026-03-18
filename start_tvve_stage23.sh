#!/bin/bash
project_name=tvve_stage23
mkdir -p logs/tvve_stage23
mkdir -p logs/eval_camera_logs
mkdir -p logs/camera

# Replace these placeholders with paths on your machine before running,
# or override them via environment variables.
TRAIN_DEMO_FOLDER=${TRAIN_DEMO_FOLDER:-/path/to/rlbench/data/train}
EVAL_DATA_FOLDER=${EVAL_DATA_FOLDER:-/path/to/rlbench/data/test}
STAGE1_INIT_WEIGHTS=${STAGE1_INIT_WEIGHTS:-/path/to/stage1_checkpoint.pth}
STAGE23_JOINT_INIT_WEIGHTS=${STAGE23_JOINT_INIT_WEIGHTS:-/path/to/stage23_joint_checkpoint.pth}
STAGE23_ONLY_IL_WEIGHTS=${STAGE23_ONLY_IL_WEIGHTS:-/path/to/stage23_only_il_checkpoint.pth}
STAGE23_PPO_IL_WEIGHTS=${STAGE23_PPO_IL_WEIGHTS:-/path/to/stage23_ppo_il_checkpoint.pth}
STAGE23_EVAL_WEIGHTS=${STAGE23_EVAL_WEIGHTS:-/path/to/stage23_eval_checkpoint.pth}

bs=64
epoch=100
gpus=1
moe_x_depth=4
depth=8
amp=false
lr=8.0e-7
il_lr=1.0e-5
optimizer_type=lamb

BRANCH=${1:-b}
update_mode=${2:-1}

case $update_mode in
  1)
    echo "purePPO"
    ppo_il_update=false
    weights=${STAGE1_INIT_WEIGHTS}
    py_module=tvve_stage23
    ;;
  2)
    echo "PPO AND IL"
    ppo_il_update=true
    weights=${STAGE23_JOINT_INIT_WEIGHTS}
    py_module=tvve_stage23
    ;;
  *)
    echo "Error: Invalid branch selection. usage: $0 [1|2]" >&2
    exit 1
    ;;
esac

freeze_camera_policy=false
resume=false

nviews=3
camera_names=[top,left,front]
rmin=0.75
rmax=1.3

only_il_or_only_ppo=${3:-oi}

case $only_il_or_only_ppo in
  op)
    only_il_update=false
    optimizer_type=lamb
    ppo_il_update=false
    bs=48
    epoch=10
    amp=true
    gpus=1
    lr=2.0e-6
    py_module=tvve_stage23
    ;;
  oi)
    only_il_update=true
    optimizer_type=lamb
    ppo_il_update=false
    lr=1.0e-5
    bs=32
    epoch=20
    amp=true
    weights=${STAGE23_ONLY_IL_WEIGHTS}
    freeze_camera_policy=true
    py_module=tvve_stage23
    ;;
  pi)
    gpus=4
    lr=8.0e-7
    il_lr=1.0e-5
    amp=true
    only_il_update=false
    ppo_il_update=true
    optimizer_type=lamb
    bs=96
    epoch=130
    resume=false
    freeze_camera_policy=false
    py_module=tvve_stage23
    weights=${STAGE23_PPO_IL_WEIGHTS}
    ;;
  *)
    echo "Error: Invalid branch selection. usage: $0 [A|B]" >&2
    exit 1
    ;;
esac

case $BRANCH in
  C|c)
    echo "Perform log-free training..."
    LOG_FILE="logs/camera/train_$(date +'%Y%m%d_%H%M%S')_lr_${lr}_bs_${bs}_epoch_${epoch}_gpus_${gpus}_amp_${amp}_depth${depth}_moexdepth_${moe_x_depth}.log"
    python train_tvve_stage23.py \
      config=./configs/tvve_stage23.yaml \
      hydra.job.name=camera_rmin_${rmin}_rmax_${rmax}_or_${only_il_or_only_ppo}_nviews_${nviews} \
      train.num_gpus=1 \
      train.bs=64 \
      train.demo_folder="${TRAIN_DEMO_FOLDER}" \
      train.save_freq=100 \
      py_module=${py_module} \
      model.hp.mvt_cameras=${camera_names} \
      model.hp.rmin=${rmin} \
      model.hp.rmax=${rmax} \
      model.hp.lr=${lr} \
      model.hp.optimizer_type=${optimizer_type} \
      model.hp.il_lr=${il_lr} \
      model.hp.ppo_il_update=${ppo_il_update} \
      model.hp.only_il_update=${only_il_update} \
      model.hp.freeze_camera_policy=${freeze_camera_policy} \
      model.hp.resume=${resume} \
      train.num_workers=8 \
      model.hp.clip_grad_norm=1.0 \
      train.epochs=${epoch} \
      model.hp.depth=${depth} \
      model.hp.moe_x_depth=${moe_x_depth} \
      model.hp.amp=${amp} \
      model.weights="${weights}" \
      env.tasks=stack_cups
    ;;
  A|a)
    echo "Execute training..."
    LOG_FILE="logs/camera/train_$(date +'%Y%m%d_%H%M%S')_camera_rmin_${rmin}_rmax_${rmax}_or_${only_il_or_only_ppo}_nviews_${nviews}.log"
    nohup python train_tvve_stage23.py \
      config=./configs/tvve_stage23.yaml \
      hydra.job.name=camera_rmin_${rmin}_rmax_${rmax}_or_${only_il_or_only_ppo}_nviews_${nviews} \
      train.num_gpus=${gpus} \
      train.demo_folder="${TRAIN_DEMO_FOLDER}" \
      train.bs=${bs} \
      py_module=${py_module} \
      model.hp.mvt_cameras=${camera_names} \
      model.hp.rmin=${rmin} \
      model.hp.rmax=${rmax} \
      model.hp.lr=${lr} \
      model.hp.optimizer_type=${optimizer_type} \
      model.hp.il_lr=${il_lr} \
      model.hp.ppo_il_update=${ppo_il_update} \
      model.hp.only_il_update=${only_il_update} \
      model.hp.freeze_camera_policy=${freeze_camera_policy} \
      model.hp.resume=${resume} \
      train.num_workers=8 \
      train.save_freq=1000 \
      model.hp.clip_grad_norm=1.0 \
      train.epochs=${epoch} \
      wandb=RLBench_arp_plus \
      model.hp.depth=${depth} \
      model.hp.moe_x_depth=${moe_x_depth} \
      model.hp.amp=${amp} \
      model.weights="${weights}" \
      wandb_name=camera_rmin_${rmin}_rmax_${rmax}_or_${only_il_or_only_ppo}_nviews_${nviews}_epoch_${epoch}_lr_${lr}_bs_${bs}_gpus_${gpus}_amp_${amp}_depth${depth}_moexdepth_${moe_x_depth} >> "$LOG_FILE" 2>&1 &
    ;;
  B|b)
    echo "Perform verification..."
    ckpt=${STAGE23_EVAL_WEIGHTS}
    rmin=0.75
    rmax=1.3
    nviews=3
    LOG_FILE="logs/eval_camera_logs/eval_$(date +'%Y%m%d_%H%M%S')_camera_rmin_${rmin}_rmax_${rmax}_or_${only_il_or_only_ppo}_nviews_${nviews}.log"

    model_steps=8334
    gui_id=70

    nohup xvfb-run -s "-screen 0 1400x900x24" -n ${gui_id} python eval.py config=./configs/${project_name}.yaml \
                            model.weights="${ckpt}" \
                            hydra.job.name=eval_camera_rmin_${rmin}_rmax_${rmax}_or_${only_il_or_only_ppo}_nviews_${nviews} \
                            eval.device=0 \
                            eval.datafolder="${EVAL_DATA_FOLDER}" \
                            py_module=${py_module} \
                            model.hp.rmin=${rmin} \
                            model.hp.rmax=${rmax} \
                            model.hp.mvt_cameras=${camera_names} \
                            model.hp.depth=${depth} \
                            model.hp.moe_x_depth=${moe_x_depth} \
                            output_dir=outputs/eval.${project_name}/eval_camera_rmin_${rmin}_rmax_${rmax}_or_${only_il_or_only_ppo}_nviews_${nviews}/model_${model_steps}/`date +"%Y-%m-%d_%H-%M"` >> "$LOG_FILE" 2>&1 &
    ;;
  *)
    echo "Error: Invalid branch selection. usage: $0 [A|B]" >&2
    exit 1
    ;;
esac

echo "-----------------------------------------------"
echo "Execution will continue after the terminal is closed"
echo "Real-time log viewing command: tail -f $LOG_FILE"
echo "process PID: $!"
echo "-----------------------------------------------"
