#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
# bash ./eval_og.sh
DEMO_PATH_ROOT=./data/RLBench-OG/Generalization/test
export eval_root_dir=rlbench_og
export eval_episodes=25
export trial=1
export epoch=<select_which_epoch_to_eval>
export model_folder=<path_to_STAGE23_EVAL_WEIGHTS_directory>
MAX_JOBS=6 # max concurrent jobs. If resources are limited, consider reducing this number.
# create a function to run tasks
run_task() {
    local root_task=$1
    local display_num=$2
    local variation=$3
    export DEMO_PATH=$DEMO_PATH_ROOT/$root_task
    export task_list=${root_task}_${variation}
    export eval_dir=$eval_root_dir/epoch_${epoch}_trial_${trial}/$variation/$root_task/  
    export DATA_PATH=$DEMO_PATH/$task_list/
    if [ ! -e "$DATA_PATH" ]; then
        echo "$DATA_PATH does not exist, skipping $task_list."
        return
    fi
    # use xvfb-run to run the evaluation in a virtual display
    xvfb-run -s "-screen 0 1400x900x24" -n $display_num python eval_og.py \
        --model-folder $model_folder  \
        --eval-datafolder $DEMO_PATH \
        --tasks $task_list \
        --eval-episodes $eval_episodes \
        --log-name $eval_dir \
        --device 0 \
        --headless \
        --model-name model_$epoch.pth
    echo "Completed task: $root_task (display :$display_num, variation: $variation)"
}

for variation in 8 9 10 11 12 14 ; do
    echo "Starting evaluation for variation: $variation"
    current_jobs=0
    display_counter=930
    tasks_all='block_pyramid,straighten_rope,close_drawer,solve_puzzle,toilet_seat_down,take_plate_off_colored_dish_rack,take_usb_out_of_computer,scoop_with_spatula,basketball_in_hoop,water_plants'
    IFS=',' read -r -a task_array <<< "$tasks_all"
    for root_task in "${task_array[@]}"; do
        while [ $current_jobs -ge $MAX_JOBS ]; do
            wait -n
            current_jobs=$((current_jobs - 1))
            echo "Job completed. Current jobs: $current_jobs"
        done
        run_task "$root_task" $display_counter $variation &
        current_jobs=$((current_jobs + 1))
        echo "Started task: $root_task on display :$display_counter, variation: $variation. Current jobs: $current_jobs"
        display_counter=$((display_counter + 1))
        sleep 1
    done
    echo "Waiting for all tasks in variation $variation to complete..."
    wait
    echo "All tasks for variation $variation completed."
    sleep 5
done
echo "All variations and tasks completed."