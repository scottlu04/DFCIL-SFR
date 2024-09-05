#!/bin/bash

trial_id=$1
# few_shot_seed_id=$2

src_dir=/home/luzhenyu/DFCIL/SFR/src
#src_dir=/guided_MI/src
# scripts_dir=/ogr_cmu/scripts
#cd ${scripts_dir}


# General config
split_type="agnostic"
gpu=0
datasets=("hgr_shrec_2017")
baselines=("Metric-CL")
trial_id=1

cd ${src_dir}/drivers

#set PYTHONPATH=.

run_driver() {

    python main_driver.py \
    --train ${train} \
    --dataset ${dataset} \
    --split_type ${split_type} \
    --cfg_file ${cfg_file} \
    --root_dir ${root_dir} \
    --log_dir ${log_dir} \
    --gpu ${gpu} \
    --trial_id ${trial_id}
}
#   --save_last_only ${0} \

for dataset_name in ${datasets[*]}; do
    if [ $dataset_name = "hgr_shrec_2017" ] 
    then
        dataset="hgr_shrec_2017"
        root_dir="/media/exx/HDD/zhenyulu/HandGestureDataset_SHREC2017"
    elif [ $dataset_name = "ego_gesture" ]
    then
        dataset="ego_gesture"
        root_dir="/media/exx/HDD/zhenyulu/EgoGesture"
    elif [ $dataset_name = "ntu" ]
    then
        dataset="ntu"
        root_dir="/media/exx/HDD/zhenyulu/EgoGesture"
    fi

    for baseline_name in ${baselines[*]}; do
        ############################ Run baseline ############################
        #Train
        train=1
        cfg_file=/home/luzhenyu/DFCIL/SFR/src/configs/params/$dataset/$baseline_name.yaml
        log_dir=/home/luzhenyu/DFCIL/SFR/output/$dataset/$baseline_name
        trial_id=$trial_id
        run_driver

        #Test
        train=-1
        run_driver
    done
done


#./summarize_results.sh $src_dir "${datasets[*]}" "${baselines[*]}"