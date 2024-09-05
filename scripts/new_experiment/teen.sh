#!/bin/bash

trial_id=$1
# few_shot_seed_id=$2

src_dir=/home/luzhenyu/DFCIL/guided_MI/src
#src_dir=/guided_MI/src
# scripts_dir=/ogr_cmu/scripts
#cd ${scripts_dir}


# General config
split_type="agnostic"
gpu=0
datasets=("ego_gesture")
# datasets=("ego_gesture")
#datasets=("hgr_shrec_2017"  "ego_gesture")
#baselines=("Base" "Fine_tuning" "Feature_extraction" "LwF" "LwF.MC" "DeepInversion" "ABD" "Rdfcil")
#baselines=("Base-BN" "Fine_tuning-BN" "Feature_extraction-BN" "LwF-BN" "LwF.MC-BN" "DeepInversion-BN" "ABD-BN" "Rdfcil-BN")
baselines=("TEEN-CE")
trial_id=2
few_shot_seed=3
# Run trial
# trial_id=$1
# src_dir=$2
# split_type=$3
# gpu=$4
# datasets=$5
# baselines=$6

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
    --trial_id ${trial_id}\
    --few_shot_seed ${few_shot_seed}
}
#   --save_last_only ${0} \

for dataset_name in ${datasets[*]}; do
    if [ $dataset_name = "hgr_shrec_2017" ] 
    then
        dataset="hgr_shrec_2017"
        #root_dir="/ogr_cmu/data/SHREC_2017"
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
        cfg_file=/home/luzhenyu/DFCIL/guided_MI/src/configs/params/$dataset/$baseline_name.yaml
        log_dir=/home/luzhenyu/DFCIL/guided_MI/output/$dataset/$baseline_name
        trial_id=$trial_id
        run_driver

        #Test
        train=-1
        run_driver
    done
done


#./summarize_results.sh $src_dir "${datasets[*]}" "${baselines[*]}"