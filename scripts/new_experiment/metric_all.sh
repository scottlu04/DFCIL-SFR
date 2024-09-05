#!/bin/bash

trial_ids=(0 1 2)

#Run trials
for trial_id in ${trial_ids[*]}; do
    ./metric.sh $trial_id 
done