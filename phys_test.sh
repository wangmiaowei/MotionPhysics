#!/bin/sh
scene_list="test_telephone"
method_type="MotionPhysics" # "DreamPhysics", "PhysDreamer", "PhysFlow","MotionPhysics"
dataset_root="./dataset"
exp_dir="./output/${method_type}/real_scenes"
for s in $scene_list; do
    python ms_simulation_adapted.py --model_path ${dataset_root}/${s} \
    --output_path ${exp_dir}/${s} \
    --method_type ${method_type} \
    --total_batch 40 \
    --physics_config ./config/${method_type}/${s}_config.json 
    ### uncomment --eval for evaluation after training
done