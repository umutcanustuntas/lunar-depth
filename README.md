# lunar-depth
python eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --absolute_depth \
    --resize
    --max_gt_distance 100
    #If Lusnar data will be evaluated and all files starts with moon1, moon2, moon3...
    --per_scene

python eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --relative_depth \
    --resize
    --max_gt_distance 100
    print(f"\nEvaluating {pred_files}:")
