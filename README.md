# lunar-depth
python eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --absolute_depth \
    --resize
    --max_gt_distance 100

python eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --relative_depth \
    --resize
    --max_gt_distance 100
