# lunar-depth
python eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --absolute_depth \
    --resize

python eval2results.py ground_truth/ predictions/ \
    --config_info config_info \
    --relative \
    --resize
