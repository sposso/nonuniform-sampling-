python3 main.py \
    --experiment_name "sweeps" \
    --data_localization "$PROJECT/lsa273_uksr/breastcancer/data/raw_data/locations.csv" \
    --model-checkpoint-folder "$PROJECT/lsa273_uksr/breastcancer/runs/checkpoints" \
    --logs-folder "$PROJECT/lsa273_uksr/breastcancer/runs/logs" \
    --devices 4 \
    --nodes $SLURM_JOB_NUM_NODES \
    --epochs 50
