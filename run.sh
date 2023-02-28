python3 main.py \
    --model-checkpoint-folder "$PROJECT/lsa273_uksr/breastcancer/runs/checkpoints" \
    --logs-folder "$PROJECT/lsa273_uksr/breastcancer/runs/logs" \
    --data_localization "$PROJECT/lsa273_uksr/breastcancer/data/final_data/loc.csv" \
    --devices 4 \
    --nodes $SLURM_JOB_NUM_NODES \
    --epochs 30
