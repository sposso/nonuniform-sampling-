python3 main.py \
    --experiment_name "stride_8" \
    --data_localization "$PROJECT/lsa273_uksr/breastcancer/data/final_data/loc.csv" \
    --model-checkpoint-folder "$PROJECT/lsa273_uksr/breastcancer/runs/checkpoints" \
    --logs-folder "$PROJECT/lsa273_uksr/breastcancer/runs/logs" \
    --devices 4 \
    --nodes $SLURM_JOB_NUM_NODES \
    --epochs 30 \
    --batch_size 8
