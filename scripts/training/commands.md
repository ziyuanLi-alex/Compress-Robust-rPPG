# Clear cache
ls ./*-cache
ls ./*-cache/*
rm ./*-cache/*

# Batch

bash scripts/inference/batch_inference.sh \
    configs/train_configs/A/A1 \
    results/training_logs/A1