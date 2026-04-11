# Training Commands

## Single Config Training
```bash
# Train with a single config (logs to terminal)
python main.py --config_file configs/train_configs/A/A1/joint_A1.yaml

# Train and save logs to file
python main.py --config_file configs/train_configs/A/A1/joint_A1.yaml 2>&1 | tee results/training_logs/A/joint_A1.log
```

## Batch Training (Multiple Configs)
```bash
# Train all configs in a directory, logs saved to output folder
bash scripts/inference/batch_inference.sh \
    configs/train_configs/A/A1 \
    results/training_logs/A1
```

## Resume Training (if interrupted)
```bash
# Training will auto-resume from last saved epoch if checkpoint exists
# Checkpoints are saved in: runs/exp/<experiment_name>/<model_dir>/
python main.py --config_file configs/train_configs/A/A1/joint_A1.yaml
```

## Monitor Training
```bash
# Watch training logs in real-time
tail -f results/training_logs/A/joint_A1.log

# View loss plots (generated after training completes)
# Location: runs/exp/<experiment_name>/plots/
```