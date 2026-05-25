#!/bin/bash
# Parse training logs into CSV summaries per experiment group.
set -euo pipefail

extract_metric() {
    # Usage: extract_metric "MAE" logfile
    # Line format: "FFT MAE (FFT Label): 1.7578 +/- 0.85"
    grep "FFT $1 " "$2" | tail -1 | sed -E 's/.*\): ([0-9.-]+) \+\/- ([0-9.-]+).*/\1 \2/'
}

LOGDIR="results/training_logs"

for group in "$LOGDIR"/*/; do
    group_name=$(basename "$group")
    csv="$group/${group_name}_training_summary.csv"

    echo "Experiment,Phase,Epoch,TrainLoss,ValLoss,BestEpoch,MinValLoss,MAE,MAE_Std,RMSE,RMSE_Std,MAPE,MAPE_Std,Pearson,Pearson_Std,SNR,SNR_Std" > "$csv"

    for log in "$group"/*.log; do
        [ -f "$log" ] || continue
        exp_name=$(basename "$log" .log)

        phase=$(grep -m1 "Training Epoch" "$log" | sed -E 's/.*==== (Joint|STVEN) Training Epoch:.*/\1/')

        # Extract per-epoch train + val losses (strip CR from progress bars)
        tmp=$(mktemp)
        grep -E "Epoch [0-9]+ (Avg|Average) Loss:" "$log" | tr -d '\r' | awk '{print $2, $NF}' > "$tmp.train"
        grep "Validation Average Loss:" "$log" | tr -d '\r' | awk '{print $NF}' > "$tmp.val"
        paste "$tmp.train" "$tmp.val" > "$tmp.paired"
        n_epochs=$(wc -l < "$tmp.train")

        # Best epoch info
        best_epoch=""
        min_val_loss=""
        if grep -q "best trained epoch" "$log"; then
            best_epoch=$(grep "best trained epoch" "$log" | sed -E 's/.*best trained epoch: ([0-9]+).*/\1/')
            min_val_loss=$(grep "min_val_loss:" "$log" | sed -E 's/.*min_val_loss: ([0-9.]+).*/\1/')
        fi

        # Test metrics via extract_metric helper
        read -r mae mae_std <<< "$(extract_metric "MAE" "$log")"
        read -r rmse rmse_std <<< "$(extract_metric "RMSE" "$log")"
        read -r mape mape_std <<< "$(extract_metric "MAPE" "$log")"
        read -r pearson pearson_std <<< "$(extract_metric "Pearson" "$log")"
        read -r snr snr_std <<< "$(extract_metric "SNR" "$log")"

        i=0
        while read -r ep_num ep_train ep_val; do
            if [ "$i" -eq $((n_epochs - 1)) ]; then
                echo "${exp_name},${phase},${ep_num},${ep_train},${ep_val},${best_epoch},${min_val_loss},${mae},${mae_std},${rmse},${rmse_std},${mape},${mape_std},${pearson},${pearson_std},${snr},${snr_std}"
            else
                echo "${exp_name},${phase},${ep_num},${ep_train},${ep_val},,,,,,,,,,"
            fi
            i=$((i + 1))
        done < "$tmp.paired" >> "$csv"

        rm -f "$tmp" "$tmp.train" "$tmp.val" "$tmp.paired"
    done

    echo "Saved $csv"
done
