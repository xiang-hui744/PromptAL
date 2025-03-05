#!/bin/bash

acquisitions=("trec" "agnews" "yahoo" "imdb" "agnews" "dbpedia")
for acq in "${acquisitions[@]}"; do

        echo "Executing with - acquisition: $acq"
        python3 run_AL2.py --dataset_name "$acq" --multi_head 4 --instance_tokens 1 --num_virtual_tokens 4
    done
done