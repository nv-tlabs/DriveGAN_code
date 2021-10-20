#!/bin/bash


modelPath=$1
port=$2
latent_decoder_model_path=$3


python server.py  \
    --saved_model ${modelPath} \
    --initial_screen rand \
    --play \
    --seed 222 \
    --gpu 0 \
    --port ${port} \
    --latent_decoder_model_path ${latent_decoder_model_path} \
