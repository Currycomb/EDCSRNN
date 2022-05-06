#!/usr/bin/python3.6

# train a model
python train.py --gpu_id 0 --gpu_memory_fraction 1 --mixed_precision_training 1 --data_dir "your own data path" --save_weights_dir "your own weights path to be saved" --patch_height 128 --patch_width 128 --input_channels 1 --scale_factor 2 --iterations 200000 --validate_num 500 --batch_size 4 --start_lr 1e-4 --lr_decay_factor 0.5 --load_weights_dir "your own weights path" --optimizer_name "adam"

# predict
python predict.py --input_path "your own path"  --output_path "your own output path" --model_weights "your own model weights path"