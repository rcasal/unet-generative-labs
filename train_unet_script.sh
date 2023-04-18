python unet-generative-labs/train_unet.py \
    --num_epochs 1000 \
    --root_dir "roto_latent" \
    --experiment_name "midas_bs4_1_05_001_0001" \
    --batch_size 4 \
    --lambda_mse 1.0 \
    --lambda_perceptual=0.5 \
    --lambda_l1 0.01 \
    --lr 0.0001 \
    --is_latent \
    --midas \
    #--resume_training \
    #--debug_verbose \
    #--remove_deep_conv \