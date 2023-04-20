models=("resnet50" "resnet101" "resnet152" "swin_tiny_patch4_window7_224" "swin_small_patch4_window7_224" "swin_base_patch4_window7_224" "swin_large_patch4_window7_224" "vit_base_patch16_224" "vit_base_patch8_224" "vit_large_patch16_224" "vit_small_patch32_224" "vit_tiny_patch16_224")

weights=("resnet50_a1_0-14fe96d1.pth" "resnet101_a1h-36d3f2aa.pth" "resnet152_a1h-dc400468.pth" "swin_tiny_patch4_window7_224.pth" "swin_small_patch4_window7_224.pth" "swin_base_patch4_window7_224_22kto1k.pth" "swin_large_patch4_window7_224_22kto1k.pth" "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz" "B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz" "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz" "S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz" "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz")

weight_folder="img_model_weights"

for i in "${!models[@]}"; do
    bash train-ctl.sh run_20230308 ${models[i]} "${weight_folder}/${weights[i]}"
    bash eval-ctl.sh run_20230308 ${models[i]} "${weight_folder}/${weights[i]}"
done