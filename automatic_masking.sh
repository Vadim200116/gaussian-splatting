#!/bin/bash

# Define the source and target folders
#source_folder="../2_3_person/3_person"
source_folder="$1"
folder="$source_folder/automatic_masking"
# transform to abs path
source_folder=$(realpath $source_folder)
folder=$(realpath $folder)
# Create the target folders
dynamic_pred_folder="$folder/dynamic_pred"
deva_folder="$folder/deva"
output_folder="$folder/output"
masks_folder="$folder/auto_masks_consistent"

echo "source_folder: $dynamic_pred_folder"
mkdir -p "$dynamic_pred_folder"
mkdir -p "$deva_folder"
mkdir -p "$output_folder"
mkdir -p "$masks_folder"
# random port
port=$(( ( RANDOM % 1000 )  + 6000 ))


# Navigate to masked-gaussian-splatting directory and execute training, rendering, and video conversion
(
  conda run -n gaussian_splatting --no-capture-output python train_with_transient.py -s "$source_folder" -m "$dynamic_pred_folder" --port $port
  conda run -n gaussian_splatting --no-capture-output python render_with_transient.py -m "$dynamic_pred_folder"
  python ../images_to_video.py "$dynamic_pred_folder/train/ours_40000/renders" "$dynamic_pred_folder/video.mp4" 5
  python ../images_to_video.py "$dynamic_pred_folder/train/ours_40000/overlay" "$dynamic_pred_folder/overlay.mp4" 5
  python ../images_to_video.py "$dynamic_pred_folder/train/ours_40000/pred" "$dynamic_pred_folder/pred.mp4" 5
)


# Navigate to Tracking-Anything-with-DEVA directory and execute the demo script
#(
#  cd ../Tracking-Anything-with-DEVA
#  rm -r "$source_folder/images/.ipynb_checkpoints"
#  conda run -n base  --no-capture-output python demo/demo_automatic.py \
#    --img_path "$source_folder/images" \
#    --amp --temporal_setting semionline \
#    --size 480 \
#    --disable_long_term \
#    --SAM_NUM_POINTS_PER_BATCH 1 \
#    --SAM_NUM_POINTS_PER_SIDE 32 \
#    --max_num_objects 400 \
#    --chunk_size 1 \
#    --output "$deva_folder"
#  python ../images_to_video.py "$deva_folder/Annotations" "$deva_folder/masks.mp4" 5
#)


# Convert the DEVA masks to a single one according to the dynamic score
#(
#  conda run -n gaussian_splatting --no-capture-output python dynamic_masking.py --mask_path "$deva_folder/Annotations" \
#  --dynamic_score_path "$dynamic_pred_folder/train/ours_40000/pred" --output_mask_path "$masks_folder"
#  python ../images_to_video.py "$masks_folder/color_masks" "$masks_folder/auto_masks.mp4" 5
#)


# Retrain with masked-gaussian-splatting with the DEVA masks
#(
#  conda run -n gaussian_splatting --no-capture-output python train.py -s "$source_folder" -m "$output_folder" \
#  --masked "$masks_folder/masks" --port $port
#  conda run -n gaussian_splatting --no-capture-output python render.py -m "$output_folder"
#  python ../images_to_video.py "$output_folder/train/ours_40000/renders" "$output_folder/video.mp4" 5
#)
