Environment: \
conda create -n APGD python=3.10 -y \
conda activate APGD \
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 



cd APGD\
pip install -r Requirements.txt
pip install -e ./groundingdino

mkdir weights\
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

Train:\
python train.py \
    --config_file GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --checkpoint_path weights/groundingdino_swint_ogc.pth \
    --image_path /path/to/images \
    --label_path /path/to/labels.csv \
    --text_prompt "adversarial patchs" \
    --num_epochs 200 \
    --learning_rate 0.0005 \
    --batch_size 8

python demo/predict.py \
    --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --checkpoint_path weights/groundingdino_swint_ogc.pth \
    --input_folder ./path/to/test_images \
    --target_phrase "adversarial patchs" \
    --embedding_path pt/final_learned_token_embedding.pt \
    --output_dir ./results \
    --box_threshold 0.4 \
    --text_threshold 0.25
