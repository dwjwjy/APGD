Environment: \
conda create -n APGD python=3.10 -y \
conda activate APGD \
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 



cd APGD \
pip install -r Requirements.txt \
cd GroundingDINO \
pip install -e .

mkdir weights\
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

Train:\
python demo/train3.py \
    --config_file GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --checkpoint_path weights/groundingdino_swint_ogc.pth \
    --image_path /path/to/images \
    --label_path /path/to/labels.csv \
    --text_prompt "adversarial patchs" \
    --num_epochs 200 \
    --learning_rate 0.0005 \
    --batch_size 8

