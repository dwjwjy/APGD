Environment: \
conda create -n APGD python=3.10 -y \
conda activate APGD \
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 



cd APGD\
pip install -r Requirements.txt\
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

Data:

https://drive.google.com/drive/folders/1TyngImn4SSK9Be_YBwooSMN2ctqkZKkw

If you want to generate your own training and test sets, please download the code for T-SEA (https://github.com/VDIGPKU/T-SEA). Run evaluation.py and save the output image. T-SEA is a highly effective and inspiring adversarial patch attack method. We sincerely thank the authors for open-sourcing their code.

We provide the adversarial patches used on the training and test sets:

Training set: see train_image/patch\
Test set: see patch/...

The labels of the training set are stored in train_image/label/adversarial_patchs.csv, bbox_x,bbox_y,bbox_width, and bbox_height denote the center point coordinates of the patch, as well as the width and height, respectively. If you use your own training images, replace with the actual positions of the adversarial patches.

Adaptive attack patch：

λ=0（no adaptive attack）：\
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/ad4733b5-acd1-4e8c-b0fa-c2bb84163a0e" />


λ=1\
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/450f0c15-1983-45e0-8033-c972d896dba5" />




