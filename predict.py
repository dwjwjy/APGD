import argparse
import os
import sys
from pathlib import Path
import torch.nn as nn
import numpy as np
import torch
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import logging
import cv2  # Retained for inline visualization logic
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('inference_log.txt')
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set seeds for all random number generators to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class GroundingDINOWithLearnablePrompt(nn.Module):
    """Extend GroundingDINO model to add learnable text prompt functionality"""

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.tokenizer = self.model.tokenizer
        self.bert = self.model.bert
        self.learnable_embedding = nn.ParameterList([])  # Use ParameterList to store multiple learnable embeddings
        self.target_phrase = None
        self.target_token_ids = None
        self.hook_handle = None  # Handle for saving hooks
        self.device = next(base_model.parameters()).device
        logger.info(f"Model initialized on device: {self.device}")

    def init_learnable_text(self, target_phrase="adversarial patchs", embedding_path=None):
        """Initialize learnable text embeddings"""
        self.target_phrase = target_phrase
        tokenized = self.tokenizer(target_phrase, add_special_tokens=False)
        self.target_token_ids = tokenized['input_ids']
        
        logger.info(f"Target phrase: '{target_phrase}'")
        logger.info(f"Token IDs after tokenization: {self.target_token_ids}")

        # Load pretrained learnable embeddings
        if embedding_path and os.path.exists(embedding_path):
            try:
                # Load trained embeddings
                loaded_data = torch.load(embedding_path, map_location=self.device)
                logger.info(f"Loaded data from {embedding_path}, type: {type(loaded_data)}")

                # Handle different types of loaded data
                if isinstance(loaded_data, list):
                    logger.info(f"Loaded data is a list, length: {len(loaded_data)}")
                    if len(loaded_data) > 0:
                        if isinstance(loaded_data[0], torch.Tensor):
                            for i, emb in enumerate(loaded_data):
                                if i < len(self.target_token_ids):
                                    self.learnable_embedding.append(nn.Parameter(emb.to(self.device)))
                        else:
                            try:
                                emb_tensor = torch.tensor(loaded_data, device=self.device)
                                if len(emb_tensor.shape) == 1:
                                    for _ in range(len(self.target_token_ids)):
                                        self.learnable_embedding.append(nn.Parameter(emb_tensor.clone()))
                                elif len(emb_tensor.shape) == 2:
                                    for i in range(min(emb_tensor.shape[0], len(self.target_token_ids))):
                                        self.learnable_embedding.append(nn.Parameter(emb_tensor[i].clone()))
                            except Exception as e:
                                logger.error(f"Error converting list to tensor: {e}")
                                self._init_from_original_embeddings()
                    else:
                        self._init_from_original_embeddings()
                elif isinstance(loaded_data, torch.Tensor):
                    logger.info(f"Loaded data is a tensor, shape: {loaded_data.shape}")
                    if len(loaded_data.shape) == 1 or (len(loaded_data.shape) == 2 and loaded_data.shape[0] == 1):
                        if len(loaded_data.shape) == 1:
                            loaded_data = loaded_data.unsqueeze(0)
                        for _ in range(len(self.target_token_ids)):
                            self.learnable_embedding.append(nn.Parameter(loaded_data.clone()))
                    elif loaded_data.shape[0] == len(self.target_token_ids):
                        for i in range(len(self.target_token_ids)):
                            self.learnable_embedding.append(nn.Parameter(loaded_data[i].clone()))
                    else:
                        self._init_from_original_embeddings()
                else:
                    self._init_from_original_embeddings()

            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
                import traceback
                traceback.print_exc()
                self._init_from_original_embeddings()
        else:
            logger.info("No embedding path provided or file does not exist, initializing with original embeddings")
            self._init_from_original_embeddings()

        if len(self.learnable_embedding) == 0:
            self._init_from_original_embeddings()
        elif len(self.learnable_embedding) < len(self.target_token_ids):
            with torch.no_grad():
                for i in range(len(self.learnable_embedding), len(self.target_token_ids)):
                    token_id = self.target_token_ids[i]
                    token_embedding = self.bert.bert.embeddings.word_embeddings.weight[token_id].clone()
                    self.learnable_embedding.append(nn.Parameter(token_embedding.to(self.device)))

        # Register forward hook
        if self.hook_handle is not None:
            self.hook_handle.remove()
        self.hook_handle = self.bert.bert.embeddings.word_embeddings.register_forward_hook(
            self.embedding_hook
        )

        logger.info(f"Successfully initialized {len(self.learnable_embedding)} learnable embeddings")
        return self.target_phrase

    def _init_from_original_embeddings(self):
        """Initialize learnable embeddings from original embeddings"""
        with torch.no_grad():
            for token_id in self.target_token_ids:
                token_embedding = self.bert.bert.embeddings.word_embeddings.weight[token_id].clone()
                self.learnable_embedding.append(nn.Parameter(token_embedding.to(self.device)))
        logger.info(f"Initialized {len(self.learnable_embedding)} learnable embeddings from original embeddings")

    def embedding_hook(self, module, input_ids_tuple, output_embeds):
        """Forward hook function to replace embeddings at runtime"""
        try:
            input_ids = input_ids_tuple[0]
            for batch_idx in range(input_ids.shape[0]):
                batch_ids = input_ids[batch_idx].tolist()
                for i in range(len(batch_ids) - len(self.target_token_ids) + 1):
                    if batch_ids[i:i + len(self.target_token_ids)] == self.target_token_ids:
                        for j, learnable_emb in enumerate(self.learnable_embedding):
                            if learnable_emb.device != output_embeds.device:
                                learnable_emb = learnable_emb.to(output_embeds.device)
                            output_embeds[batch_idx, i + j] = learnable_emb
            return output_embeds
        except Exception as e:
            logger.error(f"Embedding hook error: {e}")
            return output_embeds

    def forward(self, image, captions=None, **kwargs):
        if image.device != self.device:
            image = image.to(self.device)
        return self.model(image, captions=captions, **kwargs)

    def __del__(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()


def init_prompt(model, target_phrase="adversarial patchs", embedding_path=None):
    """Initialize learnable text prompt"""
    wrapped_model = GroundingDINOWithLearnablePrompt(model)
    target_phrase = wrapped_model.init_learnable_text(
        target_phrase=target_phrase,
        embedding_path=embedding_path
    )
    logger.info(f"Using target phrase: '{target_phrase}'")
    return wrapped_model, target_phrase


def plot_boxes_to_image_box(image_pil, tgt, output_path=None, score_threshold=0.0):
    """Draw detection boxes on the image"""
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    scores = tgt.get("scores", [1.0] * len(boxes))

    assert len(boxes) == len(labels), "boxes and labels must have the same length"

    image_pil = image_pil.copy()
    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)

        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white", font=font)
        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    if output_path:
        image_pil.save(output_path)

    return image_pil, mask


def plot_Binary_mask(image_pil, tgt, output_path=None, score_threshold=0.0):
    """Generate binary mask"""
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    scores = tgt.get("scores", [1.0] * len(boxes))

    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, score in zip(boxes, scores):
        if score < score_threshold:
            continue

        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        border_width = 2
        inner_x0 = x0 + border_width
        inner_y0 = y0 + border_width
        inner_x1 = x1 - border_width
        inner_y1 = y1 - border_width

        if inner_x1 >= inner_x0 and inner_y1 >= inner_y0:
            mask_draw.rectangle([inner_x0, inner_y0, inner_x1, inner_y1], fill=255)
        else:
            mask_draw.rectangle([x0, y0, x1, y1], fill=255)

    return image_pil, mask


def plot_boxes_to_image_mask(image_pil, tgt, output_path=None, score_threshold=0.0):
    """Fill detection box areas with black"""
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    scores = tgt.get("scores", [1.0] * len(boxes))

    image_pil = image_pil.copy()
    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, score in zip(boxes, scores):
        if score < score_threshold:
            continue

        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        border_width = 2
        inner_x0 = x0 + border_width
        inner_y0 = y0 + border_width
        inner_x1 = x1 - border_width
        inner_y1 = y1 - border_width

        if inner_x1 >= inner_x0 and inner_y1 >= inner_y0:
            draw.rectangle([inner_x0, inner_y0, inner_x1, inner_y1], fill=(0, 0, 0))
        else:
            draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))

        mask_draw.rectangle([x0, y0, x1, y1], fill=255)

    if output_path:
        image_pil.save(output_path)

    return image_pil, mask


def load_image(image_path):
    """Load and preprocess image"""
    try:
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None, None


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    """Load GroundingDINO model"""
    try:
        logger.info(f"Loading model config from {model_config_path}")
        args = SLConfig.fromfile(model_config_path)
        args.device = "cuda" if not cpu_only and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {args.device}")

        model = build_model(args)
        logger.info(f"Loading model weights from {model_checkpoint_path}")
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        logger.debug(f"Load result: {load_res}")

        model.eval()
        model = model.to(args.device)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False,
                         token_spans=None, image_name=None):
    """Get object detection output"""
    assert text_threshold is not None or token_spans is not None, "text_threshold and token_spans cannot both be None!"

    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    device = image.device
    if next(model.parameters()).device != device:
        model = model.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption], image_name=image_name)

    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    
    # Filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        else:
            tokenizer = model.model.tokenizer 

        tokenized = tokenizer(caption)
        pred_phrases = []
        for i, (logit, box) in enumerate(zip(logits_filt, boxes_filt)):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        else:
            tokenizer = model.model.tokenizer

        positive_maps = create_positive_map_from_span(
            tokenizer(caption),
            token_span=token_spans
        ).to(device)

        logits_for_phrases = positive_maps @ logits.T
        all_phrases = []
        all_boxes = []

        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            filt_mask = logit_phr > box_threshold
            all_boxes.append(boxes[filt_mask])
            
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(filt_mask.sum().item())])

        if all_boxes and any(len(b) > 0 for b in all_boxes):
            boxes_filt = torch.cat([b for b in all_boxes if len(b) > 0], dim=0).cpu()
        else:
            boxes_filt = torch.zeros(0, 4)

        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


def process_images_in_folder(input_folder, output_folder, model, text_prompt, box_threshold, text_threshold,
                             cpu_only=False, token_spans=None):
    """Process all images in folder"""
    os.makedirs(output_folder, exist_ok=True)
    encoder_vis_dir = os.path.join(output_folder, "encoder_feature_maps")
    os.makedirs(encoder_vis_dir, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f'*{ext}')))
        image_files.extend(list(Path(input_folder).glob(f'*{ext.upper()}')))

    if not image_files:
        logger.warning(f"No image files found in {input_folder}")
        return

    logger.info(f"Found {len(image_files)} image files")
    device = next(model.parameters()).device
    
    # Hook Encoder Output
    feature_storage = {}

    def get_encoder_output(name):
        def hook(model, input, output):
            feature_storage[name] = output
        return hook

    hook_handles = []
    try:
        if hasattr(model, 'model'):
            transformer = model.model.transformer
        else:
            transformer = model.transformer
        h = transformer.enc_output.register_forward_hook(get_encoder_output('enc_feat'))
        hook_handles.append(h)
    except Exception as e:
        logger.error(f"Hook registration failed: {e}")

    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            feature_storage.clear()
            image_pil, image = load_image(image_path)
            if image_pil is None or image is None: continue
            image = image.to(device)
            img_name = Path(image_path).stem

            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, cpu_only=cpu_only, token_spans=token_spans,
                image_name=img_name
            )

            # Visualize Encoder Features (Inline Logic)
            if 'enc_feat' in feature_storage:
                feat = feature_storage['enc_feat'][0].detach().cpu()
                activation_map = feat.norm(dim=1)
                
                if len(image.shape) == 3:
                    tensor_h, tensor_w = image.shape[1], image.shape[2]
                else:
                    tensor_h, tensor_w = image.shape[2], image.shape[3]

                feat_h = int(tensor_h / 8)
                feat_w = int(tensor_w / 8)
                num_tokens = feat_h * feat_w

                if activation_map.shape[0] >= num_tokens:
                    level0_map = activation_map[:num_tokens]
                    try:
                        heatmap = level0_map.reshape(feat_h, feat_w).numpy()
                        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                        heatmap = np.uint8(255 * heatmap)
                        W_orig, H_orig = image_pil.size
                        heatmap_resized = cv2.resize(heatmap, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)
                        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                        save_path = os.path.join(encoder_vis_dir, f"{img_name}_encoder.png")
                        cv2.imwrite(save_path, heatmap_color)
                    except Exception:
                        pass

            size = image_pil.size
            pred_dict = {
                "boxes": boxes_filt,
                "size": [size[1], size[0]],
                "labels": pred_phrases,
            }
            output_pred_path = os.path.join(output_folder, f"{Path(image_path).name}")
            plot_boxes_to_image_mask(image_pil, pred_dict, output_path=output_pred_path)
            _, binary_mask = plot_Binary_mask(image_pil, pred_dict, score_threshold=box_threshold)
            mask_output_dir = os.path.join(output_folder, "masks")
            os.makedirs(mask_output_dir, exist_ok=True)

            mask_save_path = os.path.join(mask_output_dir, f"{Path(image_path).stem}.png")
            binary_mask.save(mask_save_path)
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    for h in hook_handles:
        h.remove()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO Inference", add_help=True)
    # Please modify the following default paths according to your environment
    parser.add_argument("--config_file", "-c", type=str, default="./config/GroundingDINO_SwinT_OGC.py", help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, default="./weights/groundingdino_swint_ogc.pth", help="path to checkpoint file")
    parser.add_argument("--input_folder", "-i", type=str, default="./images", help="path to input folder containing images")
    parser.add_argument("--target_phrase", "-t", type=str, default="adversarial perturbation", help="target phrase")
    parser.add_argument("--output_dir", "-o", type=str, default="./output", help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.4, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help="The positions of phrases of interest.")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only")
    parser.add_argument("--embedding_path", type=str, default="./weights/learned_token_embedding.pt", help="Path to the learned token embedding")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--token_ids", type=str, default=None, help="custom token IDs")

    args = parser.parse_args()

    set_seed(args.seed)

    config_file = args.config_file
    checkpoint_path = args.checkpoint_path
    input_folder = args.input_folder
    target_phrase = args.target_phrase
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    embedding_path = args.embedding_path

    logger.info("Loading base model...")
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    token_ids = None
    if args.token_ids:
        token_ids = [int(id.strip()) for id in args.token_ids.split(',')]
        logger.info(f"Using specified token IDs: {token_ids}")
        target_phrase = None

    logger.info("Initializing learnable prompts...")
    model, target_phrase = init_prompt(
        model,
        target_phrase=target_phrase,
        embedding_path=embedding_path
    )

    if token_spans is not None:
        text_threshold = None
        token_spans = eval(f"{token_spans}")
        logger.info("Using token_spans. Setting text_threshold to None.")

    logger.info(f"Starting image processing, output directory: {output_dir}")
    process_images_in_folder(
        input_folder,
        output_dir,
        model,
        target_phrase,
        box_threshold,
        text_threshold,
        cpu_only=args.cpu_only,
        token_spans=token_spans,
    )

    logger.info("Processing complete!")
