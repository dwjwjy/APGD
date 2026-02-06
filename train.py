import argparse
import os
import sys
import ast
from pathlib import Path
from typing import List, Dict, Any

S = '0'

os.environ['CUDA_VISIBLE_DEVICES'] = S
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
import torch.nn as nn
import tqdm
import random
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.multiprocessing as mp
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from groundingdino.util.losses import SetCriterion
from groundingdino.util.matchers import build_matcher
from config import ConfigurationManager, DataConfig, ModelConfig
from groundingdino.datasets.dataset import GroundingDINODataset
from groundingdino.util.misc import nested_tensor_from_tensor_list

def set_seed(seed):
    """Set seeds for all random number generators to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Extra settings to ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class GroundingDINOWithLearnablePrompt(nn.Module):
    """Extends GroundingDINO model to add learnable text prompt functionality (using all embeddings of 'adversarial patchs')"""

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.tokenizer = self.model.tokenizer
        self.bert = self.model.bert
        self.learnable_embedding = nn.ParameterList()  # Use ParameterList instead of ModuleList
        self.phrase_token_ids = None  # Store token IDs of the phrase
        self.placeholder_token = None
        self.hook_handle = None  # Handle to save the hook

    def init_learnable_text(self, phrase="adversarial patchs", placeholder_token="sks", tokenizer=None):
        """Initialize learnable text embeddings using all token embeddings of the specified phrase"""
        if tokenizer is None:
            tokenizer = self.tokenizer

        device = next(self.parameters()).device
        tokenized = tokenizer(phrase, return_tensors="pt")
        input_ids = tokenized["input_ids"][0].tolist()  # Remove batch dimension, convert to list

        if input_ids[0] == tokenizer.cls_token_id:
            input_ids = input_ids[1:]
        if input_ids[-1] == tokenizer.sep_token_id:
            input_ids = input_ids[:-1]

        print(f"Token IDs for '{phrase}': {input_ids}")

        self.phrase_token_ids = input_ids
        self.learnable_embedding = nn.ParameterList()
        with torch.no_grad():
            for token_id in input_ids:
                # Get token embedding from the word embedding matrix
                token_embedding = self.bert.bert.embeddings.word_embeddings.weight[token_id].clone()
                # Create parameter and add to ParameterList
                self.learnable_embedding.append(nn.Parameter(token_embedding.to(device)))

        print(f"Initialized {len(self.learnable_embedding)} learnable embeddings")

        self.placeholder_token = placeholder_token

        # Register forward hook
        if self.hook_handle is not None:
            self.hook_handle.remove()  # Remove existing hook if any
        self.hook_handle = self.bert.bert.embeddings.word_embeddings.register_forward_hook(
            self.embedding_hook
        )

        # Return the original phrase as the prompt
        return self.learnable_embedding, self.phrase_token_ids, placeholder_token, phrase

    def embedding_hook(self, module, input_ids_tuple, output_embeds):
        """
        Forward hook function to replace embeddings at runtime.
        When token IDs of the phrase are detected, replace them with learnable embeddings.
        """
        # input_ids_tuple is a tuple, we need the first element
        input_ids = input_ids_tuple[0]

        # For each sequence in the batch
        for batch_idx in range(input_ids.shape[0]):
            seq = input_ids[batch_idx].tolist()

            # Find the position of the phrase token sequence in the input
            for i in range(len(seq) - len(self.phrase_token_ids) + 1):
                if seq[i:i + len(self.phrase_token_ids)] == self.phrase_token_ids:
                    # Match found, replace embeddings
                    for j, learnable_emb in enumerate(self.learnable_embedding):
                        output_embeds[batch_idx, i + j] = learnable_emb

        return output_embeds

    def get_learnable_params(self):
        """Returns a list of all learnable parameters for the optimizer"""
        return self.learnable_embedding  # ParameterList can be passed directly to optimizer

    def forward(self, image, captions=None, **kwargs):
        """Forward pass. The hook automatically handles embedding replacement."""
        return self.model(image, captions=captions, **kwargs)

    def __del__(self):
        # Ensure hook is removed when object is destroyed to prevent memory leaks
        if self.hook_handle is not None:
            self.hook_handle.remove()


# ====================================================================================

def init_prompt(model, phrase="adversarial patchs", placeholder_token="sks"):
    """
    Initialize learnable text prompt using all token embeddings of the specified phrase.
    Args:
        model: Base model
        phrase: The phrase whose embeddings will be used
        placeholder_token: Placeholder token (actually not used in this implementation)
    Returns:
        Wrapped model and the new text prompt
    """
    # Create wrapped model
    wrapped_model = GroundingDINOWithLearnablePrompt(model)
    phrase = phrase.lower().strip()
    if not phrase.endswith("."):
        phrase = phrase + "."
    # Initialize learnable text
    learnable_embeddings, phrase_token_ids, _, text_prompt = wrapped_model.init_learnable_text(
        phrase=phrase,
        placeholder_token=placeholder_token,
        tokenizer=model.tokenizer
    )

    print(f"Initialized learnable prompt: '{text_prompt}'")
    print(f"Using token IDs for phrase: {phrase_token_ids}")

    # Check if learnable embeddings were initialized correctly
    if not learnable_embeddings:
        print("Warning: Learnable embeddings were not initialized correctly!")
    else:
        print(f"Number of learnable embeddings: {len(learnable_embeddings)}")
        for i, emb in enumerate(learnable_embeddings):
            print(f"  Embedding {i} shape: {emb.shape}, Requires grad: {emb.requires_grad}")

    return wrapped_model, text_prompt


# Custom collate function to handle PIL images
def custom_collate_fn(batch):
    """
    Custom collate function to handle batches containing PIL images.
    """
    if len(batch) == 0:
        return {}

    elem = batch[0]
    result = {}

    # Process each key
    for key in elem:
        try:
            if key == 'image_pil':
                # PIL images do not need stacking, return as a list
                result[key] = [item[key] for item in batch]
            elif key == 'image_tensor':
                # Image tensors need stacking
                result[key] = torch.stack([item[key] for item in batch])
            elif key == 'label':
                # Labels might be None or tensors, handle separately
                result[key] = [item[key] for item in batch]
            elif isinstance(elem[key], torch.Tensor):
                # For other tensors, try stacking
                try:
                    result[key] = torch.stack([item[key] for item in batch])
                except:
                    result[key] = [item[key] for item in batch]
            else:
                # Other keys (strings, etc.) return as lists
                result[key] = [item[key] for item in batch]
        except Exception as e:
            print(f"Error processing key '{key}': {str(e)}")
            print(f"Type of key '{key}': {[type(item.get(key)) for item in batch]}")
            # On error, still add key but with None values
            result[key] = [None] * len(batch)

    return result


# Custom Dataset class for batch processing
class ImageDataset(Dataset):
    def __init__(self, image_paths, label_dir=None, transform=None):
        """
        Initialize dataset
        Args:
            image_paths: List of image paths
            label_dir: Directory for labels (if any)
            transform: Image transformation function
        """
        self.image_paths = image_paths
        self.label_dir = label_dir
        self.transform = transform or self._default_transform

    def _default_transform(self, image):
        """Default transformation function"""
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return transform(image, None)[0]

    def _load_label(self, img_path):
        """Load corresponding label file"""
        if not self.label_dir:
            return None
        img_stem = Path(img_path).stem
        label_path = Path(self.label_dir) / f"{img_stem}.txt"
        if not label_path.exists():
            return None
        try:
            # Load image to get dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size

            boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Check if normalization is needed
                        if x_center > 1 or y_center > 1 or width > 1 or height > 1:
                            x_center /= img_width
                            y_center /= img_height
                            width /= img_width
                            height /= img_height

                        boxes.append([x_center, y_center, width, height])
            return torch.tensor(boxes, dtype=torch.float32) if boxes else None
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")
            return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load image
        try:
            image_pil = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image_pil)

            # Load label (if any)
            label = self._load_label(img_path)

            return {
                "image_path": str(img_path),
                "image_name": Path(img_path).name,
                "image_pil": image_pil,
                "image_tensor": image_tensor,
                "label": label
            }
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            # Return an empty dummy image
            dummy_image = Image.new("RGB", (100, 100), color=(0, 0, 0))
            return {
                "image_path": str(img_path),
                "image_name": Path(img_path).name,
                "image_pil": dummy_image,
                "image_tensor": self.transform(dummy_image),
                "label": None,
                "error": str(e)
            }


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, label in zip(boxes, labels):
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        font = ImageFont.load_default()
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font=font)
        else:
            w, h = draw.textsize(str(label), font=font)
            bbox = (x0, y0, x0 + w, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white", font=font)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if (not cpu_only and torch.cuda.is_available()) else "cpu"
    model = build_model(args)

    # Load pretrained weights
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold,
                         text_threshold=None, with_logits=True,
                         cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, \
        "text_threshold and token_spans cannot be None at the same time"

    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if (not cpu_only and torch.cuda.is_available()) else "cpu"
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else model.model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        for logit in logits_filt:
            phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            if with_logits:
                pred_phrases.append(phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(phrase)
    else:
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else model.model.tokenizer
        positive_maps = create_positive_map_from_span(
            tokenizer(caption),
            token_span=token_spans
        ).to(image.device)  # (n_phrase, 256)

        logits_for_phrases = positive_maps @ logits.T  # (n_phrase, nq)
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (span_group, logit_phr) in zip(token_spans, logits_for_phrases):
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in span_group])
            filt_mask = logit_phr > box_threshold
            sel_boxes = boxes[filt_mask]
            sel_logits = logit_phr[filt_mask]
            if sel_boxes.shape[0] == 0:
                continue
            all_boxes.append(sel_boxes)
            all_logits.append(sel_logits)
            if with_logits:
                all_phrases.extend([phrase + f"({str(l.item())[:4]})" for l in sel_logits])
            else:
                all_phrases.extend([phrase] * sel_boxes.shape[0])
        if len(all_boxes):
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        else:
            boxes_filt = torch.zeros((0, 4))
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases

# Ensure multiprocessing sharing strategy
try:
    mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass  # Ignore if already set


def prepare_batch(batch, device):
    images, targets = batch
        # Convert list of images to NestedTensor and move to device
    if isinstance(images, (list, tuple)):
        images = nested_tensor_from_tensor_list(images)  # Convert list to NestedTensor
    images = images.to(device)
    captions = []
    for target in targets:
        target['boxes'] = target['boxes'].to(device)
        target['size'] = target['size'].to(device)
        target['labels'] = target['labels'].to(device)
        # Use caption (already processed string) for the caption
        captions.append(target['caption'])

    return images, targets, captions

def train_prompt(
        model,
        dataloader,
        text_prompt,
        num_epochs=100,
        learning_rate=0.3,
        cpu_only=False,
        # --- New arguments for building loss function (derived from GroundingDINOTrainer) ---
        class_loss_coef=1.0,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,  # Typically 2.0 in DETR/DINO, though your trainer might use 1.0
        eos_coef=0.1,
        max_txt_len=256,
        output_dir="output"
):
    print(learning_rate, "Learning Rate check")
    """
    Train learnable text prompt (Refactored).

    This version integrates the refined SetCriterion construction logic used in GroundingDINOTrainer.

    Args:
        model (nn.Module): Model wrapping GroundingDINO with learnable prompts.
        dataloader (DataLoader): Training data loader.
        text_prompt (str): Text prompt, e.g., "adversarial patch sks.".
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        cpu_only (bool): Whether to use CPU only.
        class_loss_coef (float): Weight for classification loss.
        bbox_loss_coef (float): Weight for BBox L1 loss.
        giou_loss_coef (float): Weight for GIoU loss.
        eos_coef (float): Relative weight for 'no object' class.
        max_txt_len (int): Maximum length for text encoder.
        output_dir (str): Directory to save checkpoints.
    """
    device = "cuda" if not cpu_only and torch.cuda.is_available() else "cpu"
    model.to(device)

    # Use a relative path derived from output_dir for checkpoints
    pt_path = os.path.join(output_dir, "checkpoints")
    os.makedirs(pt_path, exist_ok=True)

    # Get learnable parameters
    if hasattr(model, 'get_learnable_params'):
        # Use new method to get parameters
        learnable_params = model.get_learnable_params()
    else:
        # Fallback to old method
        learnable_params = []
        if hasattr(model, 'learnable_embedding'):
            if isinstance(model.learnable_embedding, nn.ParameterList):
                # If ParameterList, use directly
                learnable_params = model.learnable_embedding
            else:
                # If single parameter
                model.learnable_embedding.requires_grad_(True)
                learnable_params.append(model.learnable_embedding)

    # Check if trainable parameters exist
    if not learnable_params:
        print("Error: No trainable embedding parameters found in model!")
        return []

    print(f"Starting training... Total {len(learnable_params)} trainable parameters.")
    optimizer = torch.optim.AdamW(learnable_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate * 0.1)

    matcher = build_matcher(
        set_cost_class=class_loss_coef * 2,  # Classification weight in matching cost is typically higher
        set_cost_bbox=bbox_loss_coef,
        set_cost_giou=giou_loss_coef
    )

    losses_to_compute = ['labels', 'boxes']
    weight_dict = {
        'loss_ce': class_loss_coef,
        'loss_bbox': bbox_loss_coef * 2,
        'loss_giou': giou_loss_coef
    }

    criterion = SetCriterion(max_txt_len, matcher, eos_coef, losses_to_compute)
    criterion.to(device)
    print("SetCriterion built using GroundingDINOTrainer logic.")
    embedding_distance_loss = 0
    cc = 0
    # --- 3. Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        criterion.train()

        epoch_loss_agg = {}
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for batch_idx, batch in enumerate(progress_bar):
            image_tensors, targets, original_captions = prepare_batch(batch, device)
            batch_size = len(targets)  # Get batch size from targets
            # Prepare targets format
            captions = [text_prompt] * batch_size

            outputs = model(image_tensors, captions=captions)
            
            # <<< Mod: Pass captions and tokenizer when calling criterion >>>
            loss_dict = criterion(outputs, targets, captions=captions, tokenizer=model.tokenizer)
            
            # <<< Mod: Calculate total loss using weight_dict defined above >>>
            total_loss = 1 * sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # Backpropagation and Optimization
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(learnable_params, max_norm=1.0)
            
            # Optional: Print norm for debugging (commented out to reduce clutter)
            # print(torch.norm(learnable_params[1], p=2).item(), "Gradient Norm Check")
            
            optimizer.step()

            # Record and display loss
            num_batches += 1
            for k, v in loss_dict.items():
                if k in weight_dict:
                    epoch_loss_agg[k] = epoch_loss_agg.get(k, 0.0) + v.item()
            epoch_loss_agg['total_loss'] = epoch_loss_agg.get('total_loss', 0.0) + total_loss.item()

            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        # Post-epoch processing
        scheduler.step()

        # Print Epoch Average Loss
        print(f"\n--- Epoch {epoch + 1} Summary ---")
        avg_loss_str = [f"Avg {k}: {v / num_batches:.4f}" for k, v in sorted(epoch_loss_agg.items())]
        print(" | ".join(avg_loss_str))
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * (len("--- Epoch Summary ---") + 4))

        # Save learnable embeddings
        embedding_path = f"{pt_path}/learned_token_embedding_epoch{epoch + 1}.pt"
        # Check type of learnable_embedding and save correctly
        if isinstance(model.learnable_embedding, nn.ParameterList):
            # If ParameterList, save as list
            embeddings_list = [param.detach().cpu() for param in model.learnable_embedding]
            torch.save(embeddings_list, embedding_path)
        else:
            # If single parameter
            torch.save(model.learnable_embedding.detach().cpu(), embedding_path)
        print(f"Saved learnable embedding to: {embedding_path}\n")

    # --- 4. Training Finished ---
    model.eval()
    print("Training completed.")
    final_embedding_path = f"{pt_path}/final_learned_token_embedding.pt"
    # Check type and save correctly
    if isinstance(model.learnable_embedding, nn.ParameterList):
        embeddings_list = [param.detach().cpu() for param in model.learnable_embedding]
        torch.save(embeddings_list, final_embedding_path)
    else:
        torch.save(model.learnable_embedding.detach().cpu(), final_embedding_path)
    print(f"Saved final learnable embedding to: {final_embedding_path}")

    # --- DELETED: The inference call that was causing the crash ---
    # The function 'inference_with_model' (or batch_inference) call here is removed as requested.
    # Training simply ends here.
    return []



def evaluate_model(model, dataloader, text_prompt, box_threshold, text_threshold,
                   cpu_only=False, max_samples=5):
    """Evaluate model performance"""
    device = "cuda" if (not cpu_only and torch.cuda.is_available()) else "cpu"
    model = model.to(device)
    model.eval()

    total_detections = 0
    total_images = 0
    total_iou = 0.0
    total_valid_iou = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_samples:
                break

            image_tensors = batch["image_tensor"].to(device)
            labels = batch["label"]

            for j in range(len(image_tensors)):
                image_tensor = image_tensors[j:j + 1]
                gt_boxes = labels[j]

                # Run inference
                outputs = model(image_tensor, captions=[text_prompt])

                # Extract predictions
                logits = outputs["pred_logits"].sigmoid()[0]
                boxes = outputs["pred_boxes"][0]

                # Filter predictions
                filt_mask = logits.max(dim=1)[0] > box_threshold
                pred_boxes = boxes[filt_mask]

                # Calculate detection count
                num_detections = pred_boxes.shape[0]
                total_detections += num_detections
                total_images += 1

                # If GT boxes exist, calculate IoU
                if gt_boxes is not None and len(gt_boxes) > 0 and num_detections > 0:
                    gt_boxes = gt_boxes.to(device)

                    # Convert to xyxy format
                    pred_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes)
                    gt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes)

                    # Calculate IoU
                    iou_matrix = box_ops.box_iou(pred_boxes_xyxy, gt_boxes_xyxy)[0]
                    max_iou = iou_matrix.max(dim=1)[0].mean().item()

                    total_iou += max_iou
                    total_valid_iou += 1

    # Calculate averages
    avg_detections = total_detections / max(1, total_images)
    avg_iou = total_iou / max(1, total_valid_iou)

    return {
        "avg_detections": avg_detections,
        "avg_iou": avg_iou
    }


def batch_inference(model, dataloader, text_prompt, box_threshold, text_threshold,
                    token_spans=None, cpu_only=False):
    """
    Batch Inference

    Args:
        model: The model
        dataloader: Data loader
        text_prompt: Text prompt
        box_threshold: BBox threshold
        text_threshold: Text threshold
        token_spans: token spans
        cpu_only: Whether to use CPU only

    Returns:
        List of results, each containing image path, predicted boxes, and labels
    """
    device = "cuda" if (not cpu_only and torch.cuda.is_available()) else "cpu"
    model = model.to(device)

    results = []
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batch Inference")):
        print(batch)
        image_paths = batch["image_path"]
        image_names = batch["image_name"]
        image_tensors = batch["image_tensor"]
        image_pils = batch["image_pil"]

        # Process images in batch individually
        for i in range(len(image_paths)):
            image_path = image_paths[i]
            image_name = image_names[i]
            image_tensor = image_tensors[i].to(device)
            image_pil = image_pils[i]

            # If error flag exists, skip
            if "error" in batch and batch["error"][i]:
                print(f"Skipping image with error: {image_path}")
                continue

            # Execute inference
            try:
                boxes_filt, pred_phrases = get_grounding_output(
                    model, image_tensor, text_prompt, box_threshold, text_threshold,
                    cpu_only=cpu_only, token_spans=token_spans
                )

                # Save results
                results.append({
                    "image_path": image_path,
                    "image_name": image_name,
                    "image_pil": image_pil,
                    "boxes": boxes_filt,
                    "labels": pred_phrases
                })

            except Exception as e:
                print(f"  - Error processing image {image_path}: {e}")

    return results


def collect_images(path_str: str,
                   extensions: List[str],
                   recursive: bool) -> List[Path]:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    exts = tuple("." + e.lower().strip() for e in extensions if e.strip())
    results = []
    if p.is_file():
        if p.suffix.lower() in exts:
            results.append(p)
        else:
            raise ValueError(f"File extension not supported: {p.suffix}")
    else:
        if recursive:
            it = p.rglob("*")
        else:
            it = p.glob("*")
        for f in it:
            if f.is_file() and f.suffix.lower() in exts:
                results.append(f)
    results.sort()
    return results


def process_results(results, out_root, save_raw, suffix):
    """Process inference results and save images"""
    meta_results = []

    for result in results:
        image_path = result["image_path"]
        image_name = result["image_name"]
        image_pil = result["image_pil"]
        boxes = result["boxes"]
        labels = result["labels"]

        # Prepare to draw bounding boxes
        size = image_pil.size
        pred_dict = {
            "boxes": boxes,
            "size": [size[1], size[0]],  # H,W
            "labels": labels,
        }

        # Draw prediction boxes
        image_with_box = plot_boxes_to_image(image_pil.copy(), pred_dict)[0]

        # Output path
        stem = Path(image_path).stem
        out_dir = out_root
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        if save_raw:
            raw_path = out_dir / f"{stem}_raw.jpg"
            image_pil.save(raw_path)
        pred_path = out_dir / f"{stem}{suffix}.png"
        image_with_box.save(pred_path)

        # Record metadata
        meta_result = {
            "image": image_path,
            "output": str(pred_path),
            "num_boxes": int(boxes.shape[0]),
            "labels": labels
        }
        meta_results.append(meta_result)

    return meta_results


def parse_token_spans(token_spans_str: str):
    if token_spans_str is None:
        return None
    try:
        return ast.literal_eval(token_spans_str)
    except Exception as e:
        raise ValueError(f"Failed to parse token_spans: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO with Learnable Prompt", add_help=True)
    parser.add_argument("--config_file", "-c", type=str,
                        default="./groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="Path to model config")
    parser.add_argument("--checkpoint_path", "-p", type=str,
                        default="./weights/groundingdino_swint_ogc.pth",
                        help="Path to checkpoint")
    parser.add_argument("--image_path", "-i", type=str,
                        default="./data/images",
                        help="Path to image file or directory")
    parser.add_argument("--label_path", "-l", type=str,
                        default="./data/labels/adversarial_patches.csv",
                        help="Path to label file or directory")
    parser.add_argument("--text_prompt", "-t", type=str,
                        default="adversarial patchs",
                        help="Text prompt")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="output",
                        help="Output directory")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--token_spans", type=str, default=None,
                        help="String representation of token span list, e.g.: '[[[2,5]], [[0,1],[2,5]]]'")
    parser.add_argument("--cpu-only", action="store_true")

    # Training parameters
    parser.add_argument("--train", action="store_true", help="Whether to train learnable prompts")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--learnable_token", type=str, default="sks", help="Learnable token text")

    # Batch processing parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--extensions", type=str,
                        default="jpg,jpeg,png,bmp,webp",
                        help="Allowed extensions for batch processing, comma separated")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursive search in directories")
    parser.add_argument("--suffix", type=str, default="_pred",
                        help="Suffix for prediction result filenames")
    parser.add_argument("--save-raw", action="store_true",
                        help="Save copy of raw image")
    parser.add_argument("--save-meta", action="store_true",
                        help="Save detection info JSON for each image (meta_results.json)")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Argument parsing
    config_file = args.config_file
    checkpoint_path = args.checkpoint_path
    input_path = args.image_path
    label_path = args.label_path
    text_prompt = args.text_prompt
    output_dir = Path(args.output_dir)
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = parse_token_spans(args.token_spans)
    cpu_only = args.cpu_only
    batch_size = args.batch_size
    extensions = [e.strip().lower() for e in args.extensions.split(",") if e.strip()]
    save_raw = args.save_raw
    num_workers = args.num_workers

    # Training parameters
    # Note: Corrected logic. If --train is passed, train_mode is True.
    train_mode = args.train
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    learnable_token = args.learnable_token

    if token_spans is not None:
        text_threshold = None
        print("Using token_spans mode, text_threshold set to None")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    base_model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)

    # If training mode, initialize learnable prompts
    if train_mode:
        print("Initializing learnable text prompts...")
        model, text_prompt = init_prompt(
            base_model,
            phrase=f"{text_prompt}",
            placeholder_token="sks"  # This arg is not strictly required in this impl
        )
        print(f"Using learnable prompt: '{text_prompt}'")
    else:
        model = base_model
        new_text_prompt = text_prompt

    # Process paths
    path_obj = Path(input_path)
    label_path_obj = Path(label_path) if label_path else None

    # Check if single image or directory
    if path_obj.is_file():
        print(f"Single image inference: {path_obj}")
        # Single image also uses batch flow but with batch size 1
        image_paths = [path_obj]
        dataset = ImageDataset(image_paths, label_dir=label_path)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Single image uses batch size 1
            shuffle=False,
            num_workers=1,
            collate_fn=custom_collate_fn
        )

        if train_mode:
            print("Warning: Single image mode is not recommended for training, performing inference instead")
            train_mode = False

        results = batch_inference(
            model, dataloader, new_text_prompt, box_threshold, text_threshold,
            token_spans=token_spans, cpu_only=cpu_only
        )

        meta_results = process_results(results, output_dir, save_raw, args.suffix)
        print(f"Done: {meta_results[0] if meta_results else 'No results'}")
    else:
        print(f"Directory batch processing: {path_obj}")
        # Collect all images
        image_paths = collect_images(str(path_obj), extensions, args.recursive)
        if not image_paths:
            print("No matching image files found, exiting.")
            sys.exit(0)
        print(f"Found {len(image_paths)} images, Batch size: {batch_size}")

        # Create Dataset and DataLoader
        dataset = GroundingDINODataset(
            input_path,
            label_path,
            transforms=None,
            negative_sampling_rate=0,
            add_extra_classes=False,
            min_scale=1,  # Min scale
            max_scale=1,  # Max scale
            fill_color=(0, 0, 0)  # Use black fill
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: tuple(zip(*x))
        )

        # Execute Training or Inference
        if train_mode:

            print(f"Starting learnable prompt training for {num_epochs} epochs...")
            results = train_prompt(
                model, dataloader, text_prompt, num_epochs=num_epochs, learning_rate=learning_rate, cpu_only=cpu_only,
                class_loss_coef=1, bbox_loss_coef=5, giou_loss_coef=2, eos_coef=0.1, max_txt_len=256, output_dir=str(output_dir)
            )
        else:
            print("Executing batch inference...")
            results = batch_inference(
                model, dataloader, new_text_prompt, box_threshold, text_threshold,
                token_spans=token_spans, cpu_only=cpu_only
            )

        # Process results
        meta_results = process_results(results, output_dir, save_raw, args.suffix)
        print(f"Done: Successfully processed {len(meta_results)} images")

        # Save metadata
        if args.save_meta and meta_results:
            import json

            meta_path = output_dir / "meta_results.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_results, f, ensure_ascii=False, indent=2)
            print(f"Detection info saved to: {meta_path}")
