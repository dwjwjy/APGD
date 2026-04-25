# import os
# import csv
# import random
# import torch
# from collections import defaultdict
# from torch.utils.data import Dataset
# from groundingdino.util.train import load_image
# from groundingdino.util.vl_utils import build_captions_and_token_span
#
# class GroundingDINODataset(Dataset):
#     def __init__(self, img_dir, ann_file, transforms=None, negative_sampling_rate=None, add_extra_classes=False):
#         """
#         Args:
#             img_dir (str): Path to image directory
#             ann_file (str): Path to annotation CSV
#             transforms: Optional transform to be applied
#             negative_sampling_rate (float): Rate of negative samples to include in captions (0.0 to 1.0)
#
#         """
#         self.img_dir = img_dir
#         self.transforms = transforms
#         self.negative_sampling_rate = negative_sampling_rate
#         self.annotations = self.read_dataset(img_dir, ann_file)
#         self.image_paths = list(self.annotations.keys())
#         # Collect all unique categories for negative sampling
#         self.all_categories = set()
#         for img_path in self.annotations:
#             self.all_categories.update(self.annotations[img_path]['phrases'])
#         self.all_categories = list(self.all_categories)
#         if add_extra_classes:
#             extra_classes = ['person', 'cat', 'dog']
#             self.all_categories.extend(extra_classes)
#
#     def read_dataset(self, img_dir, ann_file):
#         """
#         Read dataset annotations and convert to [x,y,w,h] format
#         """
#         ann_dict = defaultdict(lambda: defaultdict(list))
#         with open(ann_file) as file_obj:
#             ann_reader = csv.DictReader(file_obj)
#             for row in ann_reader:
#                 img_path = os.path.join(img_dir, row['image_name'])
#                 # Store in [x,y,w,h] format directly
#                 x = float(row['bbox_x'])
#                 y = float(row['bbox_y'])
#                 w = float(row['bbox_width'])
#                 h = float(row['bbox_height'])
#
#                 # Convert to center format [cx,cy,w,h]
#                 cx = x
#                 cy = y
#                 ann_dict[img_path]['boxes'].append([cx, cy, w, h])
#                 ann_dict[img_path]['phrases'].append(row['label_name'])
#         return ann_dict
#
#     def sample_negative_categories(self, positive_categories, num_negative=None):
#         """
#         Sample negative categories that are not present in the current image
#         """
#         if num_negative is None:
#             # Default to same number as positive categories or 1, whichever is larger
#             num_negative = max(1, len(positive_categories))
#
#         # Get candidates that are not in positive categories
#         candidates = [cat for cat in self.all_categories if cat not in positive_categories]
#
#         # Handle case where there are no candidates
#         if not candidates:
#             # If no candidates, return empty list or duplicate some categories
#             return []
#
#         # Sample negative categories
#         if len(candidates) >= num_negative:
#             negative_categories = random.sample(candidates, num_negative)
#         else:
#             # If not enough candidates, sample with replacement
#             negative_categories = random.choices(candidates, k=num_negative)
#
#         return negative_categories
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         # Load and transform image
#         image_source, image = load_image(img_path)
#         h, w = image_source.shape[0:2]
#         print(h,w)
#         boxes = torch.tensor(self.annotations[img_path]['boxes'], dtype=torch.float32)
#         str_cls_lst = self.annotations[img_path]['phrases'] # ['adversarial patch'],.....,['adversarial patch']
#
#         # Sample negative categories if needed
#         if self.negative_sampling_rate > 0 and len(self.all_categories) > len(str_cls_lst):
#             # print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
#             # Determine number of negative samples based on rate
#             num_negative = max(1, int(len(str_cls_lst) * self.negative_sampling_rate))
#             negative_categories = self.sample_negative_categories(str_cls_lst, num_negative)
#             # Combine positive and negative categories
#             combined_categories = str_cls_lst + negative_categories
#         else:
#             # print("nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
#             combined_categories = str_cls_lst
#
#         # Create caption mapping and format
#         caption_dict = {item: idx for idx, item in enumerate(str_cls_lst)}  # Only for positive categories
#         # print(caption_dict,"++++++++++++++++++++++++++++++++++++++++++++++++")
#         captions, cat2tokenspan = build_captions_and_token_span(combined_categories, force_lowercase=True)
#         # print(cat2tokenspan,"sssssssssssssssssssssssssssssssssssssssssssss")
#         # Labels for positive categories only
#         classes = torch.tensor([caption_dict[p] for p in str_cls_lst], dtype=torch.int64)
#         # print(classes,"---------------------------------")
#
#         target = {
#             'boxes': boxes,  # Already in [cx,cy,w,h] format
#             'size': torch.as_tensor([int(h), int(w)]),
#             'orig_img': image_source,
#             'str_cls_lst': str_cls_lst,  # Positive categories only
#             'all_categories': combined_categories,  # Positive + negative categories
#             'caption': captions,
#             'labels': classes,
#             'cat2tokenspan': cat2tokenspan
#         }
#
#         return image, target

import os
import csv
import random
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from groundingdino.util.train import load_image
from groundingdino.util.vl_utils import build_captions_and_token_span
from PIL import Image
import torchvision.transforms.functional as F
import groundingdino.datasets.transforms as T

class GroundingDINODataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None, negative_sampling_rate=None, add_extra_classes=False,
                 target_size=(416, 416), min_scale=0.5, max_scale=0.9, fill_color=(128, 128, 128)):
        """
        Args:
            img_dir (str): Path to image directory
            ann_file (str): Path to annotation CSV
            transforms: Optional transform to be applied
            negative_sampling_rate (float): Rate of negative samples to include in captions (0.0 to 1.0)
            target_size (tuple): Target size for images (width, height)
            min_scale (float): Minimum scale factor for random resizing
            max_scale (float): Maximum scale factor for random resizing
            fill_color (tuple): RGB color to fill the padding area
        """
        self.img_dir = img_dir
        self.transforms = transforms
        self.negative_sampling_rate = negative_sampling_rate
        self.annotations = self.read_dataset(img_dir, ann_file)
        self.image_paths = list(self.annotations.keys())
        self.target_size = target_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.fill_color = fill_color

        # Collect all unique categories for negative sampling
        self.all_categories = set()
        for img_path in self.annotations:
            self.all_categories.update(self.annotations[img_path]['phrases'])
        self.all_categories = list(self.all_categories)
        if add_extra_classes:
            extra_classes = ['person', 'cat', 'dog']
            self.all_categories.extend(extra_classes)

    def read_dataset(self, img_dir, ann_file):
        """
        Read dataset annotations and convert to [x,y,w,h] format
        """
        ann_dict = defaultdict(lambda: defaultdict(list))
        with open(ann_file) as file_obj:
            ann_reader = csv.DictReader(file_obj)
            for row in ann_reader:
                img_path = os.path.join(img_dir, row['image_name'])
                # Store in [x,y,w,h] format directly
                x = float(row['bbox_x'])
                y = float(row['bbox_y'])
                w = float(row['bbox_width'])
                h = float(row['bbox_height'])

                # Convert to center format [cx,cy,w,h]
                cx = x
                cy = y
                ann_dict[img_path]['boxes'].append([cx, cy, w, h])
                ann_dict[img_path]['phrases'].append(row['label_name'])
        return ann_dict

    def sample_negative_categories(self, positive_categories, num_negative=None):
        """
        Sample negative categories that are not present in the current image
        """
        if num_negative is None:
            # Default to same number as positive categories or 1, whichever is larger
            num_negative = max(1, len(positive_categories))

        # Get candidates that are not in positive categories
        candidates = [cat for cat in self.all_categories if cat not in positive_categories]

        # Handle case where there are no candidates
        if not candidates:
            return []

        # Sample negative categories
        if len(candidates) >= num_negative:
            negative_categories = random.sample(candidates, num_negative)
        else:
            # If not enough candidates, sample with replacement
            negative_categories = random.choices(candidates, k=num_negative)

        return negative_categories

    def random_resize_with_padding(self, image):
        """
        Randomly resize the image to a smaller size and pad with fill_color to target_size
        Returns:
            - Processed image
            - Scale factor
            - Padding offsets (left, top)
        """
        # Convert to PIL Image if it's not already
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image))

        orig_w, orig_h = image.size

        # Choose a random scale factor between min_scale and max_scale
        scale_factor = random.uniform(self.min_scale, self.max_scale)

        # Calculate new dimensions
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)

        # Resize the image
        resized_img = image.resize((new_w, new_h), Image.BILINEAR)

        # Create a new image with the target size and fill it with fill_color
        target_w, target_h = self.target_size
        padded_img = Image.new("RGB", (target_w, target_h), self.fill_color)
        # Calculate padding offsets (centered)
        left = (target_w - new_w) // 2
        top = (target_h - new_h) // 2

        # Paste the resized image onto the padded image
        padded_img.paste(resized_img, (left, top))

        return padded_img, scale_factor, (left, top)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        image_pil = Image.open(img_path).convert("RGB")
        image_source = np.asarray(image_pil)

        # Apply random resize with padding
        gate = torch.tensor([0]).bernoulli_(0.5)
        if gate.item() == 0:
            processed_image = image_pil
            scale_factor = 1.0
            pad_left, pad_top = 0, 0
        else:
            processed_image, scale_factor, (pad_left, pad_top) = self.random_resize_with_padding(image_pil)

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        image, _ = transform(processed_image, None)

        # Get original boxes in [cx, cy, w, h] format
        orig_boxes = self.annotations[img_path]['boxes']

        # Adjust boxes according to scaling and padding
        adjusted_boxes = []
        for cx, cy, w, h in orig_boxes:
            # Scale the coordinates and dimensions
            new_cx = cx * scale_factor + pad_left
            new_cy = cy * scale_factor + pad_top
            new_w = w * scale_factor
            new_h = h * scale_factor

            adjusted_boxes.append([new_cx, new_cy, new_w, new_h])

        boxes = torch.tensor(adjusted_boxes, dtype=torch.float32)
        str_cls_lst = self.annotations[img_path]['phrases']
        # print(str_cls_lst,"qqqqqqqqqqqqqqqqqqqqqqqqq")

        # Get dimensions of processed image
        h, w = image_source.shape[0:2]


        # Sample negative categories if needed
        if self.negative_sampling_rate > 0 and len(self.all_categories) > len(str_cls_lst):
            num_negative = max(1, int(len(str_cls_lst) * self.negative_sampling_rate))
            negative_categories = self.sample_negative_categories(str_cls_lst, num_negative)
            combined_categories = str_cls_lst + negative_categories
        else:
            combined_categories = str_cls_lst
        # print(combined_categories,"xxxxxxxxxxxxxxxxxxxxxxxxxx")
        # Create caption mapping and format
        caption_dict = {item: idx for idx, item in enumerate(str_cls_lst)}  # Only for positive categories
        captions, cat2tokenspan = build_captions_and_token_span(combined_categories, force_lowercase=True)

        # print(cat2tokenspan,"ssssssssssssssssssssssssssssss")
        # Labels for positive categories only
        classes = torch.tensor([caption_dict[p] for p in str_cls_lst], dtype=torch.int64)

        target = {
            'boxes': boxes,  # Adjusted boxes in [cx,cy,w,h] format
            'size': torch.as_tensor([int(h), int(w)]),
            'orig_img': image_source,
            'str_cls_lst': str_cls_lst,  # Positive categories only
            'all_categories': combined_categories,  # Positive + negative categories
            'caption': captions,
            'labels': classes,
            'cat2tokenspan': cat2tokenspan,
            'scale_factor': scale_factor,
            'padding': (pad_left, pad_top)
        }

        return image, target
