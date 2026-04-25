import os
import random
from typing import List

import torch


def create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Input:
        - tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        - token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float) # token_span is the number of patch

    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            # print(beg,"aaaaaaaaa", end,"bbbbbbbbb")
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)

            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            if os.environ.get("SHILONG_DEBUG_ONLY_ONE_POS", None) == "TRUE":
                positive_map[j, beg_pos] = 1
                break
            else:  # yes
                positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map


def build_captions_and_token_span(cat_list, force_lowercase):
    """
    Return:
        captions: str
        cat2tokenspan: dict
            {
                'dog': [[0, 2]],
                ...
            }
    """

    cat2tokenspan = {}
    captions = ""
    for catname in cat_list:

        class_name = catname
        if force_lowercase:
            class_name = class_name.lower()
        if "/" in class_name:
            class_name_list: List = class_name.strip().split("/")
            class_name_list.append(class_name)
            class_name: str = random.choice(class_name_list)

        tokens_positive_i = []
        subnamelist = [i.strip() for i in class_name.strip().split(" ")]

        for subname in subnamelist:
            if len(subname) == 0:
                continue

            strat_idx = len(captions)
            end_idx = strat_idx + len(subname)

            tokens_positive_i.append([strat_idx, end_idx])
            captions = captions + subname

        if len(tokens_positive_i) > 0:

            if class_name == "adversarial patch":
                cat2tokenspan[class_name] = [[0, 11], [12, 17]]
            elif class_name == "adversarial patchs":
                cat2tokenspan[class_name] = [[0, 11], [12, 18]]
            elif class_name == "adversarial patches":
                cat2tokenspan[class_name] = [[0, 11], [12, 19]]
            elif class_name == "adversarial sticker":
                cat2tokenspan[class_name] = [[0, 11], [12, 19]]
            elif class_name == "adversarial noise":
                cat2tokenspan[class_name] = [[0, 11], [12, 17]]
            elif class_name == "adversarial perturbation":
                cat2tokenspan[class_name] = [[0, 11], [12, 24]]
            elif class_name == "perturbation block":
                cat2tokenspan[class_name] = [[0, 12], [13, 18]]

    return captions, cat2tokenspan


def build_id2posspan_and_caption(category_dict: dict):
    """Build id2pos_span and caption from category_dict

    Args:
        category_dict (dict): category_dict
    """
    cat_list = [item["name"].lower() for item in category_dict]
    id2catname = {item["id"]: item["name"].lower() for item in category_dict}
    caption, cat2posspan = build_captions_and_token_span(cat_list, force_lowercase=True)
    id2posspan = {catid: cat2posspan[catname] for catid, catname in id2catname.items()}
    return id2posspan, caption
