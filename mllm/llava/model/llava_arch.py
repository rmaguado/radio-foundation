#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import logging

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from mllm.llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

logger = logging.getLogger("DeepSpeed")


class LlavaMetaModel:

    def __init__(self, config):
        super().__init__(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_mm_projector(self):
        mm_projector = getattr(self, "mm_projector", None)
        if type(mm_projector) is list:
            mm_projector = mm_projector[0]
        return mm_projector

    def initialize_vision_modules(self, model_args, torch_dtype):
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        checkpoint_path = model_args.pretrain_checkpoint_path

        self.config.mm_vision_tower = model_args.vision_tower
        self.config.image_tokens = model_args.image_tokens

        vision_tower = build_vision_tower(model_args, torch_dtype=torch_dtype)
        vision_tower.requires_grad_(False)

        self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "attn_pool"
        )
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.mm_projector = build_vision_projector(self.config)


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        if images is None:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        image_features = self.encode_images(images)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels_dtype = labels.dtype

        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []

        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_labels = labels[batch_idx]
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum().item()

            assert (
                num_images == 1
            ), f"Only one image token is allowed per input. {cur_input_ids}"

            image_token_index = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[
                0
            ].item()

            # Split around the image token
            before_ids = cur_input_ids[:image_token_index]
            after_ids = cur_input_ids[image_token_index + 1 :]

            before_labels = cur_labels[:image_token_index]
            after_labels = cur_labels[image_token_index + 1 :]

            before_embed = self.get_model().embed_tokens(before_ids.to(self.device))
            after_embed = self.get_model().embed_tokens(after_ids.to(self.device))
            cur_image_features = image_features[batch_idx]

            cur_input_embeds = torch.cat(
                [before_embed, cur_image_features, after_embed], dim=0
            )
            cur_label_ids = torch.cat(
                [
                    before_labels,
                    torch.full(
                        (cur_image_features.shape[0],),
                        IGNORE_INDEX,
                        device=self.device,
                        dtype=cur_labels.dtype,
                    ),
                    after_labels,
                ],
                dim=0,
            )

            new_input_embeds.append(cur_input_embeds)
            new_labels.append(cur_label_ids)

        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=self.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=self.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=self.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=self.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0).to(self.device)

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels_padded,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if "Llama-3" in model_args.model_name_or_path:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            tokenizer.pad_token_id = 128004
        else:
            tokenizer.pad_token = tokenizer.unk_token
