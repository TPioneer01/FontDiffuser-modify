import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)

class StructureConditionAdapter(nn.Module):
    """Lightweight adapter that converts structure maps into multi-scale features."""

    def __init__(self, in_channels: int, hidden_dim: int, num_pyramid_levels: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(),
        )
        self.down_layers = nn.ModuleList()
        for _ in range(max(num_pyramid_levels - 1, 0)):
            self.down_layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
                    nn.SiLU(),
                )
            )

    def forward(self, x: torch.Tensor):
        features = []
        h = self.stem(x)
        features.append(h)
        for layer in self.down_layers:
            h = layer(h)
            features.append(h)

        tokens = features[-1].flatten(2).transpose(1, 2)
        return features, tokens


class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.structure_adapter = None
        self.structure_feature_adapters = nn.ModuleDict()
        self.structure_feature_keys = []
        if hasattr(unet.config, "structure_feature_keys"):
            self.structure_feature_keys = list(unet.config.structure_feature_keys)
        elif isinstance(unet.config, dict) and "structure_feature_keys" in unet.config:
            self.structure_feature_keys = list(unet.config["structure_feature_keys"])
        self.structure_adapter = None
        self.structure_feature_adapters = nn.ModuleDict()
        self.structure_feature_keys = []
        if hasattr(unet.config, "structure_feature_keys"):
            self.structure_feature_keys = list(unet.config.structure_feature_keys)
        elif isinstance(unet.config, dict) and "structure_feature_keys" in unet.config:
            self.structure_feature_keys = list(unet.config["structure_feature_keys"])
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        style_images,
        content_images,
        content_encoder_downsample_size,
        structure_features: dict = None,
    ):
        style_img_feature, _, _ = self.style_encoder(style_images)
    
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
    
        # Get the content feature
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        if structure_features is None:
            structure_features = {}

        if structure_features:
            structure_tensors = []
            for key in self.structure_feature_keys:
                if key in structure_features:
                    structure_tensors.append(structure_features[key].to(x_t.device))
            if structure_tensors:
                structure_tensor = torch.cat(structure_tensors, dim=1)
                if self.structure_adapter is None:
                    token_dim = getattr(self.unet.config, "structure_token_dim", 128)
                    pyramid_levels = content_encoder_downsample_size + 1
                    self.structure_adapter = StructureConditionAdapter(
                        in_channels=structure_tensor.shape[1],
                        hidden_dim=token_dim,
                        num_pyramid_levels=pyramid_levels,
                    ).to(structure_tensor.device)
                structure_feats, structure_tokens = self.structure_adapter(structure_tensor)
                style_hidden_states = torch.cat([style_hidden_states, structure_tokens], dim=1)

                for level, style_feature in enumerate(style_content_res_features):
                    adapter_key = f"level_{level}"
                    resized = F.interpolate(
                        structure_tensor,
                        size=style_feature.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    if adapter_key not in self.structure_feature_adapters:
                        adapter = nn.Sequential(
                            nn.Conv2d(resized.shape[1], style_feature.shape[1], kernel_size=3, padding=1),
                            nn.SiLU(),
                        ).to(resized.device)
                        self.structure_feature_adapters[adapter_key] = adapter
                    style_content_res_features[level] = style_content_res_features[level] + self.structure_feature_adapters[adapter_key](resized)

        input_hidden_states = [style_img_feature, content_residual_features,
                               style_hidden_states, style_content_res_features]

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]
        aux_outputs = out[2] if len(out) > 2 else {}

        return noise_pred, offset_out_sum, aux_outputs


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """
    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        cond,
        content_encoder_downsample_size,
        version,
        structure_features: dict = None,
        head_weights: dict = None,
    ):
        content_images = cond[0]
        style_images = cond[1]
        style_img_feature, _, style_residual_features = self.style_encoder(style_images)
        
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        
        # Get content feature
        content_img_feture, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feture)
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, style_hidden_states, style_content_res_features]

        structure_token_list = []
        if structure_features is None:
            structure_features = {}

        config_keys = self.structure_feature_keys if self.structure_feature_keys else []

        if structure_features:
            structure_tensors = []
            for key in config_keys:
                if key in structure_features:
                    tensor = structure_features[key].to(x_t.device)
                    if tensor.shape[0] != x_t.shape[0]:
                        if tensor.shape[0] * 2 == x_t.shape[0]:
                            pad = torch.zeros_like(tensor)
                            tensor = torch.cat([pad, tensor], dim=0)
                        else:
                            tensor = tensor.repeat(int(x_t.shape[0] / tensor.shape[0]), 1, 1, 1)
                    structure_tensors.append(tensor)
            if structure_tensors:
                structure_tensor = torch.cat(structure_tensors, dim=1)
                if self.structure_adapter is None:
                    token_dim = getattr(self.unet.config, "structure_token_dim", 128)
                    pyramid_levels = content_encoder_downsample_size + 1
                    self.structure_adapter = StructureConditionAdapter(
                        in_channels=structure_tensor.shape[1],
                        hidden_dim=token_dim,
                        num_pyramid_levels=pyramid_levels,
                    ).to(structure_tensor.device)
                structure_feats, structure_tokens = self.structure_adapter(structure_tensor)
                structure_token_list = [structure_tensor, structure_tokens, structure_feats]
                style_hidden_states = torch.cat([style_hidden_states, structure_tokens], dim=1)
                for level, style_feature in enumerate(style_content_res_features):
                    adapter_key = f"level_{level}"
                    resized = F.interpolate(
                        structure_tensor,
                        size=style_feature.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    if adapter_key not in self.structure_feature_adapters:
                        adapter = nn.Sequential(
                            nn.Conv2d(resized.shape[1], style_feature.shape[1], kernel_size=3, padding=1),
                            nn.SiLU(),
                        ).to(resized.device)
                        self.structure_feature_adapters[adapter_key] = adapter
                    style_content_res_features[level] = style_content_res_features[level] + self.structure_feature_adapters[adapter_key](resized)

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        aux_outputs = out[2] if len(out) > 2 else {}

        if head_weights is None:
            head_weights = {}

        if aux_outputs:
            weights = self._resolve_head_weights(timesteps, head_weights)
            for name, weight in weights.items():
                if name in aux_outputs:
                    noise_pred = noise_pred + weight * aux_outputs[name]

        if structure_token_list and "structure" in aux_outputs:
            structure_tensor = structure_token_list[0]
            self._apply_structure_feedback(
                noise_pred=noise_pred,
                aux_structure=aux_outputs["structure"],
                structure_targets=structure_tensor,
                interval=head_weights.get("feedback_interval"),
                eta=head_weights.get("feedback_eta"),
            )

        return noise_pred

    def _resolve_head_weights(self, timesteps, schedule_dict):
        if not schedule_dict:
            return {}

        weights = {}
        for name, schedule in schedule_dict.items():
            if name in {"feedback_interval", "feedback_eta"}:
                continue
            if isinstance(schedule, dict):
                start = float(schedule.get("start", 0.0))
                end = float(schedule.get("end", 0.0))
                t = timesteps.float().view(-1, 1, 1, 1)
                t = torch.clamp(t, 0.0, 1.0)
                weight = start * t + end * (1 - t)
            else:
                weight = torch.full_like(timesteps.float(), float(schedule)).view(-1, 1, 1, 1)
            if weight.dim() == 1:
                weight = weight.view(-1, 1, 1, 1)
            weights[name] = weight
        return weights

    def _apply_structure_feedback(self, noise_pred, aux_structure, structure_targets, interval=None, eta=None):
        if interval is None or eta is None:
            return
        if not hasattr(self, "_feedback_counter"):
            self._feedback_counter = 0
        self._feedback_counter += 1
        if self._feedback_counter % max(int(interval), 1) != 0:
            return

        if aux_structure.shape[0] != noise_pred.shape[0]:
            return
        half = aux_structure.shape[0] // 2
        if half == 0:
            return
        structure_target = structure_targets
        if structure_target.shape[0] != aux_structure.shape[0]:
            structure_target = structure_target.expand(aux_structure.shape[0], -1, -1, -1)
        structure_target = structure_target.to(aux_structure.device)
        feedback = torch.tanh(aux_structure[half:]) - torch.sigmoid(structure_target[half:])
        noise_pred[half:] = noise_pred[half:] - float(eta) * feedback
