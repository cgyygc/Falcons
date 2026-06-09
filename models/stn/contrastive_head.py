"""
Contrastive Head for Registration Network.

This module implements contrastive learning components that can be embedded
into the encoder of the registration network to learn better feature representations.

Reference:
    - SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
    - MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps features to a lower-dimensional space where contrastive loss is applied.

    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output projection dimension (typically 128 or 256)
        num_layers: Number of projection layers (typically 2 or 3)
    """

    def __init__(self, in_dim, hidden_dim=None, out_dim=128, num_layers=2):
        super(ProjectionHead, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim

        layers = []
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers (if more than 2 layers)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # Output layer (no batch norm or activation)
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W] or [B, N, C]

        Returns:
            Projected features [B, out_dim]
        """
        # Global average pooling
        if x.dim() == 4:  # [B, C, H, W]
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
        elif x.dim() == 3:  # [B, N, C]
            x = x.mean(dim=1)  # [B, C]

        return self.projection(x)


class ContrastiveEmbedding(nn.Module):
    """
    Contrastive embedding module that extracts features from encoder outputs
    and projects them for contrastive learning.

    This can be attached to multiple stages of the encoder.
    """

    def __init__(self, in_channels, hidden_dim=None, proj_dim=128, num_layers=2,
                 use_pool=True, pool_type='avg'):
        """
        Args:
            in_channels: Number of input feature channels
            hidden_dim: Hidden dimension for projection head
            proj_dim: Output projection dimension
            num_layers: Number of projection layers
            use_pool: Whether to use pooling before projection
            pool_type: Type of pooling ('avg' or 'max')
        """
        super(ContrastiveEmbedding, self).__init__()
        self.use_pool = use_pool
        self.pool_type = pool_type

        # Optional 1x1 conv to reduce dimension before pooling
        self.conv_reduce = nn.Conv2d(in_channels, in_channels // 2, 1) if use_pool else None

        # Projection head
        self.projection_head = ProjectionHead(
            in_dim=in_channels // 2 if use_pool else in_channels,
            hidden_dim=hidden_dim,
            out_dim=proj_dim,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Projected features [B, proj_dim]
        """
        if self.use_pool:
            x = self.conv_reduce(x)

        return self.projection_head(x)


class ContrastiveHead(nn.Module):
    """
    Main contrastive head that can be embedded into the encoder.

    This module collects features from multiple encoder stages and produces
    contrastive embeddings for each stage.
    """

    def __init__(self, encoder_channels, hidden_dims=None, proj_dim=128,
                 num_layers=2, num_stages=None):
        """
        Args:
            encoder_channels: List of channel dimensions for each encoder stage
            hidden_dims: Hidden dimensions for each stage (optional)
            proj_dim: Output projection dimension (same for all stages)
            num_layers: Number of projection layers
            num_stages: Number of encoder stages to attach contrastive heads to
                       (None means all stages)
        """
        super(ContrastiveHead, self).__init__()

        if num_stages is None:
            num_stages = len(encoder_channels)

        self.num_stages = num_stages
        self.proj_dim = proj_dim

        # Create contrastive embedding for each stage
        self.embeddings = nn.ModuleList()
        for i in range(min(num_stages, len(encoder_channels))):
            hidden_dim = hidden_dims[i] if hidden_dims is not None else None
            self.embeddings.append(
                ContrastiveEmbedding(
                    in_channels=encoder_channels[i],
                    hidden_dim=hidden_dim,
                    proj_dim=proj_dim,
                    num_layers=num_layers,
                    use_pool=True,
                    pool_type='avg'
                )
            )

        # Final projection that combines all stage embeddings
        self.final_projection = nn.Sequential(
            nn.Linear(proj_dim * num_stages, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of feature maps from encoder stages [(B, C1, H1, W1), ...]

        Returns:
            contrastive_features: Dictionary containing:
                - 'stage_features': List of per-stage projected features
                - 'combined_feature': Combined feature from all stages
        """
        stage_features = []

        for i, feat in enumerate(encoder_features[:self.num_stages]):
            proj_feat = self.embeddings[i](feat)  # [B, proj_dim]
            stage_features.append(proj_feat)

        # Combine all stage features
        combined = torch.cat(stage_features, dim=1)  # [B, proj_dim * num_stages]
        combined = self.final_projection(combined)  # [B, proj_dim]

        # L2 normalize
        combined = F.normalize(combined, dim=-1)
        stage_features = [F.normalize(f, dim=-1) for f in stage_features]

        return {
            'stage_features': stage_features,
            'combined_feature': combined
        }


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.

    This is the standard contrastive loss used in SimCLR and similar methods.
    """

    def __init__(self, temperature=0.07, use_cosine_similarity=True):
        """
        Args:
            temperature: Temperature parameter for softmax
            use_cosine_similarity: Whether to use cosine similarity (True) or
                                  dot product (False)
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, query, positive_key, negative_keys=None):
        """
        Args:
            query: Query features [B, D] (typically from source image)
            positive_key: Positive key features [B, D] (from target image)
            negative_keys: Negative key features [B, N, D] (optional, if None uses
                          other samples in batch as negatives)

        Returns:
            loss: Contrastive loss scalar
        """
        batch_size = query.shape[0]

        # Normalize features
        if self.use_cosine_similarity:
            query = F.normalize(query, dim=-1)
            positive_key = F.normalize(positive_key, dim=-1)

        # Compute similarity matrix
        # query: [B, D] -> [B, 1, D]
        # keys: [B, D] -> [1, B, D]
        # similarity: [B, B] where similarity[i, j] is sim(query[i], key[j])
        similarity = torch.matmul(query.unsqueeze(1), positive_key.unsqueeze(0).transpose(-2, -1)).squeeze(1)
        similarity = similarity / self.temperature

        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=query.device)

        # Use cross entropy (equivalent to InfoNCE)
        loss = self.cross_entropy(similarity, labels)

        return loss


class ByolLoss(nn.Module):
    """
    BYOL (Bootstrap Your Own Latent) loss for contrastive learning.

    BYOL doesn't use negative pairs and instead predicts the target
    representation from an online encoder using a momentum target encoder.
    """

    def __init__(self):
        super(ByolLoss, self).__init__()

    def forward(self, prediction, target):
        """
        Args:
            prediction: Predicted features [B, D]
            target: Target features [B, D] (normalized)

        Returns:
            loss: BYOL loss scalar
        """
        # Normalize both
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target, dim=-1)

        # Mean squared error between normalized vectors
        # This is equivalent to 2 - 2 * cos_similarity
        loss = 2 - 2 * (prediction * target).sum(dim=-1).mean()

        return loss


class ContrastiveLearningWrapper(nn.Module):
    """
    Wrapper module that combines encoder with contrastive head.

    This makes it easy to add contrastive learning to existing encoders.
    """

    def __init__(self, encoder, contrastive_head):
        """
        Args:
            encoder: The encoder network (e.g., UKAN_Backbone, ResUnet)
            contrastive_head: The contrastive head module
        """
        super(ContrastiveLearningWrapper, self).__init__()
        self.encoder = encoder
        self.contrastive_head = contrastive_head

    def forward(self, x_a, x_b, return_features=False):
        """
        Forward pass with contrastive features.

        Args:
            x_a: Source image [B, C, H, W]
            x_b: Target image [B, C, H, W]
            return_features: Whether to return intermediate features

        Returns:
            If return_features=False: Returns encoder output (deformation field)
            If return_features=True: Returns encoder output and contrastive features
        """
        # Get encoder output (deformation field)
        deformation = self.encoder(x_a, x_b)

        if return_features:
            # Get contrastive features from encoder intermediate outputs
            # This requires the encoder to have a method to extract features
            if hasattr(self.encoder, 'get_encoder_features'):
                features_a = self.encoder.get_encoder_features(x_a)
                features_b = self.encoder.get_encoder_features(x_b)

                contrastive_a = self.contrastive_head(features_a)
                contrastive_b = self.contrastive_head(features_b)

                return deformation, (contrastive_a, contrastive_b)
            else:
                return deformation, None

        return deformation


def get_encoder_channels(stn_type, cfg='A'):
    """
    Helper function to get encoder channel dimensions for different STN types.

    Args:
        stn_type: Type of STN ('unet', 'ukan')
        cfg: Configuration name

    Returns:
        List of channel dimensions for each encoder stage
    """
    if stn_type == 'unet':
        from .unet_stn import ndf
        return ndf[cfg]
    elif stn_type == 'ukan':
        # UKAN encoder channels: [8, 16, 64, 128, 256]
        return [8, 16, 64, 128]
    else:
        raise ValueError(f"Unknown STN type: {stn_type}")
