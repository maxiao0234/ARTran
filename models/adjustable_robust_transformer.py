import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer


class AnisotropicPatchEmbedding(nn.Module):
    """Base class for anisotropic patch embedding (APE).

    Args:
        hw_shape: shape of input image
        kernel_size: patch size
        stride: stride of patch embeddings
        in_chans: number of input channels
        embed_dim: embedding dimension
        norm_layer: (nn.Module): normalization layer
    """
    def __init__(self, hw_shape=(224, 224), kernel_size=(56, 8), stride=None,
                 in_chans=1, embed_dim=192, norm_layer=None):
        super().__init__()

        self.hw_shape = hw_shape
        num_tokens_h = (hw_shape[0] - kernel_size[0]) // stride[0] + 1
        num_tokens_w = (hw_shape[1] - kernel_size[1]) // stride[1] + 1
        self.num_tokens = num_tokens_h * num_tokens_w
        if stride is None:
            stride = kernel_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert ((self.hw_shape[0], self.hw_shape[1]) == (H, W))
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class AdjustableClassEmbedding(nn.Module):
    """Base class for adjustable class embedding (ACE).
    This block contains two parameterized vectors. The
    ACE constructs a linear combination with a given
    shift and the vectors to adapt to variable conditions.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.parameter_vectors = nn.Parameter(torch.zeros(1, 2, embed_dim))
        trunc_normal_(self.parameter_vectors, std=.02)

    def forward(self, shift=None):
        if shift is None:
            return torch.mean(self.parameter_vectors, dim=1, keepdim=True)
        else:
            s_ = torch.cat([(1 + shift) / 2, (1 - shift) / 2], dim=-1)
            class_embed_shift = torch.einsum('b e, n e c -> b n c', s_, self.parameter_vectors)
            return class_embed_shift


class AdjustableRobustTransformer(VisionTransformer):
    """The adjustable robust transformer (ARTran) based on ViT.

    Input: A image (x) and an adjustment coefficient (shift)
    Output: volume loss and clean and noisy posteriors

    Args:
        hw_shape: shape of input image
        kernel_size: patch size
        stride: stride of patch embeddings
        in_chans: number of input channels
        embed_dim: embedding dimension
        depth: depth of transformer
        num_heads: number of attention heads
        num_classes: number of classes for classification head
    TODO: multi classification
    """
    def __init__(self, hw_shape=(224, 224), kernel_size=(56, 8), stride=(28, 8), in_chans=3,
                 embed_dim=384, depth=12, num_heads=6, num_classes=2, **kwargs):
        assert num_classes == 2
        super().__init__(num_classes=num_classes, in_chans=in_chans, embed_dim=embed_dim, depth=depth,
                         num_heads=num_heads, **kwargs)

        self.patch_embed = AnisotropicPatchEmbedding(hw_shape=hw_shape, kernel_size=kernel_size,
                                                     stride=stride, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_tokens

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cls_token = None
        self.adjustable_class_embedding = AdjustableClassEmbedding(embed_dim)

        self.register_buffer('diagonal_mask', torch.eye(num_classes))
        self.register_buffer('offset_benchmark', torch.zeros((1, 1, 1)))
        self.diagonal_class = nn.Parameter(torch.ones((1, 1, num_classes)))
        self.diagonal_shift = nn.Parameter(-torch.ones((1, 1, 1)))

    def transition(self, post_clean, diagonal_benchmark, adjustment):
        diagonal_mask = self.diagonal_mask.expand(post_clean.shape[0], -1, -1)
        diagonal = diagonal_benchmark + adjustment.expand(-1, -1, 2).transpose(1, 2)
        transition_matrix = torch.where(diagonal_mask > 0, diagonal, 1 - diagonal)
        post_noise = torch.einsum('b n c, b c -> b n', transition_matrix, post_clean)
        return post_noise

    def forward(self, x, shift):
        class_embed_benchmark = self.adjustable_class_embedding().expand(x.shape[0], -1, -1)
        class_embed_shift = self.adjustable_class_embedding(shift)

        x = self.patch_embed(x)
        x = torch.cat((class_embed_benchmark, class_embed_shift, x + self.pos_embed), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.fc_norm(x)

        s_i = torch.sigmoid(self.diagonal_class)
        s_0 = torch.sigmoid(self.diagonal_shift)

        diagonal_benchmark = 0.5 + 0.5 * s_0 * s_i + 0.5 * (0.5 - 0.5 * s_0)
        diagonal_benchmark = diagonal_benchmark.expand(x.shape[0], 2, -1)
        volume_loss = torch.log(torch.sum(1 + 0.5 * s_0 * (s_i - 1)) - 1)

        post_clean_benchmark = torch.softmax(self.head(x[:, 0, :]), dim=-1)
        offset_benchmark = self.offset_benchmark.expand(x.shape[0], -1, -1)
        post_noise_benchmark = self.transition(post_clean_benchmark, diagonal_benchmark, offset_benchmark)

        post_clean_shift = torch.softmax(self.head(x[:, 1, :]), dim=-1)
        offset_shift = torch.stack([-shift, shift], dim=1).expand(-1, 2, -1) * 0.5 * (0.5 - 0.5 * s_0)
        post_noise_shift = self.transition(post_clean_shift, diagonal_benchmark, offset_shift)

        outputs = {
            'volume': volume_loss,
            'post_clean_benchmark': post_clean_benchmark,
            'post_noise_benchmark': post_noise_benchmark,
            'post_clean_shift': post_clean_shift,
            'post_noise_shift': post_noise_shift
        }

        return outputs


if __name__ == "__main__":
    x = torch.rand((3, 3, 224, 224))

    model = AdjustableRobustTransformer(
        hw_shape=(224, 224),
        kernel_size=(56, 8),
        stride=(28, 8)
    )

    shift = torch.tensor([[0.], [0.5], [0.75]])
    out = model(x, shift)
    print(out['post_clean_shift'])
