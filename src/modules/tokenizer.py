from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

#from src.models.attention import RMSGroupNorm
from models.attention import RMSGroupNorm

class Tokenizer(nn.Module, ABC):
    
    @abstractmethod
    def encode(self, x: torch.Tensor, state_labels: torch.Tensor | None) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, x: torch.Tensor, state_labels: torch.Tensor | None) -> torch.Tensor:
        pass


class FoldUnFold(Tokenizer):

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def encode(self, x: torch.Tensor, state_labels: torch.Tensor | None) -> torch.Tensor:
        """ x of shape b t c s """
        return rearrange(x, "b t c (q p) -> b t (c p) q", p=self.patch_size)
        # return x.view(x.size(0), x.size(1), self.patch_size, -1)

    def decode(self, x: torch.Tensor, state_labels: torch.Tensor | None) -> torch.Tensor:
        """ x of shape b t c patch_size """
        return rearrange(x, "b t (c p) q -> b t c (q p)", p=self.patch_size)
        # return x.reshape(x.size(0), x.size(1), 1, -1)


def conv_module(ndim, transpose):
    if ndim == 1:
        return nn.Conv1d if not transpose else nn.ConvTranspose1d
    elif ndim == 2:
        return nn.Conv2d if not transpose else nn.ConvTranspose2d
    elif ndim == 3:
        return nn.Conv3d if not transpose else nn.ConvTranspose3d
    raise ValueError("ndim should be 1, 2 or 3.")


class Downsample(nn.Module):
    """ Subsample the data tensor """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode, bias, ndim):
        super().__init__()
        self.conv = conv_module(ndim, False)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode, bias, ndim):
        super().__init__()
        self.convT = conv_module(ndim, True)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        return self.convT(x)


class SubsampledLinear(nn.Module):
    """
    Cross between a linear layer and EmbeddingBag - takes in input 
    and list of indices denoting which state variables from the state
    vocab are present and only performs the linear layer on rows/cols relevant
    to those state variables
    
    Assumes (... C) input
    """
    def __init__(self, dim_in, dim_out, subsample_in=True):
        super().__init__()
        self.subsample_in = subsample_in
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.temp_linear = nn.Linear(dim_in, dim_out)
        # self.weight = nn.Parameter(temp_linear.weight)
        # self.bias = nn.Parameter(temp_linear.bias)
        # self.weight = nn.Parameter(temp_linear.weight)
        # self.bias = nn.Parameter(temp_linear.bias)
    
    def forward(self, x, labels):
        # Note - really only works if all batches are the same input type
        # labels = labels[0] # Figure out how to handle this for normal batches later
        label_size = len(labels)
        weight, bias = self.temp_linear.weight, self.temp_linear.bias
        if self.subsample_in:
            scale = (self.dim_in / label_size) ** .5 # Equivalent to swapping init to correct for given subsample of input
            scale = torch.tensor(scale, dtype=x.dtype)
            # print("scale.dtype", scale.dtype)
            # print("x.dtype", x.dtype)
            # print("weight.dtype", weight.dtype)
            # print("bias.dtype", bias.dtype)
            x = scale * F.linear(x, weight[:, labels], bias)
        else:
            x = F.linear(x, weight[labels], bias[labels])
        return x


class CNN(nn.Module):
    """ Image to Patch Embedding """

    def __init__(
        self, embed_dim=768, ndims=[1,2], patch_size=16,
        groups=12, padding_mode='reflect', n_states=12,
        customize=False, finetune=False
    ):
        super().__init__()
        # self.spatial_ndims = spatial_ndims
        self.customize = customize

        if patch_size not in [8, 16]:
            raise ValueError("Patch size must be one of 8, 16")
        self.ksize = patch_size // 4

        # if finetune:
        #     n_states += 5  # additional states for finetuning on shearflow or euler fields
        self.space_bag = SubsampledLinear(dim_in=n_states, dim_out=embed_dim//4, subsample_in=True)
        self.encoder = nn.ModuleDict({
            str(ndim): nn.Sequential(*[
                Downsample(embed_dim//4, embed_dim//4, kernel_size=self.ksize, stride=self.ksize, padding_mode=padding_mode, bias=False, ndim=ndim),
                RMSGroupNorm(groups, embed_dim//4, affine=True),
                nn.GELU(),
                Downsample(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, padding_mode=padding_mode, bias=False, ndim=ndim),
                RMSGroupNorm(groups, embed_dim//4, affine=True),
                nn.GELU(),
                Downsample(embed_dim//4, embed_dim, kernel_size=2, stride=2, padding_mode=padding_mode, bias=False, ndim=ndim),
                RMSGroupNorm(groups, embed_dim, affine=True),
            ])
            for ndim in ndims
        })
        self.decoder = nn.ModuleDict({
            str(ndim): nn.Sequential(*[
                Upsample(embed_dim, embed_dim//4, kernel_size=2, stride=2, padding_mode='zeros', bias=False, ndim=ndim),
                RMSGroupNorm(groups, embed_dim//4, affine=True),
                nn.GELU(),
                Upsample(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, padding_mode='zeros', bias=False, ndim=ndim),
                RMSGroupNorm(groups, embed_dim//4, affine=True),
                nn.GELU(),
            ])
            for ndim in ndims
        })
        if 1 in ndims:
            out_head1d = conv_module(1, True)(embed_dim//4, n_states, kernel_size=self.ksize, stride=self.ksize)  # included in the decoder, won't be a problem
            self.out_kernel1d = nn.Parameter(out_head1d.weight)
            self.out_bias1d = nn.Parameter(out_head1d.bias)
        if 2 in ndims:
            out_head2d = conv_module(2, True)(embed_dim//4, n_states, kernel_size=self.ksize, stride=self.ksize)
            self.out_kernel2d = nn.Parameter(out_head2d.weight)
            self.out_bias2d = nn.Parameter(out_head2d.bias)
        if 3 in ndims:
            out_head3d = conv_module(3, True)(embed_dim//4, n_states, kernel_size=self.ksize, stride=self.ksize)
            self.out_kernel3d = nn.Parameter(out_head3d.weight)
            self.out_bias3d = nn.Parameter(out_head3d.bias)

    def encode(self, x, state_labels):
        """ x is B C H W D """
        # first linear depending on the type of fields ('state') present
        x = rearrange(x, 'b c ... -> b ... c')
        x = self.space_bag(x, state_labels)
        x = rearrange(x, 'b ... c -> b c ...')
        spatial_ndims = x.ndim - 2
        x = self.encoder[str(spatial_ndims)](x)
        return x

    def decode(self, x, state_labels):
        spatial_ndims = x.ndim - 2
        x = self.decoder[str(spatial_ndims)](x)
        if spatial_ndims == 1:
            x = F.conv_transpose1d(x, self.out_kernel1d[:,state_labels,...], self.out_bias1d[state_labels], stride=self.ksize)
        elif spatial_ndims == 2:
            x = F.conv_transpose2d(x, self.out_kernel2d[:,state_labels,...], self.out_bias2d[state_labels], stride=self.ksize)
        elif spatial_ndims == 3:
            x = F.conv_transpose3d(x, self.out_kernel3d[:,state_labels,...], self.out_bias3d[state_labels], stride=self.ksize)
        else:
            x = self.decoder(x)
        return x
