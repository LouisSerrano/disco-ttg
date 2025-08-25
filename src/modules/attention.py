import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath


class RMSGroupNorm(nn.Module):
    r"""Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = self.register_parameter('bias', None) #Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            # nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


class ContinuousPositionBias1D(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.num_heads = n_heads
        self.cpb_mlp = nn.Sequential(
            nn.Linear(1, dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(dim, n_heads, bias=False)
    )
        
    def forward(self, h):
        dtype, device = self.cpb_mlp[0].weight.dtype, self.cpb_mlp[0].weight.device
        relative_coords = torch.cat([torch.arange(1, h//2+1, dtype=dtype, device=device),
                torch.arange(-(h//2-1), h//2+1, dtype=dtype, device=device),
                torch.arange(-(h//2-1), 0, dtype=dtype, device=device)
        ])  / (h-1)

        coords = torch.arange(h, dtype=torch.float32, device=device)
        coords = coords[None, :] - coords[:, None]
        coords = coords + (h-1)

        rel_pos_model = 16 * torch.sigmoid(self.cpb_mlp(relative_coords[:, None]).squeeze())
        biases = rel_pos_model[coords.long()]
        return biases.permute(2, 0, 1).unsqueeze(0).contiguous()


class RelativePositionBias(nn.Module):

    """
    From https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16 

    Implementation of T5 relative position bias - can probably do better, but starting with something known.
    """
    def __init__(self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=2):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=32):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen, bc=0):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long,
                                        device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long,
                                       device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        if bc == 1:
            thresh = klen // 2
            relative_position[relative_position < -thresh] = relative_position[relative_position < -thresh] % thresh
            relative_position[relative_position > thresh] = relative_position[relative_position > thresh] % -thresh
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, qlen, klen, bc=0):
        return self.compute_bias(qlen, klen, bc)  # shape (1, num_heads, qlen, klen)
    

class MLP(nn.Module):
    def __init__(self, dim, exp_factor=4.):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * exp_factor))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim * exp_factor), dim)
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class AxialAttentionBlock(nn.Module):  # TODO: in 1D, will have to be unified with the time attention
    def __init__(self, 
        hidden_dim=768, num_heads=12, drop_path=0, layer_scale_init_value=1e-6, 
        bias_type='rel', norm_groups=12
    ):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSGroupNorm(norm_groups, hidden_dim, affine=True)
        self.norm2 = RMSGroupNorm(norm_groups, hidden_dim, affine=True)
        if layer_scale_init_value > 0:
            self.gamma_att = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), requires_grad=True)
            self.gamma_mlp = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), requires_grad=True)
        else:
            self.gamma_att = None
            self.gamma_mlp = None
        self.input_heads = nn.ModuleList([nn.Conv3d(hidden_dim, 3*hidden_dim, 1) for _ in range(3)])
        self.output_head = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.qnorms = nn.ModuleList([nn.LayerNorm(hidden_dim//num_heads) for _ in range(3)])
        self.knorms = nn.ModuleList([nn.LayerNorm(hidden_dim//num_heads) for _ in range(3)])
        if bias_type == 'none':
            self.rel_pos_biases = None
        elif bias_type == 'continuous':
            self.rel_pos_biases = ContinuousPositionBias1D(dim=512, n_heads=num_heads)
        else:
            self.rel_pos_biases = nn.ModuleList([RelativePositionBias(n_heads=num_heads) for _ in range(3)])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MLP(hidden_dim)
        self.mlp_norm = RMSGroupNorm(norm_groups, hidden_dim, affine=True)

    def spatial_forward(self, x, axis_index, return_att=False):
        # Get shapes to use for relative position bias
        B, C, H, W, D = x.shape
        shapes = [H, W, D]
        # Make the forward/backward strings depending on the index
        all_inds = ['h', 'w', 'd']
        # Temp until 3D added
        # axis_index = axis_index % 3
        remainder_inds = list(filter(lambda st: st != all_inds[axis_index], all_inds))
        forward_string = f'b he h w d c -> (b {" ".join(remainder_inds)}) he {all_inds[axis_index]} c'
        # rearrange(xx, '(b h) he w c -> b (he c) h w', h=H)
        backward_string = f'(b {" ".join(remainder_inds)}) he {all_inds[axis_index]} c -> b (he c) h w d'
        # Apply the input head and split into Q, K, V
        if axis_index == 0:
            x = self.input_heads[0](x)
        elif axis_index == 1:
            x = self.input_heads[1](x)
        elif axis_index == 2:
            x = self.input_heads[2](x)
        else:
            raise ValueError("Invalid axis_index")
        x = rearrange(x, 'b (he c) h w d ->  b he h w d c', he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        # Apply QK norm and split by heads
        if axis_index == 0:
            q, k = self.qnorms[0](q), self.knorms[0](k)
        elif axis_index == 1:
            q, k = self.qnorms[1](q), self.knorms[1](k)
        elif axis_index == 2:
            q, k = self.qnorms[2](q), self.knorms[2](k)
        else:
            raise ValueError("Invalid axis_index")
        qx, kx, vx = map(lambda x: rearrange(x, forward_string), [q,k,v])
        # Rel pos bias
        rel_pos_bias_x = None
        if self.rel_pos_biases is not None:
            if axis_index == 0:
                rel_pos_bias_x = self.rel_pos_biases[0](shapes[0], shapes[0])
            elif axis_index == 1:
                rel_pos_bias_x = self.rel_pos_biases[1](shapes[1], shapes[1])
            elif axis_index == 2:
                rel_pos_bias_x = self.rel_pos_biases[2](shapes[2], shapes[2])
            else:
                raise ValueError("Invalid axis_index")
        # Complicated return mask logic
        if return_att:
            if rel_pos_bias_x is not None:
                attx = torch.softmax((qx @ kx.transpose(-1, -2))/math.sqrt(kx.shape[-1]) + rel_pos_bias_x, -1)
            else:
                attx = torch.softmax((qx @ kx.transpose(-1, -2))/math.sqrt(kx.shape[-1]), -1)
        if rel_pos_bias_x is not None:
            xx = F.scaled_dot_product_attention(qx, kx, vx, attn_mask=rel_pos_bias_x)
        else:
            xx = F.scaled_dot_product_attention(qx.contiguous(), kx.contiguous(), vx.contiguous())
        if axis_index == 0:
            xx = rearrange(xx, backward_string, w=W, d=D)
        elif axis_index == 1:
            xx = rearrange(xx, backward_string, h=H, d=D)
        else:
            xx = rearrange(xx, backward_string, h=H, w=W)
        return xx

    def forward(self, x, axis_order, return_att=False):
        # input is t b c h w
        input = x.clone()
        x = self.norm1(x)

        ndim = x.squeeze((-1,-2)).ndim - 2

        out = 0
        for axis in axis_order:
            out = out + self.spatial_forward(x, axis, return_att=return_att)

        x = out / ndim
        x = self.norm2(x)
        x = self.output_head(x)
        x = self.drop_path(x*self.gamma_att[None,:,None,None,None]) + input

        # MLP
        input = x.clone()
        x = rearrange(x, 'b c h w d -> b h w d c')
        x = self.mlp(x)
        x = rearrange(x, 'b h w d c -> b c h w d')
        x = self.mlp_norm(x)
        output = input + self.drop_path(self.gamma_mlp[None,:,None,None,None] * x)
        if return_att:
            return output, []#[attx, rel_pos_bias_x, atty, rel_pos_bias_y]
        else:
            return output, []


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, drop_path=0, layer_scale_init_value=1e-6, bias_type='rel', num_groups=1):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSGroupNorm(num_groups, hidden_dim, affine=True)
        self.norm2 = RMSGroupNorm(num_groups, hidden_dim, affine=True)
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), requires_grad=True)
        else:
            self.gamma = None
        self.input_head = nn.Conv3d(hidden_dim, 3*hidden_dim, 1)
        self.output_head = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.qnorm = nn.LayerNorm(hidden_dim//num_heads)
        self.knorm = nn.LayerNorm(hidden_dim//num_heads)
        if bias_type == 'none':
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == 'continuous':
            self.rel_pos_bias = ContinuousPositionBias1D(dim=512, n_heads=num_heads)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, return_att=False):
        T, B, C, H, W, D = x.shape
        # T, B, C, H, W = x.shape
        input = x.clone()
        x = rearrange(x, 't b c ... -> (t b) c ...')
        x = self.norm1(x)
        x = self.input_head(x)
        x = rearrange(x, '(t b) (he c) ... ->  (b ...) he t c', t=T, he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)
        rel_pos_bias = self.rel_pos_bias(T, T)
        if return_att:
            if rel_pos_bias is not None:
                att = torch.softmax((q @ k.transpose(-1, -2))/math.sqrt(k.shape[-1]) + rel_pos_bias, -1)
            else:
                att = torch.softmax((q @ k.transpose(-1, -2))/math.sqrt(k.shape[-1]), -1)
        if rel_pos_bias is not None:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=rel_pos_bias) 
        else:
            x = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
        x = rearrange(x, '(b h w d) he t c -> (t b) (he c) h w d', h=H, w=W, d=D)
        x = self.norm2(x)
        x = self.output_head(x)
        x = rearrange(x, '(t b) c ... -> t b c ...', t=T)
        output = self.drop_path(x*self.gamma[None, None, :, None, None, None]) + input
        if return_att:
            return output, [att, rel_pos_bias]
        return output, []


class SpaceTimeBlock(nn.Module):
    """ Operates similar to standard MHSA -> Inverted Bottleneck but with ConvNext 
    block replacing linear part. """
    def __init__(self, 
        dim: int,
        num_heads: int,
        bias_type: str,
        drop_path: float
    ):
        super().__init__()
        self.temporal = AttentionBlock(dim, num_heads, drop_path, 1e-6, bias_type)
        self.spatial = AxialAttentionBlock(dim, num_heads, drop_path, 1e-6, bias_type, num_heads*2)

    def forward(self, x, return_att=False):
        spatial_ndims = x.ndim - 3
        x = x[(...,) + (None,) * (6 - x.ndim)]  # x is now B T C H W D
        T = x.shape[1]
        x = rearrange(x, 'b t c ... -> t b c ...')
        x, t_att = self.temporal(x, return_att=return_att)
        x = rearrange(x, 't b c ... -> (t b) c ...')
        axis_order = torch.randperm(spatial_ndims) # Used for ordering axes in axial att
        x, s_att = self.spatial(x, axis_order, return_att=return_att)
        x = rearrange(x, '(t b) c ... -> b t c ...', t=T)
        if return_att:
            return x, t_att + s_att
        else:
            return x, []
