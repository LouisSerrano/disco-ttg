""" DISCO: train a hypernetwork (hpnn) to determine the operator network (opnn) 
observed in the context. """
import sys
sys.path.append(".")
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torchode as to
from einops import rearrange
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import parameters_to_vector
from torch.func import functional_call, vmap
from ..torchdiffeq import odeint, odeint_adjoint

#from src.utils import standardize
#from src.models.tokenizer import Downsample, RMSGroupNorm
#from src.models.attention import SpaceTimeBlock

from ..utils.database import standardize
from ..modules.tokenizer import Downsample, RMSGroupNorm
from ..modules.attention import SpaceTimeBlock
from typing import Optional

def is_adaptive_solver(method: str) -> bool:
    return method in ['adaptive_heun', 'fehlberg2', 'bosh3', 'dopri5', 'dopri8']


def make_conv(
    dim: int, c_in: int, c_out: int, ksize: int, stride: int, padding: int, padding_mode: str, groups: int
) -> nn.Module:
    if dim == 1:
        return nn.Conv1d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups, bias=False)
    elif dim == 2:
        return nn.Conv2d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups, bias=False)
    elif dim == 3:
        return nn.Conv3d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups, bias=False)
    else:
        raise ValueError(f"Unsupported dim={dim}")
    

def make_conv_transpose(
    dim: int, c_in: int, c_out: int, ksize: int, stride: int, padding: int, groups: int
) -> nn.Module:
    if dim == 1:
        return nn.ConvTranspose1d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding, groups=groups, bias=False)
    elif dim == 2:
        return nn.ConvTranspose2d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding, groups=groups, bias=False)
    elif dim == 3:
        return nn.ConvTranspose3d(c_in, c_out, kernel_size=ksize, stride=stride, padding=padding, groups=groups, bias=False)
    else:
        raise ValueError(f"Unsupported dim={dim}")


###### START (hpnn) ######

class SubsampledInLinear(nn.Module):
    """
    Cross between a linear layer and EmbeddingBag - takes in input 
    and list of indices denoting which state variables from the state
    vocab are present and only performs the linear layer on rows/cols relevant
    to those state variables
    
    Assumes (... C) input
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
    
    def forward(self, x, labels):
        dim_in = self.linear.in_features
        label_size = len(labels)
        weight, bias = self.linear.weight, self.linear.bias
        scale = torch.tensor((dim_in / label_size) ** .5, dtype=x.dtype, device=x.device)
        x = scale * F.linear(x, weight[:, labels], bias)
        return x
    

class SubsampledOutLinear(nn.Module):
    """
    Cross between a linear layer and EmbeddingBag - takes in input 
    and list of indices denoting which state variables from the state
    vocab are present and only performs the linear layer on rows/cols relevant
    to those state variables
    
    Assumes (... C) input
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
    
    def forward(self, x, labels):
        weight, bias = self.linear.weight, self.linear.bias
        x = F.linear(x, weight[labels, :], bias[labels])
        return x


class Encoder(nn.Module):

    def __init__(self, 
        patch_size: int,
        embed_dim: int, 
        spatial_ndims: int, 
        padding_mode: str, 
        groups: int
    ):
        super().__init__()
        if patch_size not in [8, 16]:
            raise ValueError("Patch size must be one of 8, 16")
        ksize = patch_size // 4
        self.encoder = nn.Sequential(*[
            Downsample(embed_dim//4, embed_dim//4, kernel_size=ksize, stride=ksize, padding_mode=padding_mode, bias=False, ndim=spatial_ndims),
            RMSGroupNorm(groups, embed_dim//4, affine=True),
            nn.GELU(),
            Downsample(embed_dim//4, embed_dim//4, kernel_size=2, stride=2, padding_mode=padding_mode, bias=False, ndim=spatial_ndims),
            RMSGroupNorm(groups, embed_dim//4, affine=True),
            nn.GELU(),
            Downsample(embed_dim//4, embed_dim, kernel_size=2, stride=2, padding_mode=padding_mode, bias=False, ndim=spatial_ndims),
            RMSGroupNorm(groups, embed_dim, affine=True),
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Hypernetwork(nn.Module):

    def __init__(self, 
        patch_size: int,
        n_states: int,
        ndims: List[int], 
        embed_dim: int, 
        groups: int,
        processor_blocks: int,
        drop_path: float,
        num_heads: int,
        bias_type: str,
    ) -> None:
        super().__init__()

        # Adapters to 1d, 2d, 3d
        self.adapter = SubsampledInLinear(dim_in=n_states, dim_out=embed_dim//4)

        # encoder (one for each 1d, 2d, 3d)
        self.encoder = nn.ModuleDict({
            str(ndim): Encoder(patch_size, embed_dim, ndim, "reflect", groups)
            for ndim in ndims
        })

        # processor (common to different datasets)
        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(dim=embed_dim, num_heads=num_heads, bias_type=bias_type, drop_path=dp)
            for dp in self.dp
        ])

    def forward(self, x: Tensor, state_labels: Tensor) -> Tensor:
        """ x is B T C H (W) (D) """
        # dimensions
        B = x.shape[0]
        spatial_ndims = x.ndim - 3
        spatial_dims = tuple(range(3,x.ndim))
        x = x[(...,) + (None,) * (6 - x.ndim)]  # now x is B T C H W D

        # preprocess
        x = standardize(x, dims=(1,*spatial_dims), return_stats=False)

        # adapt to the number of in channels
        x = rearrange(x, 'b t c ... -> (b t) ... c')
        x = self.adapter(x, state_labels)
        x = rearrange(x, 'bt ... c -> bt c ...')

        # encode
        x = x.squeeze((-2,-1))
        x = self.encoder[str(spatial_ndims)](x)
        x = rearrange(x, '(b t) c ... ->  b t c ...', b=B)

        # attention layers
        all_att_maps = []
        for blk in self.blocks:
            x, att_maps = blk(x, return_att=False)
            all_att_maps += att_maps

        return x

###### END (hpnn) ######

###### START (opnn) ######

class Down(nn.Module):
    def __init__(self, 
        dim: int, 
        in_channels: int, 
        mid_channels: int,
        out_channels: int, 
        groups: int,
        norm_groups: int,
        padding_mode: str,
    ):
        super().__init__()
        self.conv1 = make_conv(dim, in_channels, mid_channels, ksize=2, stride=2, padding=0, padding_mode=padding_mode, groups=1)
        self.norm1 = nn.GroupNorm(norm_groups, mid_channels)
        self.gelu1 = nn.GELU()
        self.conv2 = make_conv(dim, mid_channels, out_channels, ksize=3, stride=1, padding=1, padding_mode=padding_mode, groups=groups)
        self.norm2 = nn.GroupNorm(norm_groups, out_channels) 
        self.gelu2 = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu2(x)
        return x


class Up(nn.Module):
    def __init__(self, 
        dim: int, 
        in_channels: int, 
        out_channels: int, 
        groups: int,
        norm_groups: int,
        padding_mode: str,
    ):
        super().__init__()
        self.up = make_conv_transpose(dim, in_channels, in_channels//2, ksize=2, stride=2, padding=0, groups=1)
        self.norm = nn.GroupNorm(norm_groups, in_channels//2)
        self.gelu = nn.GELU()
        self.conv = make_conv(dim, in_channels, out_channels, ksize=3, stride=1, padding=1, padding_mode=padding_mode, groups=groups)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = self.up(x1)
        x = torch.cat([x, x2], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x


def create_frontier_mask(x):
    """
    General frontier mask creator for arbitrary spatial dimensions (1D, 2D, 3D, ...).
    Assumes shape = (B, *spatial).
    """
    ndim = x.ndim - 1
    mask = torch.zeros_like(x)
    
    for dim in range(ndim):
        slice_down = [...] + [0 if i == dim else slice(None) for i in range(ndim)]
        slice_up = [...] + [-1 if i == dim else slice(None) for i in range(ndim)]
        
        mask[tuple(slice_down)] = 1
        mask[tuple(slice_up)] = 1

    return mask


# def create_frontier_mask(shape, dtype, device):
#     # Create a tensor of zeros
#     mask = torch.zeros(shape, dtype=dtype, device=device)
    
#     # Set the frontier (boundary) to ones
#     mask[...,0, :] = 1  # Top row
#     mask[...,-1, :] = 1  # Bottom row
#     mask[...,:, 0] = 1  # Left column
#     mask[...,:, -1] = 1  # Right column
#     mask -= mask.mean((-1,-2), keepdims=True)
#     mask /= mask.std((-1,-2), keepdims=True)
    
#     return mask


class OperatorNetwork(nn.Module):
    def __init__(self, n_states, start, spatial_ndims, boundary_conditions, norm_groups, num_heads=0, updown_ksize=1, bottleneck_multiplier=2):
        super().__init__()
        self.n_states = n_states
        self.start = start
        self.spatial_ndims = spatial_ndims
        self.boundary_conditions = boundary_conditions
        self.norm_groups = norm_groups
        self.num_heads = num_heads
        self.updown_ksize = updown_ksize

        if boundary_conditions == 'periodic':
            padding_mode = 'circular'
        elif boundary_conditions == 'reflect':
            padding_mode = 'reflect'
        elif boundary_conditions == 'replicate':
            padding_mode = 'replicate'
        elif boundary_conditions == 'zeros':
            padding_mode = 'zeros'
        else:  # None or default
            padding_mode = 'reflect'
        self.add_mask = boundary_conditions != 'periodic'
        bc_offset = int(self.add_mask)

        # layers 
        self.adapter_in = SubsampledInLinear(dim_in=n_states, dim_out=start)
        self.convs_in = nn.Sequential(
            make_conv(spatial_ndims, start+bc_offset, start, ksize=3, stride=1, padding=1, padding_mode=padding_mode, groups=1),
            make_conv(spatial_ndims, start, start, ksize=3, stride=1, padding=1, padding_mode=padding_mode, groups=1),
        )
        self.downs = nn.ModuleList([
            Down(spatial_ndims, start * 1, start * 2, start * 2, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*2),
            Down(spatial_ndims, start * 2, start * 4, start * 4, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*4),
            Down(spatial_ndims, start * 4, start * 8, start * 8, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*8),
            Down(spatial_ndims, start * 8, start * 16, start * 16, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*16),
        ])  
        self.ups = nn.ModuleList([
            Up(spatial_ndims, start * 16, start * 8, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*8),
            Up(spatial_ndims, start * 8, start * 4, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*4),
            Up(spatial_ndims, start * 4, start * 2, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*2),
            Up(spatial_ndims, start * 2, start * 1, norm_groups=norm_groups, padding_mode=padding_mode, groups=start*1),
        ])
        bottleneck_channels = start * 16 * bottleneck_multiplier
        self.bottleneck = nn.Sequential(
            make_conv(spatial_ndims, start * 16, bottleneck_channels, ksize=1, stride=1, padding=0, padding_mode=padding_mode, groups=1),
            nn.GroupNorm(norm_groups, bottleneck_channels),
            nn.GELU(),
            make_conv(spatial_ndims, bottleneck_channels, start * 16, ksize=1, stride=1, padding=0, padding_mode=padding_mode, groups=1),
        )
        self.convs_out = nn.Sequential(
            make_conv(spatial_ndims, start, start, ksize=3, stride=1, padding=1, padding_mode=padding_mode, groups=1),
            nn.GroupNorm(norm_groups, start),
            nn.GELU(),
            make_conv(spatial_ndims, start, start, ksize=3, stride=1, padding=1, padding_mode=padding_mode, groups=1),
        )
        self.adapter_out = SubsampledOutLinear(dim_in=start, dim_out=n_states)

        #self.time_scale = nn.Parameter(torch.zeros(1))  # exp(0) = 1.0

    def forward(self, x: Tensor, state_labels: Tensor) -> Tensor:
        
        # adapt to varying number of channels
        x = rearrange(x, 'b c ... -> b ... c')
        x = self.adapter_in(x, state_labels)
        x = rearrange(x, 'b ... c -> b c ...')

        # to deal with boundary conditions
        if self.add_mask:
            mask = create_frontier_mask(x[:,0,:]).unsqueeze(1)
            x = torch.cat([x, mask], dim=1)

        # first convs
        x1 = self.convs_in(x)
        
        # downsampling
        x2 = self.downs[0](x1)
        x3 = self.downs[1](x2)
        x4 = self.downs[2](x3)
        x5 = self.downs[3](x4)

        # bottleneck 
        x5 = x5 + self.bottleneck(x5)

        # upsampling
        x = self.ups[0](x5, x4)
        x = self.ups[1](x, x3)
        x = self.ups[2](x, x2)
        x = self.ups[3](x, x1)
        x = self.convs_out(x)

        # adapt to varying number of channels
        x = rearrange(x, 'b c ... -> b ... c')
        x = self.adapter_out(x, state_labels)
        x = rearrange(x, 'b ... c -> b c ...')

        # Apply time scaling with exponential to handle orders of magnitude
        #time_scale = torch.exp(self.time_scale)
        #x = time_scale * x
        return x

###### END (opnn) ######


def vectors_to_parameters(vec: torch.Tensor, parameters_dict: Dict) -> Dict:
    r"""Copy slices of vectors into an iterable of parameters.

    Args:
        vec (Tensor): a batched vector representing the parameters of a model. 
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, but got: {torch.typename(vec)}")

    # Pointer for slicing the vector for each parameter
    new_parameters_dict = {}
    B = vec.size(0)
    pointer = 0
    for name, param in parameters_dict.items():
        num_param = param.numel()
        param_chunk = vec[:, pointer : pointer + num_param]
        new_parameters_dict[name] = param_chunk.view(B, *param.shape)
        pointer += num_param
    return new_parameters_dict


class DISCOHouse(nn.Module):
    """ A simplified full DISCO model. """
    def __init__(self, 
        n_states: int,
        hidden_dim: int,
        theta_dim: int,
        patch_size: int,
        ndims: List[int],
        groups: int,
        processor_blocks: int,
        drop_path: float,
        num_heads: int,
        bias_type: str,
        hpnn_head_hidden_dim: int,
        max_steps: int = 32,
        atol: float = 1e-9,
        rtol: float = 5e-6,
        use_adjoint: bool = True,
        decoder_use_bias: bool = True,
        principled_initialization: bool = False,
        solver: str = "dopri5",
        boundary_conditions: str = None,
        opnn_channels: int = 8,
        opnn_bottleneck_multiplier: int = 2,
        default_integration_time: float = 1.0,
    ):
        super().__init__()
        
        self.use_adjoint = use_adjoint
        self.principled_initialization = principled_initialization
        self.solver = solver
        self.boundary_conditions = boundary_conditions
        self.opnn_channels = opnn_channels
        self.opnn_bottleneck_multiplier = opnn_bottleneck_multiplier
        self.default_integration_time = default_integration_time

        # Hypernetwork (hpnn)
        self.hpnn = Hypernetwork(
            patch_size=patch_size,
            n_states=n_states,
            ndims=ndims,
            embed_dim=hidden_dim,
            groups=groups,
            processor_blocks=processor_blocks,
            drop_path=drop_path,
            num_heads=num_heads,
            bias_type=bias_type,
        )
        self.hpnn_projector = nn.Linear(hidden_dim, theta_dim)
        self.scalar_projector = nn.Linear(hidden_dim, 1)

        # Operator network (opnn)
        self.opnns = nn.ModuleDict({
            str(dim): OperatorNetwork(n_states, self.opnn_channels, dim, boundary_conditions=self.boundary_conditions, norm_groups=4, bottleneck_multiplier=self.opnn_bottleneck_multiplier)
            for dim in ndims
        })
        theta_norms = self.init_normalizations()
        for k, norm in theta_norms.items():
            self.register_buffer(f"theta_norm_{k}", norm)

        # Parameter decoder 
        numel_opnn_parameters = {str(dim): parameters_to_vector(self.opnns[str(dim)].parameters()).numel() for dim in ndims}
        self.decoder_common = nn.Sequential(
            nn.Linear(theta_dim, hpnn_head_hidden_dim, bias=decoder_use_bias),
            nn.GELU(),
            nn.Linear(hpnn_head_hidden_dim, hpnn_head_hidden_dim, bias=decoder_use_bias),
            nn.GELU(),
        )
        self.decoder_head = nn.ModuleDict({
            dim: nn.Linear(hpnn_head_hidden_dim, numel, bias=decoder_use_bias)
            for (dim, numel) in numel_opnn_parameters.items()
        })

        zero_init = False
        if decoder_use_bias and zero_init:
            self.decoder_common[0].bias.data.fill_(0)
            self.decoder_common[2].bias.data.fill_(0)
            for (dim, numel) in numel_opnn_parameters.items():
                self.decoder_head[dim].bias.data.fill_(0)

        # Solver specifics
        self.max_steps = max_steps
        self.atol = atol
        self.rtol = rtol
        
        # Apply principled initialization if requested
        if principled_initialization:
            self.apply_principled_initialization()

    def apply_principled_initialization(self):
        """Apply principled initialization to decoder_head:
        - Non-bias terms: Var = 1/(dj * dk) where dk=decoder input dim, dj=opnn layer input dim
        - Bias terms: Var = 1/dk
        """
        for dim_str, decoder_layer in self.decoder_head.items():
            # dk = input dimension of decoder_head layer
            dk = decoder_layer.weight.shape[1]
            
            # Get the corresponding operator network
            opnn = self.opnns[dim_str]
            
            # Collect variance info for each parameter
            variances = []
            for name, param in opnn.named_parameters():
                if 'weight' in name:
                    # Determine dj based on parameter type
                    if isinstance(param, torch.nn.parameter.Parameter):
                        # Check the module that owns this parameter
                        for module in opnn.modules():
                            for param_name, module_param in module.named_parameters():
                                if module_param is param:
                                    if isinstance(module, nn.Linear):
                                        dj = module.in_features
                                    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                                        dj = module.in_channels * param[0,0].numel()  # in_channels * kernel_size
                                    else:
                                        dj = 1
                                    break
                    else:
                        dj = 1
                    variance = 1.0 / (dj * dk)
                else:  # bias terms
                    variance = 1.0 / dk
                
                variances.extend([variance] * param.numel())
            
            # Initialize decoder_head weights
            with torch.no_grad():
                decoder_weight = decoder_layer.weight  # shape: (total_opnn_params, dk)
                param_variances = torch.tensor(variances, device=decoder_weight.device, dtype=decoder_weight.dtype)
                param_stds = param_variances.sqrt()  # shape: (total_opnn_params,)
                
                # Vectorized initialization: each row gets its own std
                random_weights = torch.randn_like(decoder_weight)  # N(0,1)
                decoder_weight.data = random_weights * param_stds.unsqueeze(1)  # broadcast std across columns

    def init_normalizations(self) -> Dict[str, Tensor]:
        """ Determine the normalizations of the opnns. """
        norms = {}
        for k, opnn in self.opnns.items():
            params = dict(opnn.named_parameters())
            params_max = {k: torch.full_like(v, v.abs().max().item()) for k, v in params.items()}
            norm = parameters_to_vector(params_max.values())
            norm = norm / norm.pow(2.0).mean().pow(0.5)
            norms[k] = norm
        return norms
        
    def encode_theta_latent(self, x: Tensor, state_labels: Tensor) -> Tensor:
        """Encode input x and state_labels to produce theta_latent."""
        x_shape = x.shape
        B, T, C = x_shape[:3]
        spatial = x_shape[3:]
        dim = len(spatial)
        state_labels = state_labels.unsqueeze(0).repeat(B, 1)

        # preprocess
        spatial_dims = tuple(range(3,x.ndim))
        x, mean, std = standardize(x, dims=(1,*spatial_dims), return_stats=True)
        metadata = {'mean': mean, 'std': std}  # b t c h w

        # Estimate the PDE's intrisic parameters
        theta_latent = self.hpnn(x, state_labels[0])
        theta_latent = theta_latent.mean((1,3,4,5))  # average on t, x
        theta_latent = self.hpnn_projector(theta_latent)
        return theta_latent, metadata

    def decode_theta(self, theta_latent: Tensor, dim: int) -> Tensor:
        """Decode theta_latent to produce theta (operator parameters)."""
       # Decode the parameters
        theta = self.decoder_common(theta_latent)
        theta = self.decoder_head[str(dim)](theta)
        
        if not self.principled_initialization:
            # normalize weights per layer accounting for the size of each layer
            theta_norm = getattr(self, f"theta_norm_{dim}")
            def signed_sigmoid(x, factor=2.0):
                return factor * (2 * torch.sigmoid(2 * x / factor) - 1)
            theta = theta_norm * signed_sigmoid(theta)
        
        return theta

    def solve_ode_with_2_operators(self, x_input: Tensor, theta_1:Tensor, theta_2:Tensor, state_labels: Tensor, dim: int, integration_time: float, n_future_steps: int, predict_normed: bool, metadata: dict, dt: float = None, solver: Optional[str] = None) -> Tensor:
        """Integrate the operator network using a neural ODE solver."""
        B=x_input.shape[0]
        #state_labels = state_labels.unsqueeze(0).repeat(B, 1)

        def functional_call_batches(model, parameter_vec_batch, x_batch, state_labels):
            param_dict = dict(model.named_parameters())
            batched_params_dict = vectors_to_parameters(parameter_vec_batch, param_dict)
            return vmap(functional_call, in_dims=(None, 0, 0))(model, batched_params_dict, (x_batch, state_labels))

        def opnn1(t, x):
            x = x.unsqueeze(1)
            x = functional_call_batches(self.opnns[str(dim)], theta_1, x, state_labels)
            if x.isnan().any():
                print(f"x is nan")
                print(f"theta_1 {theta_1}")
                print(f"x {x}")
                print(f"x_input {x_input}")
                print(f"state_labels {state_labels}")
                print(f"t {t}")
            
            return x.squeeze(1)
    
        def opnn2(t, x):
            x = x.unsqueeze(1)
            x = functional_call_batches(self.opnns[str(dim)], theta_2, x, state_labels)
            if x.isnan().any():
                print(f"x is nan")
                print(f"theta_1 {theta_1}")
                print(f"x {x}")
                print(f"x_input {x_input}")
                print(f"state_labels {state_labels}")
                print(f"t {t}")
            
            return x.squeeze(1)

        # Determine which solver to use (override or default)
        current_solver = solver if solver is not None else self.solver
        
        options = {'min_step': 1/self.max_steps}
        if self.use_adjoint:
            t = torch.linspace(0, n_future_steps, n_future_steps+1, device=x_input.device)
            t1 = t[1]
            for t_step in t[1:]:
                x, _ = self.solve_ode(x_input, theta_1, state_labels, dim, t1, False, metadata, solver=current_solver)
                x_input = x[:, -1]
                x, _ = self.solve_ode(x_input, theta_2, state_labels, dim, t1, False, metadata, solver=current_solver)
                x_input = x[:, -1]
        else:
            if dt is None:
                dt = 1/self.max_steps
            
            options['step_size'] = dt
            
            t = torch.arange(0, integration_time + dt, dt, device=x_input.device)
            x_current = x_input
            x_trajectory = [x_current]  # Store trajectory including initial state
            
            # For each dt: apply operator1(dt) then operator2(dt)
            for t_step in t[1:]:
                t_interval = torch.tensor([0.0, dt], device=x_input.device)
                
                # Apply operator 1 for dt
                _, x1 = odeint(opnn1, x_current, t=t_interval, rtol=self.rtol, method=current_solver, options=options)
                x_current = x1[-1]
                
                # Apply operator 2 for dt  
                _, x2 = odeint(opnn2, x_current, t=t_interval, rtol=self.rtol, method=current_solver, options=options)
                x_current = x2[-1]
                x_trajectory.append(x_current)
            
            # Convert trajectory to tensor and sample at desired output times
            x_full = torch.stack(x_trajectory, dim=0)[1:]  # Remove initial state
            output_dt = integration_time / n_future_steps
            output_indices = [int(round(i * output_dt / dt)) for i in range(1, n_future_steps + 1)]
            output_indices = [min(idx, len(x_full) - 1) for idx in output_indices]  # Clamp to valid range
            x = x_full[output_indices]
          
        return x, metadata

    def solve_ode(self, x_input: Tensor, theta: Tensor, state_labels: Tensor, dim: int, integration_time: Optional[float] = None, n_future_steps: int = 1, predict_normed: bool = False, metadata: dict = None, dt: Optional[float] = None, solver: Optional[str] = None) -> Tensor:
        """Integrate the operator network using a neural ODE solver."""
        B=x_input.shape[0]
        state_labels = state_labels.unsqueeze(0).repeat(B, 1)
        
        # Use default integration time if not provided
        if integration_time is None:
            integration_time = self.default_integration_time
        
        # Compute dt if not provided
        if dt is None:
            dt = integration_time / self.max_steps
        
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}

        def functional_call_batches(model, parameter_vec_batch, x_batch, state_labels):
            param_dict = dict(model.named_parameters())
            batched_params_dict = vectors_to_parameters(parameter_vec_batch, param_dict)
            return vmap(functional_call, in_dims=(None, 0, 0))(model, batched_params_dict, (x_batch, state_labels))

        def opnn(t, x):
            x = x.unsqueeze(1)
            x = functional_call_batches(self.opnns[str(dim)], theta, x, state_labels)
            if x.isnan().any():
                print(f"x is nan")
                print(f"theta {theta}")
                print(f"x {x}")
                print(f"x_input {x_input}")
                print(f"state_labels {state_labels}")
                print(f"t {t}")
            return x.squeeze(1)

        # Determine which solver to use (override or default)
        current_solver = solver if solver is not None else self.solver

        if self.use_adjoint:
            options = {'min_step': dt}
            #dt = integration_time / n_future_steps
            #t = torch.arange(0, integration_time + dt, dt, device=x.device)
            t = torch.linspace(0, integration_time*n_future_steps, n_future_steps+1, device=x_input.device)
            n_steps, x = odeint_adjoint(opnn, x_input, t=t, rtol=self.rtol, 
                                        method=current_solver, options=options, 
                                        adjoint_params=(theta,))
            x = x[1:,...]
        else:
            options = {'step_size': dt}
            #delta_t = integration_time / n_future_steps  # Same dt calculation
            #t = torch.arange(0, integration_time + delta_t, delta_t, device=x.device)  # Same time grid
            t = torch.linspace(0, integration_time*n_future_steps, n_future_steps+1, device=x_input.device)
            n_steps, x = odeint(opnn, x_input, t=t, rtol=self.rtol, 
                                method=current_solver, options=options)
            x = x[1:,...]

        metadata['n_steps'] = n_steps
        
        if predict_normed:
            x = x * metadata['std'] + metadata['mean']

        x = rearrange(x, 't b c h -> b t c h')
        return x, metadata

    def forward(self, 
        x: Tensor, 
        state_labels: Tensor, 
        predict_normed: bool = False,
        n_future_steps: int = 1,
        integration_time: Optional[float] = None,
        y: Optional[Tensor] = None,
        solver: Optional[str] = None,
        dt: Optional[float] = None,
    ) -> Tensor:
        """ x is B T C H (W) (D) """
        x_shape = x.shape
        B, T, C = x_shape[:3]
        spatial = x_shape[3:]
        dim = len(spatial)
        state_labels = state_labels.unsqueeze(0).repeat(B, 1)
        
        # Use default integration time if not provided
        if integration_time is None:
            integration_time = self.default_integration_time
        
        # Compute dt if not provided
        if dt is None:
            dt = integration_time / self.max_steps

        # preprocess
        spatial_dims = tuple(range(3,x.ndim))
        x_input = x[:,-1,...]
        
        if y is not None:
            #x_input = torch.cat([x_input[:, None], y[:, :-1]], dim=1)
            x_input = y[:, 0]
            #T = x_input.shape[1]
            #x_input = rearrange(x_input, 'b t c h -> (b t) c h')
        else:
            T = 1

        x, mean, std = standardize(x, dims=(1,*spatial_dims), return_stats=True)
        metadata = {'mean': mean, 'std': std}  # b t c h w

        # Keep as initial condition for the solver later
        
        # Estimate the PDE's intrisic parameters
        theta_latent = self.hpnn(x, state_labels[0])
        theta_latent_agg = theta_latent.mean((1,3,4,5))  # average on t, x
        theta_latent = self.hpnn_projector(theta_latent_agg)
        scalar_time = self.scalar_projector(theta_latent_agg)

        metadata['theta_latent'] = theta_latent
        metadata['scalar_time'] = scalar_time

        # Decode the parameters
        theta = self.decoder_common(theta_latent)
        theta = self.decoder_head[str(dim)](theta)
        
        if not self.principled_initialization:
            # normalize weights per layer accounting for the size of each layer
            theta_norm = getattr(self, f"theta_norm_{dim}")
            def signed_sigmoid(x, factor=2.0):
                return factor * (2 * torch.sigmoid(2 * x / factor) - 1)
            theta = theta_norm * signed_sigmoid(theta)

        metadata['theta'] = theta

        if y is not None:
            # factor is ratio of batch size
            factor = x_input.shape[0]//x.shape[0] 
            theta = theta[:, None].repeat(1, factor, 1)
            theta = rearrange(theta, 'b t c -> (b t) c')

            scalar_time = scalar_time[:, None].repeat(1, factor, 1)
            scalar_time = rearrange(scalar_time, 'b t c -> (b t) c')
            scalar_time = scalar_time.squeeze(-1)

            state_labels = state_labels.unsqueeze(1).repeat(1, factor, 1)
            state_labels = rearrange(state_labels, 'b t c -> (b t) c')
        
        # Helper to batch computations
        def functional_call_batches(model, parameter_vec_batch, x_batch, state_labels): # nn.Module, B x dim_parameter, B x C x spatial, B x n_states
            param_dict = dict(model.named_parameters())
            batched_params_dict = vectors_to_parameters(parameter_vec_batch, param_dict)
            return vmap(functional_call, in_dims=(None, 0, 0))(model, batched_params_dict, (x_batch, state_labels))

        # The operator that is integrated by the solver
        def opnn(t, x):
            x = x.unsqueeze(1)
            #print(f"scalar_time {scalar_time.shape}")
            #print(f"x {functional_call_batches(self.opnns[str(dim)], theta, x, state_labels).shape}")
            # Reshape scalar_time to broadcast with x: [B] -> [B, 1, 1, 1]
            scalar_time_reshaped = scalar_time.view(-1, 1, 1, 1)
            x = scalar_time_reshaped * functional_call_batches(self.opnns[str(dim)], theta, x, state_labels)
            
            #if x.isnan().any():
            #    print(f"x is nan")
            #    print(f"theta {theta}")
            #    print(f"x {x}")
            #    print(f"x_input {x_input}")
            #    print(f"state_labels {state_labels}")
            #    print(f"t {t}")
            return x.squeeze(1)
        
        # Determine which solver to use (override or default)
        current_solver = solver if solver is not None else self.solver

        if self.use_adjoint:
            options = {'min_step': dt}
            #dt = integration_time / n_future_steps
            #t = torch.arange(0, integration_time + dt, dt, device=x.device)
            t = torch.linspace(0, integration_time*n_future_steps, n_future_steps+1, device=x.device)
            n_steps, x = odeint_adjoint(opnn, x_input, t=t, rtol=self.rtol, 
                                        method=current_solver, options=options, 
                                        adjoint_params=(theta,))
            x = x[1:,...]
        else:
            options = {'step_size': dt}
            #delta_t = integration_time / n_future_steps  # Same dt calculation
            #t = torch.arange(0, integration_time + delta_t, delta_t, device=x.device)  # Same time grid
            t = torch.linspace(0, integration_time*n_future_steps, n_future_steps+1, device=x.device)
            n_steps, x = odeint(opnn, x_input, t=t, rtol=self.rtol, 
                                method=current_solver, options=options)
            x = x[1:,...]

        metadata['n_steps'] = n_steps

        if predict_normed:
            x = x * metadata['std'] + metadata['mean']

        #if y is not None:
        #    x = rearrange(x, 'l b c h -> b l c h')
       #     x = rearrange(x, '(b t) l c h -> b t l c h', t=T)
        #    x = x.squeeze(2)
        #else:
        #    
        x = rearrange(x, 't b c h -> b t c h')
        
        return x, metadata


class DISCOExpert(nn.Module):
    """DISCOExpert: Model with N expert Unets, hypernetwork produces N gating weights, dynamics is a weighted sum of experts."""
    def __init__(self, 
        n_states: int,
        hidden_dim: int,
        n_experts: int,
        patch_size: int,
        ndims: list,
        groups: int,
        processor_blocks: int,
        drop_path: float,
        num_heads: int,
        bias_type: str,
        max_steps: int = 32,
        atol: float = 1e-9,
        rtol: float = 5e-6,
        use_adjoint: bool = True,
        gating_log_scale: bool = False,
        solver: str = "dopri5",
        opnn_channels: int = 32,
        opnn_bottleneck_multiplier: int = 2,
    ):
        super().__init__()
        self.use_adjoint = use_adjoint
        self.n_experts = n_experts
        self.ndims = ndims
        self.max_steps = max_steps
        self.atol = atol
        self.rtol = rtol
        self.gating_log_scale = gating_log_scale
        self.solver = solver

        # Hypernetwork (hpnn) produces N gating weights
        self.hpnn = Hypernetwork(
            patch_size=patch_size,
            n_states=n_states,
            ndims=ndims,
            embed_dim=hidden_dim,
            groups=groups,
            processor_blocks=processor_blocks,
            drop_path=drop_path,
            num_heads=num_heads,
            bias_type=bias_type,
        )
        self.hpnn_projector = nn.Linear(hidden_dim, n_experts)

        # N expert Unets (OperatorNetworks)
        #self.experts = nn.ModuleList([
        #    OperatorNetwork(n_states, 8, dim, boundary_conditions="None", norm_groups=4)
       #     for _ in range(n_experts)
        #])

        self.opnns = nn.ModuleDict({
            str(dim):  nn.ModuleList([
            OperatorNetwork(n_states, opnn_channels, dim, boundary_conditions=None, norm_groups=4, bottleneck_multiplier=opnn_bottleneck_multiplier)
            for _ in range(n_experts)
        ]) for dim in ndims
        })

    def get_gating_weights(self, x: Tensor, state_labels: Tensor) -> Tensor:
        """Produce N gating weights from the hypernetwork."""
        # x: (B, T, C, ...)
        B = x.shape[0]
        spatial_dims = tuple(range(3, x.ndim))
        x, mean, std = standardize(x, dims=(1, *spatial_dims), return_stats=True)
        theta_latent = self.hpnn(x, state_labels[0])
        # Pool over time and space
        theta_latent = theta_latent.mean((1,3,4,5))  # average on t, x
        gating_logits = self.hpnn_projector(theta_latent)  # (B, n_experts)

        #while theta_latent.ndim > 2:
        #     theta_latent = theta_latent.mean(-1)
        
        if self.gating_log_scale:
            # Use log scale for weights (e.g., exp for positive, log1p for stability)
            gating_weights = torch.exp(gating_logits)
        else:
            gating_weights = F.softplus(gating_logits) #torch.relu(gating_logits)
        # Optionally normalize (softmax or sum to 1)
        # gating_weights = gating_weights / (gating_weights.sum(-1, keepdim=True) + 1e-8)
        return gating_weights, {'mean': mean, 'std': std, 'gating_logits': gating_logits}

    def forward(self, 
        x: Tensor, 
        state_labels: Tensor, 
        predict_normed: bool = False,
        n_future_steps: int = 1,
        y: Optional[Tensor] = None,
        solver: Optional[str] = None,
    ):
        """ x is B T C H (W) (D) """
        x_shape = x.shape
        B, T, C = x_shape[:3]
        spatial = x_shape[3:]
        dim = len(spatial)
        state_labels = state_labels.unsqueeze(0).repeat(B, 1)
        spatial_dims = tuple(range(3, x.ndim))
        x_input = x[:, -1, ...]
        if y is not None:
            x_input = y[:, 0]
        gating_weights, metadata = self.get_gating_weights(x, state_labels)
        metadata['gating_weights'] = gating_weights

        expert_outputs = []
        for expert in self.opnns[str(dim)]:
            out = expert(x_input, state_labels[0])
            expert_outputs.append(out)
        expert_outputs = torch.stack(expert_outputs, dim=-1) 
        metadata['expert_outputs'] = expert_outputs

        def functional_call_batches(model, x_batch, state_labels):
            return vmap(model, in_dims=(0, 0))(x_batch, state_labels)

        def expert_opnn(t, x):
            expert_outputs = []
            for expert in self.opnns[str(dim)]:
                out = expert(x, state_labels[0])
                expert_outputs.append(out)
            expert_outputs = torch.stack(expert_outputs, dim=-1)
            
            # Weighted sum over experts
            weights = gating_weights.unsqueeze(1).unsqueeze(1)
            x = (expert_outputs * weights).sum(-1)
            if x.isnan().any():
                print(f"x is nan in expert_opnn")
            return x

        # Determine which solver to use (override or default)
        current_solver = solver if solver is not None else self.solver
        
        options = {'min_step': 1/self.max_steps}
        if self.use_adjoint:
            t = torch.linspace(0, n_future_steps, n_future_steps+1, device=x.device)
            adjoint_params = list(self.opnns.parameters()) + list(gating_weights) #list(self.hpnn.parameters()) + list(self.hpnn_projector.parameters())
            n_steps, x_out = odeint_adjoint(expert_opnn, x_input, t=t, rtol=self.rtol, method=current_solver, options=options, adjoint_params=adjoint_params)
            x_out = x_out[1:, ...]
        else:
            dt = 1/self.max_steps
            t = torch.arange(0, n_future_steps+dt, dt, device=x.device)
            n_steps, x_out = odeint(expert_opnn, x_input, t=t, rtol=self.rtol, method=current_solver, options=options)
            x_out = x_out[1:, ...]
            x_out = x_out[::self.max_steps]
        metadata['n_steps'] = n_steps
        if predict_normed:
            x_out = x_out * metadata['std'] + metadata['mean']
        x_out = rearrange(x_out, 't b c h -> b t c h')
        return x_out, metadata


if __name__ == "__main__":
    print(f"GPU available: {torch.cuda.is_available()}")
    model = DISCOHouse(
        n_states=12,
        hidden_dim=384,
        theta_dim=2,
        patch_size=16,
        ndims=[1,2,3],
        groups=12,
        processor_blocks=12,
        drop_path=0.1,
        num_heads=12,
        bias_type='rel',
        hpnn_head_hidden_dim=384,
        max_steps=32,
        atol=1e-9,
        rtol=5e-6,
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    print("1D case")
    x = torch.randn(8, 16, 1, 256).cuda()
    y = torch.randn(8, 1, 1, 256).cuda()
    state_labels = torch.tensor([0]).cuda()
    out1, metadata = model(x, state_labels, n_future_steps=1)
    loss = torch.nn.functional.mse_loss(out1, y)
    loss.backward()
    optimizer.step()
    print(loss)

    


