import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block
from .pos_embed import get_1d_sincos_pos_embed
from .conditioned.oned_unet import Unet
from einops import rearrange


class PatchEmbed1D(nn.Module):
    """ 1D patch embedding for spatial dimension with temporal stacking and channels """
    def __init__(self, grid_size=256, patch_size=16, in_chans=1, history=16, embed_dim=768):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.history = history
        self.num_patches = grid_size // patch_size
        
        # Project patches to embedding dimension
        # Input: patch_size * in_chans * history (spatial patch with all channel and temporal components)
        self.proj = nn.Linear(patch_size * in_chans * history, embed_dim)
        
    def forward(self, x):
        # x: (B, C, H, T) -> (B, num_patches, patch_size * in_chans * history)
        B, C, H, T = x.shape
        assert C == self.in_chans and H == self.grid_size #and T == self.history
        
        # Reshape to patches: (B, C, num_patches, patch_size, T)
        x = x.reshape(B, C, self.num_patches, self.patch_size, T)
        x = x.permute(0, 2, 3, 1, 4)  # (B, num_patches, patch_size, in_chans, history)
        
        # Flatten spatial, channel, and temporal dimensions for each patch
        x = x.reshape(B, self.num_patches, self.patch_size * self.in_chans * self.history)
        
        # Project to embedding dimension
        x = self.proj(x)
        return x


class DiscoEncoder(nn.Module):
    """ Disco Encoder using MAE-like architecture for encoding input sequences """
    def __init__(self, grid_size=256, patch_size=16, in_chans=3, history=16,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        # Encoder specifics from MAE
        self.patch_embed = PatchEmbed1D(grid_size, patch_size, in_chans, history, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.history = history
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embeddings
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch embedding projection
        torch.nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        
        # Initialize tokens
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        # Initialize other layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Embed patches
        x = self.patch_embed(x)
        
        # Add positional embedding (without cls token)
        x = x + self.pos_embed[:, 1:, :]
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x


class DiscoOperator(nn.Module):
    """
    Disco Operator: Combines encoder from MAE1D with neural operator from UNet1D
    
    The architecture consists of:
    1. DiscoEncoder: Encodes input sequences using MAE-like transformer architecture
    2. Feature projection: Maps encoded features to UNet input format
    3. UNet: Neural operator for sequence-to-sequence modeling
    """
    
    def __init__(self, 
                 # Encoder parameters
                 grid_size=256, patch_size=16, in_chans=1, history=16,
                 embed_dim=1024, encoder_depth=12, num_heads=16, mlp_ratio=4.,
                 
                 # UNet parameters
                 n_output_scalar_components=1, n_output_vector_components=0,
                 time_future=1, hidden_channels=64, activation="gelu",
                 norm=False, ch_mults=(1, 2, 2, 2), is_attn=(False, False, False, False),
                 mid_attn=False, n_blocks=2, param_conditioning=None,
                 use_scale_shift_norm=False, use1x1=False, n_dims=1,
                 code_dim=2,
                 
                 # Other parameters
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.history = history
        self.time_future = time_future
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.code_dim = code_dim
        
        # Encoder
        self.encoder = DiscoEncoder(
            grid_size=grid_size,
            patch_size=patch_size, 
            in_chans=in_chans,
            history=history,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )
        
        # Feature projection from encoder to UNet input
        num_patches = grid_size // patch_size
        # Project from (batch, num_patches + 1, embed_dim) to (batch, in_chans * history, grid_size)
        self.feature_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, code_dim)
        )    

        # Neural operator (UNet)
        self.neural_operator = Unet(
            n_input_scalar_components=in_chans,
            n_input_vector_components=0,
            n_output_scalar_components=n_output_scalar_components,
            n_output_vector_components=n_output_vector_components,
            time_history=1,
            time_future=time_future,
            hidden_channels=hidden_channels,
            activation=activation,
            norm=norm,
            ch_mults=ch_mults,
            is_attn=is_attn,
            mid_attn=mid_attn,
            n_blocks=n_blocks,
            param_conditioning=param_conditioning,
            use_scale_shift_norm=use_scale_shift_norm,
            use1x1=use1x1,
            n_dims=n_dims,
            code_dim=code_dim
        )
    
    def forward(self, x, y= None, conditioning=None):
        """
        Forward pass through the Disco Operator
        
        Args:
            x: Input tensor of shape (batch, channels, height, time_history)
            conditioning: Optional conditioning vector from encoder CLS token of shape (batch, embed_dim)
                         If None, will be computed from input x using the encoder
            
        Returns:
            Output tensor of shape (batch, output_channels, height, time_future)
        """
        batch_size, channels, height, time_hist = x.shape
        
        
        if conditioning is None:
            # Encode input using MAE-like encoder to get conditioning
            features = self.encoder(x)  # (batch, num_patches + 1, embed_dim)
            conditioning = features[:, 0, :]  # Extract CLS token: (batch, embed_dim)
            encoded_patches = features[:, 1:, :]  # Remove cls token: (batch, num_patches, embed_dim)
        
        # Project each patch back to original patch size
        code = self.feature_projection(conditioning)  # (batch, num_patches, in_chans * history)

        if y is not None:
            batch_size, channels, height, time_target = y.shape
            y = rearrange(y, 'b c h t -> (b t) c h')
            code = rearrange(code.unsqueeze(-1).repeat(1, 1, time_target), 'b c t -> (b t) c')
            output = self.neural_operator(y, z=code)
            output = rearrange(output, '(b t) c h -> b c h t', t=time_target)
        else:
            x = rearrange(x, 'b c h t -> (b t) c h')
            code = rearrange(code.unsqueeze(-1).repeat(1, 1, time_hist), 'b c t -> (b t) c')
            output = self.neural_operator(x, z=code)
            output = rearrange(output, '(b t) c h -> b c h t', t= time_hist)
        
        return output, conditioning


def disco_operator_mini(**kwargs):
    """Base Disco Operator configuration"""
    model = DiscoOperator(
        embed_dim=128, encoder_depth=4, num_heads=4,
        hidden_channels=64, code_dim=64, norm_layer=partial(nn.LayerNorm, eps=1e-6)) #warning hidden_channels 64 before
    return model

def disco_operator_base(**kwargs):
    """Base Disco Operator configuration"""
    model = DiscoOperator(
        embed_dim=768, encoder_depth=12, num_heads=12,
        hidden_channels=64, code_dim=768, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def disco_operator_large(**kwargs):
    """Large Disco Operator configuration"""
    model = DiscoOperator(
        embed_dim=1024, encoder_depth=24, num_heads=16,
        hidden_channels=128, code_dim=1024, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def disco_operator_huge(**kwargs):
    """Huge Disco Operator configuration"""
    model = DiscoOperator(
        embed_dim=1280, encoder_depth=32, num_heads=16,
        hidden_channels=256, code_dim=1280, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
