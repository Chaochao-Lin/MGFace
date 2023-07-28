import torch.utils.checkpoint as checkpoint
import torch
import torch.nn as nn
from .swin_transformer import BasicLayer, PatchEmbed, PatchMerging
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange


# 放大特征图尺寸，加大通道
class PatchExpanding(nn.Module):
    '''
    Patch Expanding Layer.Upsample
    [B, H*W, C] expand(Linear) -> [B, H*W, 2*C] view+rearrange+view -> [B, H*2*W*2, C//2]
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        dim_scale: Expand dimention scale.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    Returns:
        x = norm(rearrange(expand(x)))
    '''

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        '''
        x: [B, H*W, C]
        '''
        # print('PatchExpanding', x.shape)
        H, W = self.input_resolution
        x = self.expand(x)  # [B, H*W, 2*C]
        B, L, C = x.shape  # [B, H*W, 2*C]
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)  # [B, H*2*W*2, C//2]
        x = self.norm(x)
        # print('PatchExpanding end', x.shape)

        return x


class MBasicLayer(BasicLayer):
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        pre_downsample = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, pre_downsample


class MSwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        if num_classes > 0:
            # split image into non-overlapping patches
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
            num_patches = self.patch_embed.num_patches
            patches_resolution = self.patch_embed.patches_resolution
            self.patches_resolution = patches_resolution

            # absolute position embedding
            if self.ape:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
                trunc_normal_(self.absolute_pos_embed, std=.02)

            self.pos_drop = nn.Dropout(p=drop_rate)
        else:
            self.patch_embed = None
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
            self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        h, w = patches_resolution[0], patches_resolution[1]
        for i_layer in range(self.num_layers):
            is_downsample = (h % 2 == 0 and w % 2 == 0)
            downsample = None
            if i_layer < self.num_layers - 1:
                dim = int(embed_dim * 2 ** i_layer)
                if is_downsample:
                    downsample = PatchMerging((h, w), dim=dim, norm_layer=norm_layer)
                else:
                    downsample = nn.Linear(dim, dim * 2)
            layer = MBasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                input_resolution=(h, w),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=downsample,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layer)
            if is_downsample:
                h, w = h // 2, w // 2

        self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features * h * w, num_classes) if num_classes > 0 else None
        self.features = nn.BatchNorm1d(num_classes) if num_classes > 0 else None
        # nn.init.constant_(self.features.weight, 1.0)
        # self.features.weight.requires_grad = False

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def pre_forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x

    @classmethod
    def resize(cls, x, from_shape, to_shape):
        # 改变merge后的图片尺寸
        B, H, W, C = from_shape
        x = x.view(B, H, W, C)
        x = nn.functional.interpolate(x.permute(0, 3, 1, 2), size=(to_shape[1], to_shape[2]), mode="bilinear",
                                      align_corners=True)
        B, H, W, C = to_shape
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)
        return x

    def forward_features(self, x, mask=None):
        if mask is None:
            pre_down_sample_list = []
            for layer in self.layers:
                x, pre_down_sample = layer(x)
                pre_down_sample_list.append(pre_down_sample)
            return x, pre_down_sample_list
        else:
            for layer in self.layers:
                H, W = layer.input_resolution
                B, L, C = x.shape
                assert H * W == L
                x, _ = layer(x * self.resize(mask, (B, 56, 56, 1), (B, H, W, 1)))
            return x
            # return x * self.resize(mask, (B, 56, 56, 1), (B, H, W, 1))

    def last_forward(self, x):
        x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        x = self.head(x)
        x = self.features(x)
        return x

    def forward(self, x):
        if self.patch_embed is not None:
            x = self.pre_forward(x)
        x, _ = self.forward_features(x)
        if self.num_classes > 0:
            x = self.last_forward(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class STMGFACE(MSwinTransformer):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 use_checkpoint=False, mask_method=None, **kwargs):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depths, num_heads, window_size,
                         mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm,
                         use_checkpoint, **kwargs)
        self.mask_method = mask_method  # 遮罩方式【None，last，all】
        self.re_encoder = MSwinTransformer(img_size, patch_size, in_chans, 0, embed_dim, depths=depths, num_heads=num_heads, window_size=window_size)
        self.num_layers = len(depths)

        encoder_list = self.layers

        def create_up_sample_and_fc(i_layer):
            up_sample = PatchExpanding(encoder_list[i_layer].input_resolution, encoder_list[i_layer].dim, norm_layer=norm_layer)
            fc = nn.Sequential(
                nn.Linear(encoder_list[i_layer].dim, encoder_list[i_layer-1].dim, bias=False),
                nn.LayerNorm(encoder_list[i_layer-1].dim),
            )
            return up_sample, fc

        def create_up_sample_and_decoder(i_layer):
            up_sample, fc = create_up_sample_and_fc(i_layer)
            decoder = nn.Sequential(
                fc,
                BasicLayer(dim=encoder_list[i_layer-1].dim,
                           input_resolution=encoder_list[i_layer-1].input_resolution,
                           depth=2,
                           num_heads=4,
                           window_size=window_size,
                           mlp_ratio=self.mlp_ratio,
                           qkv_bias=qkv_bias,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=0.,
                           norm_layer=norm_layer,
                           use_checkpoint=False)
            )
            return up_sample, decoder
        self.up_sample_3, self.decoder_3 = create_up_sample_and_decoder(3)
        self.up_sample_2, self.decoder_2 = create_up_sample_and_decoder(2)
        self.up_sample_1, self.decoder_1 = create_up_sample_and_decoder(1)

        self.mask_fc = nn.Linear(encoder_list[0].dim, 2)
        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def forward(self, x):
        x = self.pre_forward(x)
        _, encode_list = self.forward_features(x)
        e0, e1, e2, e3 = encode_list

        # unet解码，以计算mask的loss--------------------
        d3 = self.up_sample_3(e3)  # B 14*14*256
        d3 = torch.cat((e2, d3), dim=2)
        d3 = self.decoder_3(d3)  # B 14*14 256

        d2 = self.up_sample_2(d3)  # B 28*28 128
        d2 = torch.cat((e1, d2), dim=2)
        d2 = self.decoder_2(d2)  # B 28*28 128

        d1 = self.up_sample_1(d2)  # B 56*56 64
        d1 = torch.cat((e0, d1), dim=2)
        d1 = self.decoder_1(d1)  # B 56*56 64

        # 输出mask--------------------------------------
        mask_pred = self.mask_fc(d1)

        mask_pred_softmax = torch.softmax(mask_pred, dim=2)
        mask_attention_map = torch.exp(mask_pred_softmax[:, :, 1:])

        if self.mask_method is None:
            re_e3 = e3
        elif self.mask_method == "all":
            re_e3 = self.forward_features(x, mask_attention_map)
        elif self.mask_method == "last":
            B, L, C = e3.shape
            re_e3 = e3 * self.resize(mask_attention_map, (B, 56, 56, 1), (B, 7, 7, 1))
        else:
            raise Exception()

        # 输出
        feature = self.last_forward(re_e3)
        if not self.training:
            return mask_pred
            # return feature
        return mask_pred, feature

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                print("copy value to %s" % name)
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
