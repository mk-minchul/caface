import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import random
from nets.transformer import transformer_module
from nets.transformer import tconfig


class L2Norm(nn.Module):

    def __init__(self, channel=None, dim=-1, eps=1e-5):
        super(L2Norm, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=self.dim, keepdim=True) + self.eps)

class AssignAttention(nn.Module):

    def __init__(self,
                 dim,
                 value_dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn):
        # attention dimension is -2 because it is assignment of queries into centers
        attn_dim = -2
        attn = F.softmax(attn, dim=attn_dim)
        return attn

    def forward(self, query, key=None, *, value=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v_chan = value.shape[-1]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=v_chan // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn)
            attn_dict = {'attn': attn, 'hard': hard_attn, 'soft': soft_attn}
        else:
            attn_dict = {'attn': attn}

        div = torch.clip(attn.sum(dim=-1, keepdim=True), 1, torch.inf)
        attn = attn / div.detach()
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=v_chan // self.num_heads)

        # summarize style as well
        _key = rearrange(key, 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=key.shape[2] // self.num_heads)
        out_style = rearrange(attn @ _key, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=key.shape[2] // self.num_heads)

        return out, attn_dict, out_style

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'assign_eps: {self.assign_eps}'


class ClusteringBlock(nn.Module):

    def __init__(self,
                 *,
                 dim,
                 value_dim,
                 norm_layer,
                 sum_assign=False,
                 assign_eps=1.,
                 ):
        super(ClusteringBlock, self).__init__()
        self.dim = dim
        self.sum_assign = sum_assign
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        # norm on x
        self.norm_x = norm_layer(dim)

        self.assign = AssignAttention(
            dim=dim,
            value_dim=value_dim,
            num_heads=1,
            qkv_bias=True,
            sum_assign=sum_assign,
            assign_eps=assign_eps)

    def extra_repr(self):
        return f'Attention, \n' \
               f'sum_assign={self.sum_assign}, \n'

    def project_group_token(self, group_tokens):
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, x, group_tokens, values=None, return_attn=False):
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)
        new_value, attn_dict, new_query = self.assign(group_tokens, x, value=values, return_attn=return_attn)

        return new_value, attn_dict, new_query



class ClusteringLayer(nn.Module):


    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 num_group_token,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 cluster_block=None,
                 use_checkpoint=False,
                 group_projector=None,
                 key_query_combined_info_propagate=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_group_token = num_group_token
        if num_group_token > 0:
            # self.group_token = nn.Parameter(torch.zeros(1, num_group_token, dim))
            x = torch.randn(num_group_token, dim)
            nn.init.orthogonal_(x)
            x = x / torch.norm(x, 2, -1, keepdim=True)
            x = x.unsqueeze(0)
            self.group_token = nn.Parameter(x)
        else:
            self.group_token = None

        # build blocks
        self.depth = depth
        blocks = []
        for i in range(depth):
            blocks.append(
                transformer_module.AttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)

        self.cluster_block = cluster_block
        self.use_checkpoint = use_checkpoint
        self.key_query_combined_info_propagate = key_query_combined_info_propagate

        self.group_projector = group_projector

    @property
    def with_group_token(self):
        return self.group_token is not None

    def extra_repr(self):
        return f'dim={self.dim}, \n' \
               f'depth={self.depth}, \n' \
               f'key_query_combined_info_propagate={self.key_query_combined_info_propagate}, \n' \
               f'num_group_token={self.num_group_token}, \n'

    def split_x(self, x):
        if self.with_group_token:
            return x[:, :-self.num_group_token], x[:, -self.num_group_token:]
        else:
            return x, None

    def concat_x(self, x, group_token=None):
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=1)

    def forward(self, x, prev_group_token=None, values=None, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            prev_group_token (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention maps
        """
        if self.with_group_token:
            group_token = self.group_token.expand(x.size(0), -1, -1)
            if self.group_projector is not None:
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            group_token = None

        B, L, C = x.shape
        if self.key_query_combined_info_propagate:
            cat_x = self.concat_x(x, group_token)
        else:
            cat_x = x

        # info propagate
        for blk_idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                cat_x = checkpoint.checkpoint(blk, cat_x)
            else:
                cat_x = blk(cat_x)

        if self.key_query_combined_info_propagate:
            x, new_group_token = self.split_x(cat_x)
            # we are not using the new group token to prevent group meaning from changing
        else:
            x = cat_x

        new_value, attn_dict, new_query = self.cluster_block(x, group_token, values=values, return_attn=return_attn)

        return new_value, group_token, attn_dict, new_query


class ClusterAggregateTransformer(nn.Module):

    '''
    This module contains Cluster Network (CN) and Aggregation Network (AGN).
    The variable names:
    CN : self.clustering_layer
    AGN : self.conf_mlp
    '''

    def __init__(self,
                 norm='layer',
                 embed_dim=128,
                 value_dim=512,
                 depths=[3],
                 num_heads=[8],
                 num_group_tokens=[4],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 key_query_combined_info_propagate=False,
                 confidence_dim=1,
                 random_group_discard=0.0
                 ):
        super().__init__()

        assert len(depths) == len(num_group_tokens)
        assert all(_ == 0 for _ in num_heads) or len(depths) == len(num_heads)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_group_tokens = num_group_tokens
        self.confidence_dim = confidence_dim
        self.random_group_discard = random_group_discard

        if norm == 'layer':
            norm_layer = nn.LayerNorm
            print('using layer Norm')
        elif norm == 'l2':
            norm_layer = L2Norm
            print('using L2 Norm')
        else:
            raise ValueError('not a correct norm')
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        i_layer = 0  # only 1 layer in Cluster and Aggregate
        dim = embed_dim

        cluster_block = ClusteringBlock(
            dim=dim,
            value_dim=value_dim,
            norm_layer=norm_layer,
            assign_eps=0.0001)

        self.clustering_layer = ClusteringLayer(
            dim=dim,
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            num_group_token=num_group_tokens[i_layer],
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            norm_layer=norm_layer,
            cluster_block=cluster_block,
            use_checkpoint=use_checkpoint,
            group_projector=None,
            key_query_combined_info_propagate=key_query_combined_info_propagate
        )

        self.conf_mlp = []
        for _ in range(depths[0]):
            self.conf_mlp.append(transformer_module.MixerBlock(tokens_mlp_dim=num_group_tokens[0],
                                                               channels_mlp_dim=embed_dim+embed_dim,
                                                               tokens_hidden_dim=num_group_tokens[0]*2,
                                                               channels_hidden_dim=(embed_dim+embed_dim)//2,
                                                               gated=True))
        self.conf_mlp.append(transformer_module.Mlp(embed_dim+embed_dim, embed_dim+embed_dim, confidence_dim))
        self.conf_mlp = nn.Sequential(*self.conf_mlp)

        self.apply(self._init_weights)


    @property
    def width(self):
        return self.num_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, style_features, latent_code, return_attn=False, memory=None):
        group_token = None
        attn_dict_list = []

        # prepare clustering inputs (CN)
        cluster_feature, group_token, attn_dict, cluster_style = self.clustering_layer(style_features,
                                                                                       group_token,
                                                                                       values=latent_code,
                                                                                       return_attn=return_attn)
        attn_dict_list.append(attn_dict)

        # if memery, incremental update
        if memory is not None:
            cluster_feature = self.add_memory(cluster_feature, attn_dict['attn'],
                                              memory['attn'], memory['cluster_features'])
            cluster_style = self.add_memory(cluster_style, attn_dict['attn'],
                                            memory['attn'], memory['cluster_style'])

        # aggregation (AGN)
        confidence_weight = self.conf_mlp(torch.cat([group_token, cluster_style], dim=-1))
        confidence_weight = F.softmax(confidence_weight, dim=-2)

        mu = cluster_feature

        if random.random() < self.random_group_discard and self.training:
            B,T,_ = group_token.shape
            mask = torch.ones((B,T,1), dtype=group_token.dtype, device=group_token.device)
            mask = mask.uniform_() > (1/T)
            masked_confidence_weight = confidence_weight * mask
        else:
            masked_confidence_weight = confidence_weight

        aggregated = (mu * masked_confidence_weight).sum(-2)

        # actual confidence with norm considered
        conf = confidence_weight * torch.norm(mu, 2, -1, keepdim=True)

        result = {}
        result['group_token'] = group_token
        result['aggregated'] = aggregated
        result['attn_dict_list'] = attn_dict_list
        result['cluster_feature'] = cluster_feature
        result['cluster_style'] = cluster_style
        result['mu'] = mu
        result['conf'] = conf
        result['confidence_weight'] = confidence_weight
        return result

    def add_memory(self, x, attn, prev_count, prev_val):
        new_val = x
        new_val_weight = attn.squeeze(1).sum(-1, keepdim=True)
        new_x = incremental_mean(prev_val, prev_count, new_val, new_val_weight)
        return new_x


def incremental_mean(prev_mean, prev_count, new_val, new_val_weight=1):
    return ((prev_count * prev_mean) + (new_val_weight * new_val)) / torch.clip(prev_count + new_val_weight, 1e-6, torch.inf)


def make_model(name, hparams):
    config = tconfig.return_config(name)
    style_dim = 64
    style_dim = style_dim + config['norm_dim']
    embed_dim = style_dim

    model = ClusterAggregateTransformer(
        embed_dim=embed_dim,
        depths=config['depths'],
        num_heads=config['num_heads'],
        num_group_tokens=config['num_group_tokens'],
        key_query_combined_info_propagate=config['key_query_combined_info_propagate'],
        confidence_dim=config['confidence_dim'],
        random_group_discard=config['random_group_discard']
    )
    model.config = config
    return model


if __name__ == '__main__':

    ctran = ClusterAggregateTransformer(
        depths=[3, 3],
        num_group_tokens=[64, 0],
        key_query_combined_info_propagate=True)

    out = ctran(style_features=torch.randn(4, 100, 384),
                latent_code=torch.randn(4, 100, 512),
                return_attn=True)
