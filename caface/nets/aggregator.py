import torch
from torch import nn
from einops import rearrange, repeat
from nets.styler import StyleMergeLayer
from nets.transformer import transformer_module


class SimpleAvgAggregator(nn.Module):
    def __init__(self):
        super(SimpleAvgAggregator, self).__init__()

    def compress_intermediate(self, intermediate):
        assert intermediate.ndim == 4  # B, C, H, W
        B, C, H, W = intermediate.shape
        return torch.ones((B, C), device=intermediate.device, dtype=intermediate.dtype)

    def forward(self, x, intermediate):
        x = x.mean(dim=1)
        return x


class StyleBasedAggregator(nn.Module):
    '''
    This is a wrapper around Style Input Maker (SIM), Cluster Network (CN) and Aggregation Network (AGN).
    And it is also responsible for handling the sequential update rule.
    The variable names:
    SIM : self.style_maker
    CN + AGN : self.fusion_net
    '''
    def __init__(self,
                 fusion_net, intermediate_indim, style_dim, input_maker_outdim,
                 combine_norm_emb=False, normemb_dim=32, use_memory=False):

        super(StyleBasedAggregator, self).__init__()

        self.style_maker = StyleMergeLayer(intermediate_indim, style_dim)
        self.fusion_net = fusion_net

        input_dim = style_dim
        self.norm_emb = None
        if combine_norm_emb:
            input_dim = input_dim + normemb_dim
            self.norm_emb = transformer_module.SinusoidalEncoding(d_model=normemb_dim, clip_val=3, clip_multi=10)
        if input_maker_outdim == 'same':
            input_maker_outdim = input_dim

        if input_dim == input_maker_outdim:
            self.input_maker = nn.Identity()
        else:
            self.input_maker = transformer_module.Mlp(input_dim, input_maker_outdim, input_maker_outdim)

        self.reset_memory()
        self.use_memory = use_memory


    def reset_memory(self):
        self.memory_count = -1
        self.memory = None

    def compress_intermediate(self, intermediate):
        concat_intermediate = torch.cat(intermediate,dim=1)
        compact_intermediate = self.style_maker(concat_intermediate)
        return compact_intermediate

    def forward(self, folded_features, intermediate, return_attn=True):
        assert folded_features.ndim == 3
        assert intermediate.ndim == 3  # compact

        norm = torch.norm(folded_features, 2, -1, keepdim=True)
        latent_code = folded_features / (norm+1e-5)

        compact_intermediate = intermediate
        if self.norm_emb is not None:
            norm_embed = self.norm_emb(norm)
            compact_intermediate = torch.cat([compact_intermediate, norm_embed], dim=-1)

        inputs = self.input_maker(compact_intermediate)

        if self.memory_count < 1:
            # reset the memory for memory_count=0
            self.memory = None

        result = self.fusion_net(inputs,
                                 latent_code=latent_code,
                                 return_attn=return_attn,
                                 memory=self.memory)
        # keys: [aggregated, mu, conf, cluster_feature, group_token, attn_dict_list]

        if self.memory_count > -1 and self.use_memory:
            # memory tracking on
            self.memory_count = self.memory_count + 1

            if self.training:
                new_att_sum = result['attn_dict_list'][0]['attn'].squeeze(1).sum(-1, keepdim=True)
                cluster_feature = result['cluster_feature']
                if 'cluster_style' in result:
                    cluster_style = result['cluster_style']
                else:
                    cluster_style = None
            else:
                new_att_sum = result['attn_dict_list'][0]['attn'].squeeze(1).sum(-1, keepdim=True).detach().clone()
                cluster_feature = result['cluster_feature'].detach().clone()
                if 'cluster_style' in result:
                    cluster_style = result['cluster_style'].detach().clone()
                else:
                    cluster_style = None

            if self.memory is not None:
                self.memory = {'attn': new_att_sum + self.memory['attn'],
                               'cluster_features': cluster_feature,
                               'cluster_style': cluster_style}
            else:
                self.memory = {'attn': new_att_sum,
                               'cluster_features': cluster_feature,
                               'cluster_style': cluster_style}

        if 'weights' not in result:
            weights = result['conf'] / result['conf'].sum(dim=1, keepdim=True)
            result['weights'] = weights

        return result

def incremental_mean(prev_val, prev_count, new_val, new_count=1):
    return ((prev_count * prev_val) + (new_count * new_val)) / (prev_count + new_count)


def build_aggregator(model_name='', fusion_net=None, style_index=[],
                     input_maker_outdim=None, use_memory=False, normemb_dim=32):
    if model_name == 'simple_avg':
        return SimpleAvgAggregator()
    elif model_name.startswith('style'):
        intermediate_indim = get_styledim(style_index)
        style_dim = 64
        combine_normemb = True

        return StyleBasedAggregator(fusion_net=fusion_net,
                                    intermediate_indim=intermediate_indim,
                                    style_dim=style_dim,
                                    input_maker_outdim=input_maker_outdim,
                                    combine_norm_emb=combine_normemb,
                                    use_memory=use_memory,
                                    normemb_dim=normemb_dim,
                                    )
    else:
        raise ValueError('not implented yet')


def get_styledim(style_index):
    assert isinstance(style_index, str)
    cdim = 0
    if '1' in style_index:
        cdim += 64
    if '2' in style_index:
        cdim += 56 * 56
    if '3' in style_index:
        cdim += 128
    if '4' in style_index:
        cdim += 28 * 28
    if '5' in style_index:
        cdim += 256
    if '6' in style_index:
        cdim += 14 * 14
    if '7' in style_index:
        cdim += 512
    if '8' in style_index:
        cdim += 7 * 7
    return cdim


if __name__ == '__main__':

    agg = build_aggregator(model_name='pfe')
    B = 5
    T = 10
    C = 512
    x = torch.randn([B, T, C], requires_grad=True)
    intermediate = torch.randn([B, T, C, 7, 7], requires_grad=True)
    raveled_intermediate = rearrange(intermediate, 'b t c h w -> (b t) c h w')
    compressed_intermediate = agg.compress_intermediate(raveled_intermediate)
    compressed_intermediate = rearrange(compressed_intermediate, '(b t) c -> b t c', t=T)
    out = agg(folded_features=x, intermediate=compressed_intermediate)
    for key, val in out.items():
        print(key)
        print(val.shape)
    # mu # torch.Size([5, 10, 512])
    # conf # torch.Size([5, 10, 512])
    # aggregated # torch.Size([5, 512])