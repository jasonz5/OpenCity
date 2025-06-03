import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from typing import Optional, Tuple, List, Union

def load_balancing_loss_func(
        gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
        top_k: int,
        num_experts: int = None,
        attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor], List[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length X num_nodes, num_experts].
        top_k (`int`)
            Selected Top k over the experts.
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length X num_nodes] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, (tuple, list)) or gate_logits[0] is None:
        return 0.0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, 2, num_experts))
            .reshape(-1, 2, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0))

    return overall_loss * num_experts

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(2)].unsqueeze(1).expand_as(x)

class PatchEmbedding_flow(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, his):
        super(PatchEmbedding_flow, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.his = his

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        # do patching
        x = x.squeeze(-1).permute(0, 2, 1)
        if self.his == x.shape[-1]:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            gap = self.his // x.shape[-1]
            x = x.unfold(dimension=-1, size=self.patch_len//gap, step=self.stride//gap)
            x = F.pad(x, (0, (self.patch_len - self.patch_len//gap)))
        x = self.value_embedding(x)
        x = x + self.position_encoding(x)
        x = x.permute(0, 2, 1, 3)
        return x

class PatchEmbedding_time(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, his):
        super(PatchEmbedding_time, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.his = his
        self.minute_size = 1440 + 1
        self.daytime_embedding = nn.Embedding(self.minute_size, d_model//2)
        weekday_size = 7 + 1
        self.weekday_embedding = nn.Embedding(weekday_size, d_model//2)

    def forward(self, x):
        # do patching
        bs, ts, nn, dim = x.size()
        x = x.permute(0, 2, 3, 1).reshape(bs, -1, ts)
        if self.his == x.shape[-1]:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            gap = self.his // x.shape[-1]
            x = x.unfold(dimension=-1, size=self.patch_len//gap, step=self.stride//gap)
        num_patch = x.shape[-2]
        x = x.reshape(bs, nn, dim, num_patch, -1).transpose(1, 3)
        x_tdh = x[:, :, 0, :, 0]
        x_dwh = x[:, :, 1, :, 0]
        x_tdp = x[:, :, 2, :, 0]
        x_dwp = x[:, :, 3, :, 0]

        x_tdh = self.daytime_embedding(x_tdh)
        x_dwh = self.weekday_embedding(x_dwh)
        x_tdp = self.daytime_embedding(x_tdp)
        x_dwp = self.weekday_embedding(x_dwp)
        x_th = torch.cat([x_tdh, x_dwh], dim=-1)
        x_tp = torch.cat([x_tdp, x_dwp], dim=-1)

        return x_th, x_tp

class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, prob_drop, alpha):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
        self.mlp = nn.Linear(out_dim, out_dim)
        self.dropout = prob_drop
        self.alpha = alpha

    def forward(self, x, adj):
        d = adj.sum(1)
        h = x
        a = adj / d.view(-1, 1)
        gcn_out = self.fc1(torch.einsum('bdkt,nk->bdnt', h, a))
        out = self.alpha*x + (1-self.alpha)*gcn_out
        ho = self.mlp(out)
        return ho

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()

        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoeMLP(FeedForward):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__(hidden_size, intermediate_size)

    def forward(self, hidden_state):
        return super().forward(hidden_state), None


class MoeSparseExpertsLayer(nn.Module):
    def __init__(self, num_experts_per_tok, num_experts, hidden_size, intermediate_size ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.norm_topk_prob = False

        moe_intermediate_size = intermediate_size // self.top_k

        # gating
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(
                hidden_size=self.hidden_size,
                intermediate_size=moe_intermediate_size,
            ) for _ in range(self.num_experts)]
        )

        self.shared_expert = FeedForward(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
        )
        self.shared_expert_gate = torch.nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, num_nodes, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits -> (batch * sequence_length * num_nodes, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length * num_nodes, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, num_nodes, hidden_dim)
        return final_hidden_states, router_logits



class TemporalSelfAttention(nn.Module):
    def __init__(
        self, dim, t_num_heads=6, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'),
    ):
        super().__init__()
        assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device

        self.t_q_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_k_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_v_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.norm_tatt = LlamaRMSNorm(dim)

        self.GCN = GCN(dim, dim, proj_drop, alpha=0.05)
        self.act = nn.GELU()

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_k, x_v, adj, geo_mask=None, sem_mask=None, trg_mask=False):
        B, T_q, N, D = x_q.shape
        T_k, T_v = x_k.shape[1], x_v.shape[1]

        t_q = self.t_q_conv(x_q).transpose(1, 2)
        t_k = self.t_k_conv(x_k).transpose(1, 2)
        t_v = self.t_v_conv(x_v).transpose(1, 2)
        t_q = t_q.reshape(B, N, T_q, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T_k, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T_v, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        if trg_mask:
            ones = torch.ones_like(t_attn).to(self.device)
            dec_mask = torch.triu(ones, diagonal=1)
            t_attn = t_attn.masked_fill(dec_mask == 1, -1e9)
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T_q, D).transpose(1, 2)

        t_x = self.norm_tatt(t_x)
        gcn_out = self.GCN(t_x, adj)
        x = self.proj_drop(gcn_out)
        return x


class STEncoderBlock(nn.Module):
    def __init__(
        self, dim, t_num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, device=torch.device('cpu'), type_ln="pre", output_dim=1, mlp_use_dense = True, num_experts_per_tok=1, num_experts=1
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = LlamaRMSNorm(dim)
        self.norm2 = LlamaRMSNorm(dim)
        self.st_attn = TemporalSelfAttention(dim, t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mlp_use_dense:
            self.mlp = MoeMLP(hidden_size=dim, intermediate_size=mlp_hidden_dim)
        else:
            self.mlp = MoeSparseExpertsLayer(num_experts_per_tok=num_experts_per_tok, num_experts=num_experts,
                hidden_size=dim, intermediate_size=mlp_hidden_dim)


    def forward(self, x, k, v, adj, geo_mask=None, sem_mask=None):
        if self.type_ln == 'pre':
            x_nor1 = self.norm1(x)
            k_nor1 = self.norm1(k)
            x = x + self.drop_path(self.st_attn(x_nor1, k_nor1, k_nor1, adj, geo_mask=geo_mask, sem_mask=sem_mask))
            hidden_states, router_logits = self.mlp(self.norm2(x))
            x = x + self.drop_path(hidden_states)
        elif self.type_ln == 'post':
            x = self.norm1((x + self.drop_path(self.st_attn(x, k, k, adj, geo_mask=geo_mask, sem_mask=sem_mask))))
            hidden_states, router_logits = self.mlp(x)
            x = self.norm2((x + self.drop_path(hidden_states)))
        else:
            x = x + self.drop_path(self.st_attn(x, k, k, adj, geo_mask=geo_mask, sem_mask=sem_mask))
            hidden_states, router_logits = self.mlp(self.norm2(x))
            x = x + self.drop_path(hidden_states)
        return x, router_logits


class CityMoE(nn.Module):
    def __init__(self, args, dataset_use, device, dim_in):
        super(CityMoE, self).__init__()

        self.feature_dim = dim_in
        self.adj_mx_dict = args.adj_mx_dict
        self.sh_mx_dict = args.sh_mx_dict
        self.lap_mx_dict = args.lap_mx_dict

        self.embed_dim = args.embed_dim
        self.skip_dim = args.skip_dim
        self.lape_dim = args.lape_dim
        self.t_num_heads = args.t_num_heads
        self.mlp_ratio = args.mlp_ratio
        self.qkv_bias = args.qkv_bias
        self.drop = args.drop
        self.attn_drop = args.attn_drop
        self.drop_path = args.drop_path
        self.enc_depth = args.enc_depth
        self.type_ln = args.type_ln

        self.mlp_use_dense = args.mlp_use_dense
        self.num_experts_per_tok = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.apply_aux_loss = args.apply_aux_loss
        self.router_aux_loss_factor = args.router_aux_loss_factor

        self.output_dim = dim_in
        self.input_window = args.input_window
        self.output_window = args.output_window
        self.device = device
        self.far_mask_delta = args.far_mask_delta

        self.geo_mask_dict = {}
        for i, data_graph in enumerate(dataset_use):
            sh_mx = self.sh_mx_dict[data_graph].T
            self.geo_mask_dict[data_graph] = torch.zeros_like(sh_mx)
            self.geo_mask_dict[data_graph][sh_mx >= self.far_mask_delta] = 1
            self.geo_mask_dict[data_graph] = self.geo_mask_dict[data_graph].bool()
        self.sem_mask = None

        self.patch_embedding_flow = PatchEmbedding_flow(
            self.embed_dim, patch_len=12, stride=12, padding=0, his=args.input_window)
        self.patch_embedding_time = PatchEmbedding_time(
            self.embed_dim, patch_len=12, stride=12, padding=0, his=args.input_window)
        self.spatial_embedding = LaplacianPE(self.lape_dim, self.embed_dim)

        self.liner_enc_his = nn.Linear(2*self.embed_dim, self.embed_dim, bias=False)
        self.liner_enc_pre = nn.Linear(2*self.embed_dim, self.embed_dim, bias=False)

        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, t_num_heads=self.t_num_heads,
                mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop, attn_drop=self.attn_drop, drop_path=enc_dpr[i], act_layer=nn.GELU,
                device=self.device, type_ln=self.type_ln, output_dim=self.output_dim,
                mlp_use_dense = self.mlp_use_dense, num_experts_per_tok = self.num_experts_per_tok, num_experts=self.num_experts  
            ) for i in range(self.enc_depth)
        ])

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(24*self.skip_dim, self.output_window)


    def forward(self, input, lbls, select_dataset):

        bs, time_steps, num_nodes, num_feas = input.size()
        x = input
        # Spatio-Temporal Context Encoding
        TCH = input[..., self.output_dim:].long()
        TCP = lbls[..., self.output_dim:].long()
        feas_all_his, feas_all_pre = self.patch_embedding_time(torch.cat([TCH, TCP], dim=-1))
        spa_feas = self.spatial_embedding(self.lap_mx_dict[select_dataset].to(self.device)).repeat(bs, feas_all_his.shape[1], 1, 1)
        feas_all_his = feas_all_his + spa_feas
        feas_all_pre = feas_all_pre +spa_feas # torch.Size([16, 24, 228, 128])

        # IN
        x_in = x[..., :self.output_dim]
        means = x_in.mean(1, keepdim=True).detach()
        x_in = x_in - means
        stdev = torch.sqrt(torch.var(x_in, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()
        x_in /= stdev

        # Patch Embedding
        enc = self.patch_embedding_flow(x_in) # torch.Size([16, 24, 228, 128])

        # adj
        adj = self.adj_mx_dict[select_dataset].to(self.device)

        # 融入 pre/his的 ST-Context到 query/key中
        enc_his = self.liner_enc_his(torch.cat([enc, feas_all_his], dim=-1))
        enc_pre = self.liner_enc_pre(torch.cat([enc, feas_all_his], dim=-1))
        # Spatio-Temporal Dependencies Modeling
        all_router_logits = ()
        for i, encoder_block in enumerate(self.encoder_blocks):
            if i == 0:
                enc, router_logits  = encoder_block(enc_pre, enc_his, enc_his, adj, self.geo_mask_dict[select_dataset].to(self.device), self.sem_mask)
            else:
                enc, router_logits = encoder_block(enc, enc, enc, adj, self.geo_mask_dict[select_dataset].to(self.device), self.sem_mask)
            all_router_logits += (router_logits,)

        # Prediction head
        skip = enc.permute(0, 2, 3, 1).contiguous()
        skip = self.flatten(skip)
        skip = self.linear(skip).transpose(1, 2).unsqueeze(-1)
        skip = skip[:, :time_steps, :, :]

        # DeIN
        skip = skip * stdev
        skip = skip + means
        # import ipdb; ipdb.set_trace()
        if all_router_logits and self.apply_aux_loss:
            temporal_aux_loss = load_balancing_loss_func(
                all_router_logits,
                top_k=self.num_experts_per_tok,
                num_experts=self.num_experts,
                attention_mask=None
            )
            router_aux_loss = self.router_aux_loss_factor * temporal_aux_loss

        return skip, router_aux_loss