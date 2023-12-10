import torch
import torch.nn as nn

import math
import warnings
from typing import Optional, Tuple
from einops import rearrange

def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor

class LPLayerNorm(torch.nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)

def rms_norm(x, weight=None, eps=1e-05):
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output

class RMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)

class LPRMSNorm(RMSNorm):

    def __init__(self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, weight=weight, dtype=dtype, device=device)

    def forward(self, x):
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        with torch.autocast(enabled=False, device_type=x.device.type):
            return rms_norm(downcast_x, downcast_weight, self.eps).to(dtype=x.dtype)
        
NORM_CLASS_REGISTRY = {'layernorm': torch.nn.LayerNorm, 'low_precision_layernorm': LPLayerNorm, 'rmsnorm': RMSNorm, 'low_precision_rmsnorm': LPRMSNorm}

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.intermediate_size = config.feed_forward_dim
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):

        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

def scaled_multihead_dot_product_attention(query, key, value, q_n_heads, kv_n_heads, attn_bias=None, key_padding_mask=None, is_causal=False, needs_weights=False):
    q = rearrange(query, 'b s (h d) -> b h s d', h=q_n_heads)
    k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
    v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)

    if q_n_heads != kv_n_heads:
        k = repeat_kv(k, q_n_heads//kv_n_heads)
        v = repeat_kv(v, q_n_heads//kv_n_heads)

    (b, _, s_q, d) = q.shape
    s_k = k.size(-1)
    softmax_scale = 1 / math.sqrt(d)
    attn_weight = q.matmul(k) * softmax_scale
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - s_q)
        _s_k = max(0, attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
        if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q):
            raise RuntimeError(f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.')
        attn_weight = attn_weight + attn_bias
    min_val = torch.finfo(q.dtype).min
    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn('Propogating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unneccessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
        attn_weight = attn_weight.masked_fill(~key_padding_mask.view((b, 1, 1, s_k)), min_val)
    if is_causal and (not q.size(2) == 1):
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    out = attn_weight.to(v.dtype).matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')
    if needs_weights:
        return (out, attn_weight)
    return (out, None)

def build_attn_bias(attn_impl, attn_bias, n_heads, seq_len, causal=False, alibi_bias_max=8):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        (device, dtype) = (attn_bias.device, attn_bias.dtype)
        attn_bias = attn_bias.add(build_alibi_bias(n_heads, seq_len, full=not causal, alibi_bias_max=alibi_bias_max, device=device, dtype=dtype))
        return attn_bias
    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')

def build_alibi_bias(n_heads, seq_len, full=False, alibi_bias_max=8, device=None, dtype=None):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, 1, seq_len)
    if full:
        alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)
    slopes = gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)

def attn_bias_shape(attn_impl, n_heads, seq_len, causal):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if not causal:
            return (1, n_heads, seq_len, seq_len)
        return (1, n_heads, 1, seq_len)

    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')

def gen_slopes(n_heads, alibi_bias_max=8, device=None):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = 1.0 / torch.pow(2, m)
    if _n_heads != n_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes.view(1, n_heads, 1, 1)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

class MultiheadAttention(nn.Module):

    def __init__(self, 
                 d_model: int, 
                 q_n_heads: int,
                 kv_n_heads: int, 
                 attn_impl: str='torch',
                 verbose: int=0, 
                 device: Optional[str]=None):
        super().__init__()
        self.attn_impl = attn_impl

        self.d_model = d_model
        self.head_dim = d_model // q_n_heads
        self.q_n_heads = q_n_heads
        self.kv_n_heads = kv_n_heads

        self.Wqkv = nn.Linear(d_model, d_model + 2 * self.head_dim * self.kv_n_heads, device=device)
        fuse_splits = (d_model, d_model + self.head_dim * self.kv_n_heads)
        self.Wqkv._fused = (0, fuse_splits)

        self.Wq = nn.Linear(d_model, d_model, device=device)
        self.Wk = nn.Linear(d_model, self.head_dim * self.kv_n_heads, device=device)
        self.Wv = nn.Linear(d_model, self.head_dim * self.kv_n_heads, device=device)

        if self.attn_impl == 'flash':
            raise Exception("Not implement yet")
            # self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            raise Exception("Not implement yet")
            # self.attn_fn = triton_flash_attn_fn
            # if verbose:
            #     warnings.warn('While `attn_impl: triton` can be faster than `attn_impl: flash` ' + 'it uses more memory. When training larger models this can trigger ' + 'alloc retries which hurts performance. If encountered, we recommend ' + 'using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.')
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn('Using `attn_impl: torch`. If your model does not use `alibi` or ' + '`prefix_lm` we recommend using `attn_impl: flash` otherwise ' + 'we recommend using `attn_impl: triton`.')
        else:
            raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True

    def forward(self, q, kv, attn_bias=None, attention_mask=None, is_causal=True, needs_weights=False):
        
        # qkv = self.Wqkv(x)
        # (query, key, value) = qkv.split([self.d_model, self.head_dim * self.kv_n_heads, self.head_dim * self.kv_n_heads], dim=2)
        
        query = self.Wq(q)
        key = self.Wk(kv)
        value = self.Wv(kv)
        
        key_padding_mask = attention_mask

        (context, attn_weights) = self.attn_fn(query, key, value, self.q_n_heads, self.kv_n_heads, attn_bias=attn_bias, key_padding_mask=key_padding_mask, is_causal=is_causal, needs_weights=needs_weights)
        return (self.out_proj(context), attn_weights)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config, device: Optional[str]=None):
        super(TransformerEncoderLayer, self).__init__()

        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]
        attn_class = MultiheadAttention

        d_model = config.d_model
        q_n_heads = config.q_n_heads
        kv_n_heads = config.kv_n_heads

        self.norm_1 = norm_class(d_model, device=device)
        self.self_attn = attn_class(d_model=d_model, q_n_heads=q_n_heads, kv_n_heads=kv_n_heads, attn_impl=config.attn_impl, device=device)
        self.norm_2 = norm_class(d_model, device=device)
        self.ffn = LlamaMLP(config)

    def forward(self, 
                x: torch.Tensor, 
                attn_bias: Optional[torch.Tensor]=None, 
                attention_mask: Optional[torch.ByteTensor]=None, 
                is_causal: bool=True) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        a = self.norm_1(x)
        (b, attn_weights) = self.self_attn(a, a, attn_bias=attn_bias, attention_mask=attention_mask, needs_weights=True, is_causal=is_causal)
        x = x + b
        m = self.norm_2(x)
        n = self.ffn(m)
        x = x + n
        return (x, attn_weights)

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.config = config

        self.attn_impl = config.attn_impl

        self.word_embedding = nn.Embedding(config.enc_vocab_size, config.d_model, padding_idx=config.enc_pad_id)
        self.alibi_bias_max = config.alibi_bias_max
        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]

        self.layers = nn.ModuleList([TransformerEncoderLayer(config, config.init_device) for _ in range(config.num_enc_layers)])
        
        self.norm_f = norm_class(config.d_model, device=config.init_device)

        self.is_causal = False
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attn_bias_shape(self.attn_impl, config.q_n_heads, config.max_seq_len, causal=self.is_causal)

        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                    module.register_parameter('bias', None)

    @torch.no_grad()
    def _attn_bias(self, device, dtype, attention_mask: Optional[torch.ByteTensor]=None):
        if not self._attn_bias_initialized:
            if self.attn_bias_shape:
                self.attn_bias = torch.zeros(self.attn_bias_shape, device=device, dtype=dtype)
                self.attn_bias = build_attn_bias(self.attn_impl, self.attn_bias, self.config.q_n_heads, self.config.max_seq_len, causal=self.is_causal, alibi_bias_max=self.alibi_bias_max)
            self._attn_bias_initialized = True
        if self.attn_impl == 'flash':
            return (self.attn_bias, attention_mask)
        if self.attn_bias is not None:
            self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)
        attn_bias = self.attn_bias

        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
            else:
                _s_k = max(0, attn_bias.size(-1) - s_k)
                attn_bias = attn_bias[:, :, :, _s_k:]
            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(~attention_mask.view(-1, 1, 1, s_k), min_val)
        return (attn_bias, None)

    def forward(self, enc_inputs, attention_mask=None):
        x = self.word_embedding(enc_inputs)

        if attention_mask is not None:
            attention_mask = attention_mask.bool()

        (attn_bias, attention_mask) = self._attn_bias(device=x.device, dtype=torch.float32, attention_mask=attention_mask)

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x=x,
                                    attn_bias=attn_bias,
                                    attention_mask=attention_mask,
                                    is_causal=self.is_causal)
            all_attn_weights.append(attn_weights)

        x = self.norm_f(x)

        return x, all_attn_weights

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config, device: Optional[str]=None):
        super(TransformerDecoderLayer, self).__init__()

        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]
        attn_class = MultiheadAttention

        d_model = config.d_model
        q_n_heads = config.q_n_heads
        kv_n_heads = config.kv_n_heads

        self.norm_1 = norm_class(d_model, device=device)
        self.self_attn = attn_class(d_model=d_model, q_n_heads=q_n_heads, kv_n_heads=kv_n_heads, attn_impl=config.attn_impl, device=device)

        self.norm_2 = norm_class(d_model, device=device)
        self.cross_attn = attn_class(d_model=d_model, q_n_heads=q_n_heads, kv_n_heads=kv_n_heads, attn_impl=config.attn_impl, device=device)

        self.norm_3 = norm_class(d_model, device=device)
        self.ffn = LlamaMLP(config)

    def forward(self, 
                x: torch.Tensor, 
                enc_outputs: torch.Tensor,
                self_attn_bias: Optional[torch.Tensor]=None,
                cross_attn_bias: Optional[torch.Tensor]=None, 
                self_attention_mask: Optional[torch.ByteTensor]=None, 
                cross_attention_mask: Optional[torch.ByteTensor]=None, 
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        a = self.norm_1(x)
        (b, attn_weights) = self.self_attn(a, a, attn_bias=self_attn_bias, attention_mask=self_attention_mask, needs_weights=True, is_causal=True)
        x = x + b
        
        a = self.norm_2(x)
        (b, attn_weights) = self.cross_attn(a, enc_outputs, attn_bias=cross_attn_bias, attention_mask=cross_attention_mask, needs_weights=True, is_causal=False)
        x = x + b
        
        m = self.norm_3(x)
        n = self.ffn(m)
        x = x + n
        return (x, attn_weights)

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.config = config

        self.attn_impl = config.attn_impl

        self.word_embedding = nn.Embedding(config.dec_vocab_size, config.d_model, padding_idx=config.dec_pad_id)
        self.alibi_bias_max = config.alibi_bias_max
        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]

        self.layers = nn.ModuleList([TransformerDecoderLayer(config, config.init_device) for _ in range(config.num_dec_layers)])
        
        self.norm_f = norm_class(config.d_model, device=config.init_device)

        self._self_attn_bias_initialized = False
        self.self_attn_bias = None
        self.self_attn_bias_shape = attn_bias_shape(self.attn_impl, config.q_n_heads, config.max_seq_len, causal=True)

        self._cross_attn_bias_initialized = False
        self.cross_attn_bias = None
        self.cross_attn_bias_shape = attn_bias_shape(self.attn_impl, config.q_n_heads, config.max_seq_len, causal=False)

        self.logits_fc = nn.Linear(self.config.d_model, self.config.dec_vocab_size, bias=False)

        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                    module.register_parameter('bias', None)

    @torch.no_grad()
    def _self_attn_bias(self, device, dtype, attention_mask: Optional[torch.ByteTensor]=None):
        if not self._self_attn_bias_initialized:
            if self.self_attn_bias_shape:
                self.self_attn_bias = torch.zeros(self.self_attn_bias_shape, device=device, dtype=dtype)
                self.self_attn_bias = build_attn_bias(self.attn_impl, self.self_attn_bias, self.config.q_n_heads, self.config.max_seq_len, causal=True, alibi_bias_max=self.alibi_bias_max)
            self._self_attn_bias_initialized = True
        if self.attn_impl == 'flash':
            return (self.self_attn_bias, attention_mask)
        if self.self_attn_bias is not None:
            self.self_attn_bias = self.self_attn_bias.to(dtype=dtype, device=device)
        self_attn_bias = self.self_attn_bias

        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if self_attn_bias is None:
                self_attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
            else:
                _s_k = max(0, self_attn_bias.size(-1) - s_k)
                self_attn_bias = self_attn_bias[:, :, :, _s_k:]
            min_val = torch.finfo(self_attn_bias.dtype).min
            self_attn_bias = self_attn_bias.masked_fill(~attention_mask.view(-1, 1, 1, s_k), min_val)
        return (self_attn_bias, None)

    @torch.no_grad()
    def _cross_attn_bias(self, device, dtype, attention_mask: Optional[torch.ByteTensor]=None):
        if not self._cross_attn_bias_initialized:
            if self.cross_attn_bias_shape:
                self.cross_attn_bias = torch.zeros(self.cross_attn_bias_shape, device=device, dtype=dtype)
                self.cross_attn_bias = build_attn_bias(self.attn_impl, self.cross_attn_bias, self.config.q_n_heads, self.config.max_seq_len, causal=False, alibi_bias_max=self.alibi_bias_max)
            self._cross_attn_bias_initialized = True
        if self.attn_impl == 'flash':
            return (self.cross_attn_bias, attention_mask)
        if self.cross_attn_bias is not None:
            self.cross_attn_bias = self.cross_attn_bias.to(dtype=dtype, device=device)
        cross_attn_bias = self.cross_attn_bias

        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if cross_attn_bias is None:
                cross_attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
            else:
                _s_k = max(0, cross_attn_bias.size(-1) - s_k)
                cross_attn_bias = cross_attn_bias[:, :, :, _s_k:]
            min_val = torch.finfo(cross_attn_bias.dtype).min
            cross_attn_bias = cross_attn_bias.masked_fill(~attention_mask.view(-1, 1, 1, s_k), min_val)
        return (cross_attn_bias, None)

    def forward(self, 
                dec_inputs, 
                enc_outputs,
                self_attention_mask=None,
                cross_attention_mask=None):
        
        x = self.word_embedding(dec_inputs)

        if self_attention_mask is not None:
            self_attention_mask = self_attention_mask.bool()

        if cross_attention_mask is not None:
            cross_attention_mask = cross_attention_mask.bool()


        (self_attn_bias, self_attention_mask) = self._self_attn_bias(device=x.device, dtype=torch.float32, attention_mask=self_attention_mask)
        (cross_attn_bias, cross_attention_mask) = self._cross_attn_bias(device=x.device, dtype=torch.float32, attention_mask=cross_attention_mask)

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x=x,
                                    enc_outputs=enc_outputs,
                                    self_attn_bias=self_attn_bias,
                                    cross_attn_bias=cross_attn_bias,
                                    self_attention_mask=self_attention_mask,
                                    cross_attention_mask=cross_attention_mask,
                                    )
            all_attn_weights.append(attn_weights)

        x = self.norm_f(x)
        x = self.logits_fc(x)

        return x, all_attn_weights

class Alibi_Transformer(nn.Module):
    def __init__(self, config):
        super(Alibi_Transformer, self).__init__()

        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

    def forward(self, 
                enc_input_ids: torch.Tensor, 
                dec_input_ids: torch.Tensor,
                enc_attention_mask: Optional[torch.ByteTensor]=None, 
                dec_attention_mask: Optional[torch.ByteTensor]=None, 
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        enc_outputs, enc_attn_weights = self.encoder(enc_input_ids, 
                                                     enc_attention_mask)
    
        dec_outputs, dec_attn_weights = self.decoder(dec_inputs=dec_input_ids,
                                                     enc_outputs=enc_outputs,
                                                     self_attention_mask=dec_attention_mask,
                                                     cross_attention_mask=enc_attention_mask)

        return (dec_outputs, (enc_attn_weights, dec_attn_weights))

class PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        
        self.transform_act_fn = nn.SiLU()
        self.LayerNorm = nn.LayerNorm(config.d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class LMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = PredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.d_model, config.enc_vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.enc_vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class Alibi_MLM(nn.Module):
    def __init__(self, config):
        super(Alibi_MLM, self).__init__()

        self.encoder = TransformerEncoder(config)
        self.mlm_head = LMPredictionHead(config)

    def forward(self, 
                enc_input_ids: torch.Tensor, 
                enc_attention_mask: Optional[torch.ByteTensor]=None, 
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        enc_outputs, enc_attn_weights = self.encoder(enc_input_ids, 
                                                     enc_attention_mask)
    
        mlm_logits = self.mlm_head(enc_outputs)

        return (mlm_logits, enc_attn_weights)