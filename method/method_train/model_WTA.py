# inspired by Andrew Karpathy's minGPT
# https://www.youtube.com/watch?v=kCc8FmEb1nY


import inspect
import math
from dataclasses import dataclass
from typing import Optional
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
# Use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ['RANK'])
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        assert all(isinstance(p, torch.Tensor) for p in params)
        sizes = {p.numel() for p in params}
        param_groups = [dict(params=[p for p in params if p.numel() == size],
                             update_buffer=[torch.empty(size, device='cuda', dtype=torch.bfloat16) for _ in range(self.world_size)])
                        for size in sizes]
        super().__init__(param_groups, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            update_buffers = group['update_buffer']
            # generate weight updates in distributed fashion
            params = group['params']
            handle = None
            params_world = None
            def update_prev():
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffers):
                    p_world.data.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(0) / p_world.size(1)) ** 0.5,
                    )
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                else:
                    g = update_buffers[self.rank]
                update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather(update_buffers, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------

def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),)).to(x.dtype)

class Rotary(nn.Module):
    """Rotary position embeddings for improved attention."""

    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

class CastedLinear(nn.Linear):

    def __init__(self, in_features, out_features, config):
        super().__init__(in_features, out_features, bias=False)
        self.config = config

        self.to(self.config.dtype)
        if self.config.check_model_precision:
            self.check_model_precision()

    def check_model_precision(self):
        for name, param in self.named_parameters():
            print(f"CastedLinear Parameter {name} dtype: {param.dtype}")

    def forward(self, x):
        return F.linear(x, self.weight).to(x.dtype)

class CausalSelfAttention(nn.Module):
    
    def __init__(self, dim, num_heads, config):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.config = config

        # QKV projections
        self.c_q = CastedLinear(dim, dim, config=self.config)
        self.c_k = CastedLinear(dim, dim, config=self.config)
        self.c_v = CastedLinear(dim, dim, config=self.config)

        # Learnable mixing parameter
        self.lamb = nn.Parameter(torch.tensor(0.5))

        # Rotary embeddings
        self.rotary = Rotary(dim // self.num_heads)

        # Output projection with zero initialization
        self.c_proj = CastedLinear(dim, dim, config=self.config)
        self.to(self.config.dtype)
        if self.config.check_model_precision:
            self.check_model_precision()

    def check_model_precision(self):
        for name, param in self.named_parameters():
            print(f"CausalSelfAttention Parameter {name} dtype: {param.dtype}")

    def forward(self, x, vi, block_mask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        # Generate Q, K, V and apply RMS normalization
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)
        v = (1 - self.lamb) * v + self.lamb * vi.view_as(v) # @Grad62304977s
        q, k = rms_norm(q), rms_norm(k) # QK norm suggested by @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, dim, config):
        super().__init__()
        self.config = config
        self.c_fc = CastedLinear(dim, 4 * dim, config=config)
        self.c_proj = CastedLinear(4 * dim, dim, config=config)

        self.to(self.config.dtype)
        if self.config.check_model_precision:
            self.check_model_precision()

    def check_model_precision(self):
        for name, param in self.named_parameters():
            print(f"MLP Parameter {name} dtype: {param.dtype}")
        
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square().to(x.dtype)  # ReLU squared activation
        x = self.c_proj(x)
        return x
        

class Block(nn.Module):
 
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = CausalSelfAttention(config.model_dim, config.num_heads, config)
        self.mlp = MLP(config.model_dim, config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

        self.to(self.config.dtype)
        if self.config.check_model_precision:
            self.check_model_precision()

    def check_model_precision(self):
        for name, param in self.named_parameters():
            print(f"Block Parameter {name} dtype: {param.dtype}")

    def forward(self, x, vi, x0, block_mask):
        # Weighted residual connection
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        # Attention with RMS norm and block_mask
        x = x + self.attn(x=rms_norm(x), vi=vi, block_mask=block_mask)
        # MLP with RMS norm
        x = x + self.mlp(rms_norm(x))
        return x


@dataclass
class ModelConfig:
    sequence_length: int = 10240
    vocab_size: int = (
        50304  # number of tokens in the GPT-2 vocabulary, change to taste if you use a different vocab
    )
    num_layers: int = 12
    num_heads: int = 12
    model_dim: int = 768
    weight_tying: bool = True
    check_model_precision: bool = False
    dtype: torch.dtype = torch.float32
    timeaware: bool = True
    fix_sw_size: bool = False
    context_length: int = 0

class ValueEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.model_dim)
            for _ in range(config.num_layers // 2)
        ])

        self.to(self.config.dtype)
        if self.config.check_model_precision:
            self.check_model_precision()

    def check_model_precision(self):
        for name, param in self.named_parameters():
            print(f"Block Parameter {name} dtype: {param.dtype}")


    def forward(self, inputs) -> "list[torch.Tensor]":
        ve = [emb(inputs) for emb in self.embed]
        ve += reversed(ve)
        return ve


class Method(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # U-net design by @brendanh0gan
        self.num_encoder_layers = config.num_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers # Remaining for decoder
        
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        # Core transformer components
        self.transformer = nn.ModuleDict(dict(
            weight_embeds = nn.Embedding(config.vocab_size, config.model_dim), # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
            value_embeds = ValueEmbedding(config), # value embeddings for skip connections
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
        ))
        
        # Zero-initialized output head
        self.lm_head = CastedLinear(config.model_dim, config.vocab_size, config=config)

        # Apply weight tying if configured
        if config.weight_tying:
            self.transformer.weight_embeds.weight = (self.lm_head.weight)

        # init all weights
        self.apply(self._init_weights)

        # Special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        self.to(config.dtype)
        if config.check_model_precision:
            self.check_model_precision()

    def check_model_precision(self):
        for name, param in self.named_parameters():
            print(f"METHOD Parameter {name} dtype: {param.dtype}")

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,
                inputs: torch.Tensor,
                time_indices: torch.Tensor,
                patient_indices: torch.Tensor,
                sliding_window_size: int,
                target=None,
                target_time_indices=None,
            ):
        
        device = inputs.device
        seq_len = len(inputs)
        assert inputs.dtype == time_indices.dtype, "input and time_indices must have the same dtype"
        assert inputs.ndim == 1
        
        def patient_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            context_mask = time_indices[kv_idx] == 0
            patient_mask = patient_indices[q_idx] == patient_indices[kv_idx]
            window_mask = abs(q_idx - kv_idx) < sliding_window_size
            return (causal_mask | context_mask) & patient_mask & window_mask

        block_mask = create_block_mask(
            patient_causal_mask, None, None, 
            seq_len, seq_len, device=device, _compile=self.training
        )

        # Forward pass
        x = self.transformer.weight_embeds(inputs[None])
        x = rms_norm(x)
        x0 = x
        ve = self.transformer.value_embeds(inputs)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        # Store outputs for U-Net skip connections
        skip_connections = []
        
        # Encoder pass
        for i in range(self.num_encoder_layers):
            x = self.transformer.h[i](x, ve_enc[i], x0, block_mask)
            skip_connections.append(x)
        # Decoder pass with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.transformer.h[self.num_encoder_layers + i](x, ve_dec[i], x0, block_mask)

        # Final normalization and prediction
        x = rms_norm(x)

        logits = self.lm_head(x)
        # Apply tanh scaling for stability
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()

        # Calculate loss if target is provided
        if target is not None:
            loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    target.view(-1), 
                    ignore_index=-1, 
                    reduction='none',
                    )
            # Mask out context token loss
            if self.config.context_length > 0:
                valid_positions = (time_indices != 0)
            else:
                valid_positions = (time_indices != -1)
            num_valid = valid_positions.sum()
            loss = loss * valid_positions.float()
            loss = loss.sum() / num_valid
        else:
            loss = None
        
        outputs = {}
        outputs['loss'] = loss
        outputs['logits'] = logits
        return outputs

    def configure_optimizers_muon(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizers with separate parameter groups and learning rates.
        This implementation respects weight tying and follows the learning rate
        strategy from train_gpt2.py.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Base learning rate
            betas: Adam/AdamW beta parameters
            device_type: Type of device ('cuda' or 'cpu')
        
        Returns:
            list: List of optimizers
        """
        value_embed_params = list(self.transformer.value_embeds.parameters())
        time_head_params = [self.time_head.weight] if self.config.timeaware else []
        if self.config.weight_tying:
            embed_params = [self.lm_head.weight]
            head_params = time_head_params
        else:
            embed_params = [self.transformer.weight_embeds.weight]
            head_params = [self.lm_head.weight] + time_head_params
        hidden_matrix_params = [p for p in self.transformer.h.parameters() if p.ndim == 2]
        scalar_params = [p for p in self.parameters() if p.ndim < 2]
        # Configure fused Adam if available
        fused_available = "fused" in inspect.signature(torch.optim.Adam).parameters
        use_fused = fused_available and "cuda" in device_type
        extra_args = dict(fused=True) if use_fused else dict()
        print(f"using fused Adam: {use_fused}")
        if weight_decay > 0:
            print(f"Using weight decay: {weight_decay}")
            extra_args['weight_decay'] = weight_decay
        optimizer1 = torch.optim.Adam([
                                dict(params=embed_params + value_embed_params, lr=learning_rate*15),
                                dict(params=scalar_params + head_params, lr=learning_rate),
                            ],
                            betas=betas,
                            **extra_args,
                        )   
        optimizer2 = Muon(hidden_matrix_params, lr=learning_rate, momentum=0.95)
        return [optimizer1, optimizer2]
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # estimate the model flops throughput in MFLOPS
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.num_layers, cfg.num_heads, cfg.model_dim // cfg.num_heads, cfg.sequence_length
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor,
        time_indices: torch.Tensor,
        patient_indices: torch.Tensor,
        sliding_window_size: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens with support for longer sequences and patient boundaries.
        """
        for _ in range(max_new_tokens):
            orig_len = inputs.size(0)
            real_pos = orig_len - 1

            if orig_len > self.config.sequence_length:
                keep_length = self.config.sequence_length - self.config.context_length
                inputs = torch.cat([
                    inputs[:self.config.context_length],  # patient context
                    inputs[-keep_length:]     # most recent tokens
                ])
                time_indices = torch.cat([
                    time_indices[:self.config.context_length],
                    time_indices[-keep_length:]
                ])
                patient_indices = torch.cat([
                    patient_indices[:self.config.context_length],
                    patient_indices[-keep_length:]
                ])
                real_pos = self.config.sequence_length - 1

            # Get predictions
            outputs = self(inputs=inputs, time_indices=time_indices, patient_indices=patient_indices, sliding_window_size=sliding_window_size)
            logits = outputs['logits']
            logits = logits[0, real_pos, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            inputs_next = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat((inputs, inputs_next), dim=0)
        return inputs

    @torch.no_grad()
    def get_next_token(
        self,
        inputs: torch.Tensor,
        time_indices: torch.Tensor,
        patient_indices: torch.Tensor,
        sliding_window_size: int,
        return_probs: bool = False,
        top_k: Optional[int] = None
    ): 
        if inputs.size(0) > self.config.sequence_length:
            keep_length = self.config.sequence_length - self.config.context_length
            inputs = torch.cat([
                inputs[:self.config.context_length],  # patient context
                inputs[-keep_length:]     # most recent tokens
            ])
            time_indices = torch.cat([
                time_indices[:self.config.context_length],
                time_indices[-keep_length:]
            ])
            patient_indices = torch.cat([
                patient_indices[:self.config.context_length],
                patient_indices[-keep_length:]
            ])

        outputs = self(inputs=inputs, time_indices=time_indices, patient_indices=patient_indices, sliding_window_size=sliding_window_size)
        logits = outputs['logits']
        logits = logits[0, -1, :]
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        infer_outputs = {}
        infer_outputs['next_token'] = next_token
        if return_probs:
            infer_outputs['probs'] = probs.detach().cpu()
        
        return infer_outputs