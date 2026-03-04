#!/usr/bin/env python3
"""
Generate random-weight numerical reference fixtures for Edifice architecture validation.

For each architecture: builds an architecturally-identical PyTorch model,
initializes with torch.manual_seed(42), runs a forward pass, and saves
weights + input + expected output as SafeTensors.

Requirements:
    pip install torch safetensors numpy

Usage:
    python scripts/generate_random_weight_fixtures.py
    python scripts/generate_random_weight_fixtures.py --only min_gru,lstm

Output:
    test/fixtures/numerical/{arch}_random.safetensors
"""

import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file


FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "test", "fixtures", "numerical"
)


def save_random_weight_fixture(model, inputs, output, fixture_name, extra_tensors=None):
    """Save model weights, inputs, and expected output as SafeTensors.

    Weights are prefixed with 'weight.', inputs with 'input.', output as 'expected_output'.
    """
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    tensors = {}

    # Save all model parameters with 'weight.' prefix
    for name, param in model.state_dict().items():
        tensors[f"weight.{name}"] = param.contiguous()

    # Save inputs with 'input.' prefix
    for name, tensor in inputs.items():
        tensors[f"input.{name}"] = tensor.contiguous()

    # Save expected output
    if isinstance(output, dict):
        for name, tensor in output.items():
            tensors[f"expected.{name}"] = tensor.contiguous()
    else:
        tensors["expected_output"] = output.contiguous()

    # Save any extra tensors (e.g., gradients)
    if extra_tensors:
        for name, tensor in extra_tensors.items():
            tensors[name] = tensor.contiguous()

    path = os.path.join(FIXTURES_DIR, fixture_name)
    save_file(tensors, path)

    param_count = sum(p.numel() for p in model.parameters())
    file_size = os.path.getsize(path)
    print(f"  Saved {fixture_name} ({file_size} bytes, {param_count} params)")


# =============================================================================
# MinGRU
# =============================================================================

class MinGRULayer(nn.Module):
    """Single MinGRU layer matching Edifice.Recurrent.MinGRU.build_min_gru_layer."""
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.candidate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        normed = self.norm(x)
        z = torch.sigmoid(self.gate(normed))
        c = self.candidate(normed)

        batch, seq_len, H = x.shape
        h = torch.zeros(batch, H, device=x.device)
        outputs = []
        for t in range(seq_len):
            h = (1 - z[:, t]) * h + z[:, t] * c[:, t]
            outputs.append(h)

        return torch.stack(outputs, dim=1) + x  # residual


class MinGRUModel(nn.Module):
    """Matches Edifice.Recurrent.MinGRU.build/1."""
    def __init__(self, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.input_projection = nn.Linear(embed_dim, hidden_size)
        self.layers = nn.ModuleList([MinGRULayer(hidden_size) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x[:, -1]  # last timestep


def generate_min_gru():
    print("Generating MinGRU fixture...")
    torch.manual_seed(42)
    model = MinGRUModel(embed_dim=32, hidden_size=64, num_layers=2)
    model.eval()

    torch.manual_seed(123)
    x = torch.randn(2, 8, 32)

    with torch.no_grad():
        output = model(x)

    print(f"  Output shape: {output.shape}, sample: {output[0, :5].tolist()}")
    save_random_weight_fixture(model, {"state_sequence": x}, output, "min_gru_random.safetensors")


# =============================================================================
# DeepResLSTM
# =============================================================================

class ManualLSTM(nn.Module):
    """Manual LSTM matching Axon.lstm per-gate param structure.

    Axon stores LSTM params as nested per-gate maps:
      input_kernel.wii, .wif, .wig, .wio  (each [H, H])
      hidden_kernel.whi, .whf, .whg, .who (each [H, H])
      bias.bi, .bf, .bg, .bo              (each [H])
    Gate order: i=input, f=forget, g=cell, o=output.
    """
    def __init__(self, hidden_size):
        super().__init__()
        H = hidden_size
        # Per-gate input kernels (match Axon: input_kernel.wii, .wif, .wig, .wio)
        self.input_kernel = nn.ParameterDict({
            "wii": nn.Parameter(torch.empty(H, H)),
            "wif": nn.Parameter(torch.empty(H, H)),
            "wig": nn.Parameter(torch.empty(H, H)),
            "wio": nn.Parameter(torch.empty(H, H)),
        })
        # Per-gate hidden kernels
        self.hidden_kernel = nn.ParameterDict({
            "whi": nn.Parameter(torch.empty(H, H)),
            "whf": nn.Parameter(torch.empty(H, H)),
            "whg": nn.Parameter(torch.empty(H, H)),
            "who": nn.Parameter(torch.empty(H, H)),
        })
        # Per-gate biases
        self.bias = nn.ParameterDict({
            "bi": nn.Parameter(torch.zeros(H)),
            "bf": nn.Parameter(torch.zeros(H)),
            "bg": nn.Parameter(torch.zeros(H)),
            "bo": nn.Parameter(torch.zeros(H)),
        })
        for p in [*self.input_kernel.values(), *self.hidden_kernel.values()]:
            nn.init.xavier_uniform_(p)

    def forward(self, x):
        batch, seq_len, H = x.shape
        h = torch.zeros(batch, H, device=x.device)
        c = torch.zeros(batch, H, device=x.device)
        outputs = []
        for t in range(seq_len):
            xt = x[:, t]
            i_gate = torch.sigmoid(xt @ self.input_kernel["wii"] + h @ self.hidden_kernel["whi"] + self.bias["bi"])
            f_gate = torch.sigmoid(xt @ self.input_kernel["wif"] + h @ self.hidden_kernel["whf"] + self.bias["bf"])
            g_gate = torch.tanh(xt @ self.input_kernel["wig"] + h @ self.hidden_kernel["whg"] + self.bias["bg"])
            o_gate = torch.sigmoid(xt @ self.input_kernel["wio"] + h @ self.hidden_kernel["who"] + self.bias["bo"])
            c = f_gate * c + i_gate * g_gate
            h = o_gate * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class ResLSTMBlock(nn.Module):
    """Matches Edifice.Recurrent.DeepResLSTM.res_lstm_block."""
    def __init__(self, hidden_size):
        super().__init__()
        self.prenorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.lstm = ManualLSTM(hidden_size)
        self.decoder = nn.Linear(hidden_size, hidden_size)
        # Zero-init decoder
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        normed = self.prenorm(x)
        lstm_out = self.lstm(normed)
        decoded = self.decoder(lstm_out)
        return x + decoded  # residual


class DeepResLSTMModel(nn.Module):
    """Matches Edifice.Recurrent.DeepResLSTM.build/1."""
    def __init__(self, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.encoder = nn.Linear(embed_dim, hidden_size)
        self.blocks = nn.ModuleList([ResLSTMBlock(hidden_size) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x[:, -1]  # last timestep


def generate_lstm():
    print("Generating DeepResLSTM fixture...")
    torch.manual_seed(42)
    model = DeepResLSTMModel(embed_dim=32, hidden_size=64, num_layers=2)
    model.eval()

    torch.manual_seed(123)
    x = torch.randn(2, 8, 32)

    with torch.no_grad():
        output = model(x)

    print(f"  Output shape: {output.shape}, sample: {output[0, :5].tolist()}")
    save_random_weight_fixture(model, {"state_sequence": x}, output, "lstm_random.safetensors")


# =============================================================================
# GAT
# =============================================================================

class GATLayer(nn.Module):
    """Single GAT layer matching Edifice.Graph.GAT.gat_layer."""
    def __init__(self, in_dim, out_dim, num_heads, negative_slope=0.2, concat_heads=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat_heads = concat_heads
        self.negative_slope = negative_slope

        total_dim = num_heads * out_dim
        self.proj = nn.Linear(in_dim, total_dim)
        self.attn_src = nn.Linear(total_dim, num_heads, bias=False)
        self.attn_tgt = nn.Linear(total_dim, num_heads, bias=False)

    def forward(self, nodes, adjacency):
        batch, num_nodes, _ = nodes.shape
        projected = self.proj(nodes)  # [B, N, heads*out_dim]

        # Attention scores
        src_scores = self.attn_src(projected)  # [B, N, heads]
        tgt_scores = self.attn_tgt(projected)  # [B, N, heads]

        # e_ij = LeakyReLU(src_i + tgt_j)
        e = src_scores.unsqueeze(2) + tgt_scores.unsqueeze(1)  # [B, N, N, heads]
        e = F.leaky_relu(e, self.negative_slope)

        # Mask non-edges
        mask = (adjacency > 0).unsqueeze(3)  # [B, N, N, 1]
        e_masked = e.masked_fill(~mask.expand_as(e), -1e9)

        # Softmax over neighbors
        alpha = F.softmax(e_masked, dim=2)
        alpha = alpha * mask.float().expand_as(alpha)

        # Reshape projected for attention
        z = projected.reshape(batch, num_nodes, self.num_heads, self.out_dim)
        z_t = z.permute(0, 2, 1, 3)  # [B, heads, N, out_dim]
        alpha_t = alpha.permute(0, 3, 1, 2)  # [B, heads, N, N]

        h_prime = torch.matmul(alpha_t, z_t)  # [B, heads, N, out_dim]
        h_prime = h_prime.permute(0, 2, 1, 3)  # [B, N, heads, out_dim]

        if self.concat_heads:
            return h_prime.reshape(batch, num_nodes, self.num_heads * self.out_dim)
        else:
            return h_prime.mean(dim=2)


class GATModel(nn.Module):
    """Matches Edifice.Graph.GAT.build/1."""
    def __init__(self, input_dim, hidden_size, num_heads, num_classes, num_layers=2):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        in_dim = input_dim
        for i in range(num_layers - 1):
            layer = GATLayer(in_dim, hidden_size, num_heads, concat_heads=True)
            self.hidden_layers.append(layer)
            in_dim = num_heads * hidden_size

        self.output_layer = GATLayer(in_dim, num_classes, num_heads=1, concat_heads=False)

    def forward(self, nodes, adjacency):
        x = nodes
        for layer in self.hidden_layers:
            x = F.elu(layer(x, adjacency))
        return self.output_layer(x, adjacency)


def generate_gat():
    print("Generating GAT fixture...")
    torch.manual_seed(42)
    model = GATModel(input_dim=16, hidden_size=8, num_heads=4, num_classes=7, num_layers=2)
    model.eval()

    torch.manual_seed(123)
    num_nodes = 10
    nodes = torch.randn(2, num_nodes, 16)
    # Random adjacency (symmetric, with self-loops)
    adj = torch.randint(0, 2, (2, num_nodes, num_nodes)).float()
    adj = (adj + adj.transpose(1, 2)).clamp(max=1.0)
    adj[:, range(num_nodes), range(num_nodes)] = 1.0

    with torch.no_grad():
        output = model(nodes, adj)

    print(f"  Output shape: {output.shape}, sample: {output[0, 0, :5].tolist()}")
    save_random_weight_fixture(model, {"nodes": nodes, "adjacency": adj}, output, "gat_random.safetensors")


# =============================================================================
# GQA (Grouped Query Attention)
# =============================================================================

class GQAAttention(nn.Module):
    """Matches Edifice.Attention.GQA.build_gqa_attention."""
    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.heads_per_group = num_heads // num_kv_heads

        q_dim = num_heads * self.head_dim
        kv_dim = num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_size, q_dim)
        self.k_proj = nn.Linear(hidden_size, kv_dim)
        self.v_proj = nn.Linear(hidden_size, kv_dim)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        head_dim = self.head_dim

        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, head_dim).transpose(1, 2)

        # Repeat KV heads
        k = k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1).reshape(batch, self.num_heads, seq_len, head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1).reshape(batch, self.num_heads, seq_len, head_dim)

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * head_dim)
        return self.out_proj(out)


class GQABlock(nn.Module):
    """Matches TransformerBlock.layer with GQA attention."""
    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = GQAAttention(hidden_size, num_heads, num_kv_heads)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn_up = nn.Linear(hidden_size, 4 * hidden_size)
        self.ffn_down = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, x):
        # Attention sublayer
        x = x + self.attn(self.attn_norm(x))
        # FFN sublayer
        h = self.ffn_norm(x)
        h = F.gelu(self.ffn_up(h))
        h = self.ffn_down(h)
        return x + h


class GQAModel(nn.Module):
    """Matches Edifice.Attention.GQA.build/1."""
    def __init__(self, embed_dim, hidden_size, num_heads, num_kv_heads, num_layers):
        super().__init__()
        # Edifice skips input_projection when embed_dim == hidden_size
        self.input_projection = nn.Linear(embed_dim, hidden_size) if embed_dim != hidden_size else None
        self.blocks = nn.ModuleList([
            GQABlock(hidden_size, num_heads, num_kv_heads) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        if self.input_projection is not None:
            x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x[:, -1]  # last timestep


def generate_gqa():
    print("Generating GQA fixture...")
    torch.manual_seed(42)
    model = GQAModel(embed_dim=32, hidden_size=32, num_heads=4, num_kv_heads=2, num_layers=2)
    model.eval()

    torch.manual_seed(123)
    x = torch.randn(2, 8, 32)

    with torch.no_grad():
        output = model(x)

    print(f"  Output shape: {output.shape}, sample: {output[0, :5].tolist()}")
    save_random_weight_fixture(model, {"state_sequence": x}, output, "gqa_random.safetensors")


# =============================================================================
# DeltaNet
# =============================================================================

class DeltaNetLayer(nn.Module):
    """Matches Edifice.Recurrent.DeltaNet.build_delta_net_layer."""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm = nn.LayerNorm(hidden_size)
        self.qkvb_proj = nn.Linear(hidden_size, 4 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        normed = self.norm(x)
        proj = self.qkvb_proj(normed)  # [B, T, 4*H]
        H = self.hidden_size
        q = proj[..., :H]
        k = proj[..., H:2*H]
        v = proj[..., 2*H:3*H]
        beta_pre = proj[..., 3*H:]

        batch, seq_len = x.shape[:2]
        head_dim = self.head_dim

        beta = torch.sigmoid(beta_pre)

        # Reshape to multi-head
        q = q.reshape(batch, seq_len, self.num_heads, head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, head_dim)
        beta = beta.reshape(batch, seq_len, self.num_heads, head_dim)

        # L2 normalize keys
        k = F.normalize(k, dim=-1, eps=1e-6)

        # Sequential delta rule scan
        S = torch.zeros(batch, self.num_heads, head_dim, head_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            q_t = q[:, t]  # [B, heads, head_dim]
            k_t = k[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t]

            # Retrieval: S @ k
            retrieval = torch.einsum('bhij,bhj->bhi', S, k_t)
            error = v_t - retrieval
            scaled_error = beta_t * error
            # Update: S += scaled_error outer k
            S = S + torch.einsum('bhi,bhj->bhij', scaled_error, k_t)
            # Output: S @ q
            o_t = torch.einsum('bhij,bhj->bhi', S, q_t)
            outputs.append(o_t)

        scan_out = torch.stack(outputs, dim=1)  # [B, T, heads, head_dim]
        scan_out = scan_out.reshape(batch, seq_len, self.num_heads * head_dim)

        out = self.out_proj(scan_out)
        return x + out  # residual


class DeltaNetModel(nn.Module):
    """Matches Edifice.Recurrent.DeltaNet.build/1."""
    def __init__(self, embed_dim, hidden_size, num_heads, num_layers):
        super().__init__()
        # Edifice skips input_projection when embed_dim == hidden_size
        self.input_projection = nn.Linear(embed_dim, hidden_size) if embed_dim != hidden_size else None
        self.layers = nn.ModuleList([
            DeltaNetLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        if self.input_projection is not None:
            x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x[:, -1]  # last timestep


def generate_delta_net():
    print("Generating DeltaNet fixture...")
    torch.manual_seed(42)
    # hidden_size must be divisible by num_heads
    model = DeltaNetModel(embed_dim=32, hidden_size=32, num_heads=4, num_layers=2)
    model.eval()

    torch.manual_seed(123)
    x = torch.randn(2, 8, 32)

    with torch.no_grad():
        output = model(x)

    # Also compute gradients for gradient validation test
    x_grad = x.clone().requires_grad_(True)
    model.train()  # need grad
    out_grad = model(x_grad)
    loss = out_grad.sum()
    loss.backward()
    input_grad = x_grad.grad.clone()
    model.eval()

    print(f"  Output shape: {output.shape}, sample: {output[0, :5].tolist()}")
    save_random_weight_fixture(
        model, {"state_sequence": x}, output, "delta_net_random.safetensors",
        extra_tensors={"gradient.input": input_grad}
    )


# =============================================================================
# Mamba
# =============================================================================

class MambaBlock(nn.Module):
    """Matches Edifice.SSM.Common.build_block + Mamba.build_selective_ssm_parallel."""
    def __init__(self, hidden_size, state_size=16, expand_factor=2, conv_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        inner_size = hidden_size * expand_factor
        self.inner_size = inner_size
        dt_rank = max(hidden_size // 16, 1)
        self.dt_rank = dt_rank

        self.norm = nn.LayerNorm(hidden_size)
        self.in_proj = nn.Linear(hidden_size, inner_size * 2)

        # Depthwise causal conv1d
        self.dw_conv = nn.Conv1d(inner_size, inner_size, conv_size,
                                 padding=conv_size - 1, groups=inner_size)
        self.conv_size = conv_size

        # SSM projections
        self.bc_proj = nn.Linear(inner_size, state_size * 2)
        self.dt_rank_proj = nn.Linear(inner_size, dt_rank)
        self.dt_proj = nn.Linear(dt_rank, inner_size)

        self.out_proj = nn.Linear(inner_size, hidden_size)

    def forward(self, x):
        normed = self.norm(x)
        xz = self.in_proj(normed)  # [B, T, 2*inner]
        x_branch = xz[..., :self.inner_size]
        z_branch = xz[..., self.inner_size:]

        # Depthwise conv (channels-first for Conv1d)
        x_conv = self.dw_conv(x_branch.transpose(1, 2))  # [B, inner, T+pad]
        x_conv = x_conv[..., :x_branch.shape[1]]  # causal: trim right padding
        x_conv = x_conv.transpose(1, 2)  # [B, T, inner]
        x_activated = F.silu(x_conv)

        # SSM
        bc = self.bc_proj(x_activated)
        B = bc[..., :self.state_size]
        C = bc[..., self.state_size:]

        dt = self.dt_rank_proj(x_activated)
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)
        dt = dt.clamp(0.001, 0.1)

        # Discretize and scan
        ssm_out = self._selective_scan(x_activated, dt, B, C)

        # Gating
        z_activated = F.silu(z_branch)
        gated = ssm_out * z_activated
        return self.out_proj(gated)

    def _selective_scan(self, x, dt, B, C):
        """Sequential selective scan matching Edifice.SSM.Common.selective_scan_fallback."""
        batch, seq_len, inner = x.shape
        N = self.state_size

        # Fixed A matrix: -[1, 2, ..., N]
        a_diag = -(torch.arange(N, dtype=torch.float32, device=x.device) + 1.0)

        # Discretize
        dt_expanded = dt.unsqueeze(3)  # [B, T, inner, 1]
        a_expanded = a_diag.reshape(1, 1, 1, N)
        a_bar = torch.exp(dt_expanded * a_expanded)  # [B, T, inner, N]

        dt_mean = dt.mean(dim=2, keepdim=True).unsqueeze(3)  # [B, T, 1, 1]
        b_expanded = B.unsqueeze(2)  # [B, T, 1, N]
        b_bar = dt_mean * b_expanded  # [B, T, 1, N]

        x_expanded = x.unsqueeze(3)  # [B, T, inner, 1]
        bx = b_bar * x_expanded  # [B, T, inner, N]

        # Sequential scan
        h = torch.zeros(batch, 1, inner, N, device=x.device)
        h_list = []
        for t in range(seq_len):
            a_t = a_bar[:, t:t+1]
            bx_t = bx[:, t:t+1]
            h = a_t * h + bx_t
            h_list.append(h)

        h_all = torch.cat(h_list, dim=1)  # [B, T, inner, N]

        # Output: y = sum(C * h, dim=-1)
        c_expanded = C.unsqueeze(2)  # [B, T, 1, N]
        y = (c_expanded * h_all).sum(dim=3)  # [B, T, inner]
        return y


class MambaModel(nn.Module):
    """Matches Edifice.SSM.Mamba.build/1 via Common.build_model."""
    def __init__(self, embed_dim, hidden_size, state_size, num_layers, expand_factor=2, conv_size=4):
        super().__init__()
        # Edifice skips input_projection when embed_dim == hidden_size
        self.input_projection = nn.Linear(embed_dim, hidden_size) if embed_dim != hidden_size else None
        self.blocks = nn.ModuleList([
            MambaBlock(hidden_size, state_size, expand_factor, conv_size)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        if self.input_projection is not None:
            x = self.input_projection(x)
        for block in self.blocks:
            x = x + block(x)  # residual in Common.build_model
        return x[:, -1]  # last timestep


def generate_mamba():
    print("Generating Mamba fixture...")
    torch.manual_seed(42)
    model = MambaModel(embed_dim=32, hidden_size=32, state_size=8, num_layers=2,
                       expand_factor=2, conv_size=4)
    model.eval()

    torch.manual_seed(123)
    x = torch.randn(2, 8, 32)

    with torch.no_grad():
        output = model(x)

    print(f"  Output shape: {output.shape}, sample: {output[0, :5].tolist()}")
    save_random_weight_fixture(model, {"state_sequence": x}, output, "mamba_random.safetensors")


# =============================================================================
# DiT (Diffusion Transformer)
# =============================================================================

class DiTBlock(nn.Module):
    """Matches Edifice.Generative.DiT.build_dit_block."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        mlp_dim = int(hidden_size * mlp_ratio)
        self.adaln_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_q = nn.Linear(hidden_size, hidden_size)
        self.attn_k = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, hidden_size)
        self.attn_out_proj = nn.Linear(hidden_size, hidden_size)

        self.adaln_mlp = nn.Linear(hidden_size, 3 * hidden_size)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp_up = nn.Linear(hidden_size, mlp_dim)
        self.mlp_down = nn.Linear(mlp_dim, hidden_size)

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

    def forward(self, x, condition):
        H = x.shape[-1]

        # Attention sublayer with AdaLN-Zero
        adaln_params = self.adaln_attn(condition)  # [B, 3*H]
        shift, scale, gate = adaln_params[:, :H], adaln_params[:, H:2*H], adaln_params[:, 2*H:]

        x_mod = (1 + scale) * self.attn_norm(x) + shift
        attn_out = self._self_attention(x_mod)
        attn_out = gate * attn_out  # Edifice AdaLN gate uses raw gate, no sigmoid
        x = x + attn_out

        # MLP sublayer with AdaLN-Zero
        adaln_params2 = self.adaln_mlp(condition)  # [B, 3*H]
        shift2, scale2, gate2 = adaln_params2[:, :H], adaln_params2[:, H:2*H], adaln_params2[:, 2*H:]

        x_mod2 = (1 + scale2) * self.mlp_norm(x) + shift2
        mlp_out = F.gelu(self.mlp_up(x_mod2))
        mlp_out = self.mlp_down(mlp_out)
        mlp_out = gate2 * mlp_out  # Edifice AdaLN gate uses raw gate, no sigmoid
        return x + mlp_out

    def _self_attention(self, x):
        # Handle both 2D [B, H] and 3D [B, T, H] inputs (DiT uses 2D)
        was_2d = (x.dim() == 2)
        if was_2d:
            x = x.unsqueeze(1)

        B, T, _ = x.shape
        q = self.attn_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.attn_k(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        # Numerically stable softmax
        scores_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - scores_max)
        weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + 1e-8)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        out = self.attn_out_proj(out)

        if was_2d:
            out = out.squeeze(1)
        return out


class DiTModel(nn.Module):
    """Matches Edifice.Generative.DiT.build/1."""
    def __init__(self, input_dim, hidden_size, depth, num_heads, mlp_ratio=4.0, num_steps=1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_steps = num_steps

        # Timestep MLP
        self.time_mlp_1 = nn.Linear(hidden_size, hidden_size)
        self.time_mlp_2 = nn.Linear(hidden_size, hidden_size)

        # Input projection + pos embed
        self.input_embed = nn.Linear(input_dim, hidden_size)
        self.pos_embed_bias = nn.Parameter(torch.zeros(hidden_size))

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # Final layers
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, input_dim)

    def _sinusoidal_embed(self, t):
        """Match Edifice.Blocks.SinusoidalPE.timestep_embed_impl exactly."""
        H = self.hidden_size
        half_dim = H // 2
        t_f = t.float() / self.num_steps

        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32, device=t.device)
            / max(half_dim - 1, 1)
        )
        angles = t_f.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    def forward(self, noisy_input, timestep):
        # Timestep embedding
        time_embed = self._sinusoidal_embed(timestep)
        time_embed = F.silu(self.time_mlp_1(time_embed))
        condition = self.time_mlp_2(time_embed)

        # Input projection
        x = self.input_embed(noisy_input) + self.pos_embed_bias

        # DiT blocks
        for block in self.blocks:
            x = block(x, condition)

        # Output
        x = self.final_norm(x)
        return self.output_proj(x)


def generate_dit():
    print("Generating DiT fixture...")
    torch.manual_seed(42)
    model = DiTModel(input_dim=64, hidden_size=128, depth=2, num_heads=4, mlp_ratio=4.0)
    model.eval()

    torch.manual_seed(123)
    noisy_input = torch.randn(2, 64)
    timestep = torch.tensor([100, 500], dtype=torch.float32)

    with torch.no_grad():
        output = model(noisy_input, timestep)

    print(f"  Output shape: {output.shape}, sample: {output[0, :5].tolist()}")
    save_random_weight_fixture(
        model, {"noisy_input": noisy_input, "timestep": timestep}, output, "dit_random.safetensors"
    )


# =============================================================================
# Main
# =============================================================================

GENERATORS = {
    "min_gru": generate_min_gru,
    "lstm": generate_lstm,
    "gat": generate_gat,
    "gqa": generate_gqa,
    "delta_net": generate_delta_net,
    "mamba": generate_mamba,
    "dit": generate_dit,
}


def main():
    parser = argparse.ArgumentParser(description="Generate random-weight fixtures for Edifice")
    parser.add_argument("--only", type=str, help="Comma-separated list of architectures to generate")
    args = parser.parse_args()

    if args.only:
        names = [n.strip() for n in args.only.split(",")]
    else:
        names = list(GENERATORS.keys())

    print(f"Generating fixtures for: {', '.join(names)}")
    for name in names:
        if name not in GENERATORS:
            print(f"  Unknown architecture: {name}, skipping")
            continue
        GENERATORS[name]()

    print(f"\nAll fixtures generated in {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
