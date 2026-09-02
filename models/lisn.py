"""
LISN — Lightweight Information Split Network
=============================================
Clean implementation based on:
  Liu et al., "Infrared Image Super-Resolution via Lightweight Information Split Network", 2024.

Architecture (Fig. 1 in paper):
  SFE → DFE (N×LISB) → DFF (Conv1 + Conv3 + PA) → IIR (Conv3 + PixelShuffle)

Key design (Section 3.2, Fig. 2):
  LISB = Split → SBB → Split → RDB + CCA + skip

Paper parameters (Table 1, Section 4.2):
  - 6 LISBs, embed_dim=48, ~279K params for ×4, ~258K for ×2
  - Input: 1-channel grayscale infrared
  - Patch size: 64×64 during training
  - Window size: 8 (for shift operations)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PixelAttention(nn.Module):
    """Pixel Attention (PA) — Fig. 3(b) in paper.
    
    Generates a 3D attention map (H×W×C) using two Conv1×1 layers
    with a Sigmoid gate. Minimal extra parameters.
    """
    def __init__(self, channels):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.pa(x)


class ContrastAwareChannelAttention(nn.Module):
    """Contrast-aware Channel Attention (CCA) — Fig. 3(a), Eq. 9 in paper.

    Combines global average pooling with a contrast (std-dev) term,
    then applies two Conv1×1 layers + Sigmoid to produce channel weights.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Average pooling
        avg = F.adaptive_avg_pool2d(x, 1)                   # (B, C, 1, 1)
        # Contrast: channel-wise std deviation (Eq. 9)
        contrast = torch.sqrt(
            F.adaptive_avg_pool2d((x - avg) ** 2, 1) + 1e-6
        )                                                     # (B, C, 1, 1)
        t1 = avg + contrast                                   # (B, C, 1, 1)
        weights = self.fc(t1)                                 # (B, C, 1, 1)
        return x * weights


class ResidualDepthwiseBlock(nn.Module):
    """Residual Depth-wise Convolution Block (RDB) — Fig. 3(d), Eq. 8.

    Conv1×1 → DWConv3×3 → Sigmoid, with a residual skip.
    Refines local texture details with very few parameters.
    """
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 1),                 # pointwise
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),  # depthwise
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.body(x) * x + x   # residual gating (Eq. 8 uses sigmoid·x + x)


class FeedForwardNetwork(nn.Module):
    """Feed-Forward Network used inside the Shift Building Block — Fig. 3(e).

    Conv1×1 → Sigmoid → PA → Conv1×1  (paper Fig. 3(e))
    Expansion factor = 2 (keeps the model lightweight).
    """
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.Sigmoid()
        self.pa = PixelAttention(hidden)
        self.conv2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.pa(x)
        x = self.conv2(x)
        return x


def shift_features(x, shift_size=1):
    """Zero-parameter, zero-FLOP shift operation — Fig. 3(c).
    
    Shifts 1/4 of channels in each of 4 directions (up/down/left/right).
    The remaining channels (if any) are left unchanged.
    This is the key efficiency trick: captures spatial relationships
    without any learned parameters or FLOPs.
    """
    B, C, H, W = x.shape
    quarter = C // 4
    out = x.clone()
    # Shift up
    out[:, 0:quarter, :H-shift_size, :] = x[:, 0:quarter, shift_size:, :]
    # Shift down
    out[:, quarter:2*quarter, shift_size:, :] = x[:, quarter:2*quarter, :H-shift_size, :]
    # Shift left
    out[:, 2*quarter:3*quarter, :, :W-shift_size] = x[:, 2*quarter:3*quarter, :, shift_size:]
    # Shift right
    out[:, 3*quarter:4*quarter, :, shift_size:] = x[:, 3*quarter:4*quarter, :, :W-shift_size]
    return out


class ShiftBuildingBlock(nn.Module):
    """Shift Building Block (SBB) — Fig. 3(c) + Eq. 7.

    Shift → residual add → LayerNorm → FFN → residual add.
    
    This replaces self-attention in a standard Transformer with
    the zero-parameter shift operation, drastically cutting compute.
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)   # equivalent to LayerNorm for conv features
        self.ffn = FeedForwardNetwork(channels)

    def forward(self, x):
        # Shift + residual (Eq. 7 line 1)
        shifted = shift_features(x) + x
        # LayerNorm + FFN + residual (Eq. 7 line 2)
        out = self.ffn(self.norm(shifted)) + shifted
        return out


# ---------------------------------------------------------------------------
# LISB — Lightweight Information Split Block  (Fig. 2, Section 3.2)
# ---------------------------------------------------------------------------

class LISB(nn.Module):
    """Lightweight Information Split Block — the core building block.

    Pipeline (following Fig. 2 and Eqs. 6-9):
      1. Split input C channels → r1 (C/2) + m1 (C/2)
      2. SBB processes m1
      3. Split SBB output → r2 (C/4) + m2 (C/4)
      4. RDB processes m2 → m3
      5. Concatenate [r1, r2, m3] → C channels
      6. CCA re-weights channels
      7. Long skip connection: output = CCA_out + input
    """
    def __init__(self, channels):
        super().__init__()
        assert channels % 4 == 0, f"channels ({channels}) must be divisible by 4"
        
        half = channels // 2
        quarter = channels // 4
        
        self.half = half
        self.quarter = quarter
        
        # SBB operates on C/2 channels
        self.sbb = ShiftBuildingBlock(half)
        
        # RDB operates on C/4 channels
        self.rdb = ResidualDepthwiseBlock(quarter)
        
        # CCA operates on full C channels (after concatenation)
        self.cca = ContrastAwareChannelAttention(channels)

    def forward(self, x):
        # Step 1: First channel split (Eq. 6)
        r1, m1 = x[:, :self.half], x[:, self.half:]
        
        # Step 2: SBB processes m1 (Eq. 7)
        m1 = self.sbb(m1)
        
        # Step 3: Second channel split
        r2, m2 = m1[:, :self.quarter], m1[:, self.quarter:]
        
        # Step 4: RDB processes m2 (Eq. 8)
        m3 = self.rdb(m2)
        
        # Step 5: Concatenate [r1, r2, m3] = C channels
        concat = torch.cat([r1, r2, m3], dim=1)
        
        # Step 6-7: CCA + long skip (Eq. 9)
        return self.cca(concat) + x


# ---------------------------------------------------------------------------
# LISN — Full Network  (Fig. 1, Section 3.1)
# ---------------------------------------------------------------------------

class LISN(nn.Module):
    """Lightweight Information Split Network for infrared image SR.

    Architecture (Fig. 1):
      SFE:  Conv3×3  (num_in_ch → embed_dim)
      DFE:  N × LISB  (embed_dim → embed_dim each)
      DFF:  Conv1×1 → Conv3×3 → PA  (N*embed_dim → embed_dim)
      IIR:  Conv3×3 + PixelShuffle  (embed_dim → num_out_ch * scale²)
      
    Default config (Section 4.2):
      num_in_ch=1, embed_dim=92, num_lisb=6, upscale=4
      → ~284K parameters (paper reports 279K)
    """
    def __init__(
        self,
        num_in_ch: int = 1,
        num_out_ch: int = 1,
        embed_dim: int = 92,
        num_lisb: int = 6,
        upscale: int = 4,
    ):
        super().__init__()
        self.upscale = upscale
        
        # --- SFE: Shallow Feature Extraction (Eq. 1) ---
        self.sfe = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        
        # --- DFE: Deep Feature Extraction (Eq. 2) ---
        self.dfe = nn.ModuleList([LISB(embed_dim) for _ in range(num_lisb)])
        
        # --- DFF: Dense Feature Fusion (Eq. 3) ---
        # Concatenates outputs of ALL LISBs → Conv1×1 reduces back
        self.dff_conv1 = nn.Conv2d(embed_dim * num_lisb, embed_dim, 1)
        self.dff_conv3 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.dff_pa = PixelAttention(embed_dim)
        
        # --- IIR: Infrared Image Reconstruction (Eq. 4) ---
        self.iir_conv = nn.Conv2d(embed_dim, num_out_ch * upscale ** 2, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        # SFE
        f0 = self.sfe(x)                              # (B, embed_dim, H, W)
        
        # DFE — collect ALL LISB outputs for dense fusion
        lisb_outputs = []
        feat = f0
        for lisb in self.dfe:
            feat = lisb(feat)
            lisb_outputs.append(feat)
        
        # DFF — concatenate + fuse (Eq. 3)
        concat = torch.cat(lisb_outputs, dim=1)        # (B, embed_dim*N, H, W)
        fused = self.dff_conv1(concat)                  # (B, embed_dim, H, W)
        fused = self.dff_conv3(fused)
        fused = self.dff_pa(fused)
        
        # Residual: shallow + deep (Eq. 4)
        out = fused + f0                                # (B, embed_dim, H, W)
        
        # IIR — upsample
        out = self.iir_conv(out)                        # (B, C_out * scale², H, W)
        out = self.pixel_shuffle(out)                   # (B, C_out, H*scale, W*scale)
        
        return out


# ---------------------------------------------------------------------------
# Convenience: build model with paper defaults
# ---------------------------------------------------------------------------

def build_lisn(upscale=4, num_in_ch=1, num_out_ch=1, embed_dim=92, num_lisb=6):
    """Build LISN with paper-default parameters."""
    model = LISN(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        embed_dim=embed_dim,
        num_lisb=num_lisb,
        upscale=upscale,
    )
    return model


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for scale in [2, 4]:
        model = build_lisn(upscale=scale).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        
        x = torch.randn(1, 1, 64, 64).to(device)
        with torch.no_grad():
            y = model(x)
        
        print(f"Scale ×{scale}: {total_params:,} params ({total_params/1e3:.1f}K) | "
              f"Input {tuple(x.shape)} → Output {tuple(y.shape)}")
