import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================
# 基础模块
# ============================================

class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer for channel attention"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class ConvBlock(nn.Module):
    """Basic convolutional block: Conv -> GroupNorm -> SiLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


# ============================================
# 跨模态融合模块
# ============================================

class CrossModalFusionBlock(nn.Module):
    """
    跨模态融合模块，结合通道注意力和空间注意力
    支持可学习的特征融合权重
    """
    def __init__(self, channels):
        super(CrossModalFusionBlock, self).__init__()
        
        # Channel attention for fusion weight
        self.channel_weight = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(32, channels), num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 2, 1),  # Output 2 channels for IR and VIS weights
            nn.Softmax(dim=1)
        )
        
        # SE attention after fusion
        self.se = SELayer(channels)
        
        # Spatial attention
        self.sa = SpatialAttention()
        
        # Refinement conv
        self.refine = ConvBlock(channels, channels, 3, 1, 1)

    def forward(self, f_ir, f_vis):
        # Compute adaptive fusion weights
        concat_feat = torch.cat([f_ir, f_vis], dim=1)
        weights = self.channel_weight(concat_feat)  # [B, 2, H, W]
        
        # Weighted fusion
        fused = f_ir * weights[:, 0:1, :, :] + f_vis * weights[:, 1:2, :, :]
        
        # Apply attention mechanisms
        fused = self.se(fused)
        fused = self.sa(fused)
        
        # Refinement
        fused = self.refine(fused)
        
        return fused


# ============================================
# 轻量级编码器 (基于MobileNetV3思想的简化版)
# ============================================

class LightweightEncoder(nn.Module):
    """
    轻量级编码器，输出多尺度特征金字塔
    4个阶段：C1(1/2), C2(1/4), C3(1/8), C4(1/16)
    """
    def __init__(self, in_channels=3):
        super(LightweightEncoder, self).__init__()
        
        # Initial stem
        self.stem = ConvBlock(in_channels, 32, 3, 2, 1)  # 1/2
        
        # Stage 1: 1/2 -> 1/2
        self.stage1 = nn.Sequential(
            ConvBlock(32, 32, 3, 1, 1),
            ConvBlock(32, 32, 3, 1, 1)
        )
        
        # Stage 2: 1/2 -> 1/4
        self.stage2 = nn.Sequential(
            ConvBlock(32, 64, 3, 2, 1),
            ConvBlock(64, 64, 3, 1, 1),
            SELayer(64)
        )
        
        # Stage 3: 1/4 -> 1/8
        self.stage3 = nn.Sequential(
            ConvBlock(64, 128, 3, 2, 1),
            ConvBlock(128, 128, 3, 1, 1),
            SELayer(128)
        )
        
        # Stage 4: 1/8 -> 1/16
        self.stage4 = nn.Sequential(
            ConvBlock(128, 256, 3, 2, 1),
            ConvBlock(256, 256, 3, 1, 1),
            SELayer(256)
        )

    def forward(self, x):
        c0 = self.stem(x)     # 1/2, 32
        c1 = self.stage1(c0)  # 1/2, 32
        c2 = self.stage2(c1)  # 1/4, 64
        c3 = self.stage3(c2)  # 1/8, 128
        c4 = self.stage4(c3)  # 1/16, 256
        
        return [c1, c2, c3, c4]


# ============================================
# U-Net解码器
# ============================================

class DecoderBlock(nn.Module):
    """Decoder block with skip connection"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super(DecoderBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(in_ch, out_ch, 3, 1, 1)
        )
        self.conv = nn.Sequential(
            ConvBlock(out_ch + skip_ch, out_ch, 3, 1, 1),
            ConvBlock(out_ch, out_ch, 3, 1, 1)
        )
        self.se = SELayer(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x


class UNetDecoder(nn.Module):
    """U-Net style decoder with multi-scale skip connections"""
    def __init__(self, encoder_channels=[32, 64, 128, 256]):
        super(UNetDecoder, self).__init__()
        
        # Decoder blocks (reverse order)
        self.dec4 = DecoderBlock(256, 128, 128)  # 1/16 -> 1/8
        self.dec3 = DecoderBlock(128, 64, 64)    # 1/8 -> 1/4
        self.dec2 = DecoderBlock(64, 32, 32)     # 1/4 -> 1/2
        
        # Final upsampling to original resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(32, 32, 3, 1, 1)
        )

    def forward(self, fused_features):
        """
        fused_features: list of [f1, f2, f3, f4] at scales [1/2, 1/4, 1/8, 1/16]
        """
        f1, f2, f3, f4 = fused_features
        
        x = self.dec4(f4, f3)  # 1/8
        x = self.dec3(x, f2)   # 1/4
        x = self.dec2(x, f1)   # 1/2
        x = self.final_up(x)   # 1/1
        
        return x


# ============================================
# 完整Student模型
# ============================================

class FusionStudent(nn.Module):
    """
    Real-time infrared-visible fusion student model
    
    Args:
        in_channels: Number of input channels (3 for RGB/IR, 1 for grayscale IR)
        out_channels: Number of output channels (3 for fused RGB)
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(FusionStudent, self).__init__()
        
        # Dual-branch encoders
        self.ir_encoder = LightweightEncoder(in_channels)
        self.vis_encoder = LightweightEncoder(3)  # VIS always 3 channels
        
        # Fusion blocks for each scale
        encoder_channels = [32, 64, 128, 256]
        self.fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(ch) for ch in encoder_channels
        ])
        
        # Decoder
        self.decoder = UNetDecoder(encoder_channels)
        
        # Final output head
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid()  # Output [0, 1]
        )

    def forward(self, ir, vis, return_features=False):
        """
        Args:
            ir: IR image [B, C, H, W]
            vis: Visible image [B, C, H, W]
            return_features: Whether to return intermediate features for distillation
        
        Returns:
            fused: Fused image [B, C, H, W]
            features: (Optional) List of fused features at each scale
        """
        # Extract multi-scale features
        ir_feats = self.ir_encoder(ir)    # [f1, f2, f3, f4]
        vis_feats = self.vis_encoder(vis)
        
        # Fuse features at each scale
        fused_feats = []
        for i, (f_ir, f_vis) in enumerate(zip(ir_feats, vis_feats)):
            fused = self.fusion_blocks[i](f_ir, f_vis)
            fused_feats.append(fused)
        
        # Decode to output
        x = self.decoder(fused_feats)
        fused = self.head(x)
        
        if return_features:
            return fused, fused_feats
        return fused

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================
# 模型初始化辅助函数
# ============================================

def create_student_model(in_channels=3, out_channels=3, pretrained_encoder=False):
    """
    Create student model with optional pretrained encoder initialization
    
    Args:
        in_channels: Input channels (3 for RGB, 1 for grayscale IR)
        out_channels: Output channels (3 for RGB fusion)
        pretrained_encoder: Whether to use ImageNet pretrained weights (if available)
    
    Returns:
        model: FusionStudent instance
    """
    model = FusionStudent(in_channels, out_channels)
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    print(f"Student model created with {model.count_parameters()/1e6:.2f}M parameters")
    
    return model


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_student_model(in_channels=1, out_channels=3).to(device)
    
    # Test forward pass
    ir = torch.randn(2, 1, 512, 512).to(device)
    vis = torch.randn(2, 3, 512, 512).to(device)
    
    # Test without features
    fused = model(ir, vis)
    print(f"Input shape: IR {ir.shape}, VIS {vis.shape}")
    print(f"Output shape: {fused.shape}")
    
    # Test with features
    fused, feats = model(ir, vis, return_features=True)
    print(f"\nFeature pyramid shapes:")
    for i, f in enumerate(feats):
        print(f"  Level {i+1}: {f.shape}")
    
    # Count parameters
    try:
        from thop import profile, clever_format
        macs, params = profile(model, inputs=(ir, vis), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print(f"\nModel complexity:")
        print(f"  Parameters: {params}")
        print(f"  FLOPs: {macs}")
    except ImportError:
        print("\nInstall thop for FLOPs calculation: pip install thop")
