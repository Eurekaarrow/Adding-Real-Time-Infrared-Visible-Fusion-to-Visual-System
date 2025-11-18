import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


# ============================================
# 感知损失 (Perceptual Loss)
# ============================================

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features
    Extracts features from relu2_2, relu3_3, relu4_3 layers
    """
    def __init__(self, device='cuda'):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract specific layers
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(9)])   # relu2_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(9, 16)])  # relu3_3
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(16, 23)]) # relu4_3
        
        # Normalization for ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image [B, 3, H, W] in range [0, 1]
            target: Target image [B, 3, H, W] in range [0, 1]
        """
        # Normalize inputs
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Extract features
        pred_feat1 = self.slice1(pred)
        pred_feat2 = self.slice2(pred_feat1)
        pred_feat3 = self.slice3(pred_feat2)
        
        with torch.no_grad():
            target_feat1 = self.slice1(target)
            target_feat2 = self.slice2(target_feat1)
            target_feat3 = self.slice3(target_feat2)
        
        # Compute L2 loss on features
        loss = (
            F.mse_loss(pred_feat1, target_feat1) +
            F.mse_loss(pred_feat2, target_feat2) +
            F.mse_loss(pred_feat3, target_feat3)
        )
        
        return loss


# ============================================
# SSIM损失 (Structural Similarity)
# ============================================

def gaussian_kernel(size=11, sigma=1.5):
    """Create Gaussian kernel for SSIM"""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    
    g = coords**2
    g = (-g / (2 * sigma**2)).exp()
    
    g /= g.sum()
    return g.reshape(-1, 1) * g.reshape(1, -1)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss
    Returns 1 - SSIM for minimization
    """
    def __init__(self, window_size=11, sigma=1.5, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        
        # Create Gaussian window
        kernel = gaussian_kernel(window_size, sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(channel, 1, window_size, window_size).contiguous()
        
        self.register_buffer('window', kernel)
        
        # Constants for stability
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
        """
        # Compute statistics
        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        return 1 - ssim_map.mean()


# ============================================
# 梯度损失 (Gradient Loss)
# ============================================

class GradientLoss(nn.Module):
    """
    Gradient loss using Sobel filters
    Preserves edge information
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_gradient(self, img):
        """Compute image gradients using Sobel"""
        # Convert to grayscale if needed
        if img.size(1) == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img
        
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        return grad_x, grad_y

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
        """
        pred_gx, pred_gy = self.compute_gradient(pred)
        target_gx, target_gy = self.compute_gradient(target)
        
        loss = F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
        
        return loss


# ============================================
# 特征蒸馏损失 (Feature Distillation)
# ============================================

class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation loss
    Matches intermediate features between student and teacher
    """
    def __init__(self):
        super(FeatureDistillationLoss, self).__init__()

    def forward(self, student_feats, teacher_feats):
        """
        Args:
            student_feats: List of student feature maps
            teacher_feats: List of teacher feature maps (must be same length)
        
        Returns:
            loss: MSE between aligned features
        """
        if teacher_feats is None:
            return torch.tensor(0.0).to(student_feats[0].device)
        
        if len(student_feats) != len(teacher_feats):
            raise ValueError("Student and teacher must have same number of feature levels")
        
        loss = 0
        for s_feat, t_feat in zip(student_feats, teacher_feats):
            # Resize if dimensions don't match
            if s_feat.shape != t_feat.shape:
                # Match channels first
                if s_feat.size(1) != t_feat.size(1):
                    # Simple channel matching
                    if s_feat.size(1) < t_feat.size(1):
                        t_feat = t_feat[:, :s_feat.size(1)]
                    elif s_feat.size(1) > t_feat.size(1):
                        pad_channels = s_feat.size(1) - t_feat.size(1)
                        t_feat = F.pad(t_feat, (0, 0, 0, 0, 0, pad_channels))
                else:
                    t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], 
                                          mode='bilinear', align_corners=True)
            
            loss += F.mse_loss(s_feat, t_feat.detach())
        
        return loss / len(student_feats)


# ============================================
# 总损失组合
# ============================================

class FusionLoss(nn.Module):
    """
    Combined loss for fusion student model
    
    Args:
        lambda_pix: Weight for pixel loss (L1)
        lambda_perc: Weight for perceptual loss
        lambda_ssim: Weight for SSIM loss
        lambda_grad: Weight for gradient loss
        lambda_feat: Weight for feature distillation
        device: Device for computation
    """
    def __init__(self, 
                 lambda_pix=1.0, 
                 lambda_perc=0.1, 
                 lambda_ssim=1.0, 
                 lambda_grad=0.5,
                 lambda_feat=0.5,
                 device='cuda'):
        super(FusionLoss, self).__init__()
        
        self.lambda_pix = lambda_pix
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.lambda_feat = lambda_feat
        
        # Initialize loss modules
        self.perceptual_loss = PerceptualLoss(device)
        self.ssim_loss = SSIMLoss(channel=3)
        self.gradient_loss = GradientLoss()
        self.feature_loss = FeatureDistillationLoss()

    def forward(self, pred, target, student_feats=None, teacher_feats=None):
        """
        Compute combined loss
        
        Args:
            pred: Predicted fused image [B, C, H, W]
            target: Teacher fused image (pseudo-GT) [B, C, H, W]
            student_feats: Optional list of student intermediate features
            teacher_feats: Optional list of teacher intermediate features
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        
        # 1. Pixel-wise L1 loss
        loss_pix = F.l1_loss(pred, target)
        loss_dict['pix'] = loss_pix.item()
        
        # 2. Perceptual loss
        loss_perc = self.perceptual_loss(pred, target)
        loss_dict['perc'] = loss_perc.item()
        
        # 3. SSIM loss
        loss_ssim = self.ssim_loss(pred, target)
        loss_dict['ssim'] = loss_ssim.item()
        
        # 4. Gradient loss
        loss_grad = self.gradient_loss(pred, target)
        loss_dict['grad'] = loss_grad.item()
        
        # 5. Feature distillation loss (optional)
        loss_feat = torch.tensor(0.0).to(pred.device)
        if student_feats is not None and teacher_feats is not None:
            loss_feat = self.feature_loss(student_feats, teacher_feats)
            loss_dict['feat'] = loss_feat.item()
        else:
            loss_dict['feat'] = 0.0
        
        # Combine losses
        total_loss = (
            self.lambda_pix * loss_pix +
            self.lambda_perc * loss_perc +
            self.lambda_ssim * loss_ssim +
            self.lambda_grad * loss_grad +
            self.lambda_feat * loss_feat
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# ============================================
# 对抗损失 (可选，用于GAN微调)
# ============================================

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for adversarial training
    Outputs a patch-wise realness score
    """
    def __init__(self, in_channels=3, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Layer 1: 3 -> 64
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 64 -> 128
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 128 -> 256
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 256 -> 512
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: 512 -> 1
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN training
    Uses LSGAN (least squares GAN) for stability
    """
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred, is_real):
        """
        Args:
            pred: Discriminator output
            is_real: Boolean, whether target is real or fake
        """
        if is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        
        return self.criterion(pred, target)


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test losses
    pred = torch.rand(2, 3, 256, 256).to(device)
    target = torch.rand(2, 3, 256, 256).to(device)
    
    # Test individual losses
    print("Testing individual losses:")
    
    perc_loss = PerceptualLoss(device)
    print(f"Perceptual Loss: {perc_loss(pred, target).item():.4f}")
    
    ssim_loss = SSIMLoss()
    print(f"SSIM Loss: {ssim_loss(pred, target).item():.4f}")
    
    grad_loss = GradientLoss()
    print(f"Gradient Loss: {grad_loss(pred, target).item():.4f}")
    
    # Test combined loss
    print("\nTesting combined loss:")
    fusion_loss = FusionLoss(device=device)
    total, loss_dict = fusion_loss(pred, target)
    
    print(f"Total Loss: {total.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # Test with feature distillation
    print("\nTesting with feature distillation:")
    student_feats = [torch.rand(2, 64, 128, 128).to(device),
                     torch.rand(2, 128, 64, 64).to(device)]
    teacher_feats = [torch.rand(2, 64, 128, 128).to(device),
                     torch.rand(2, 128, 64, 64).to(device)]
    
    total, loss_dict = fusion_loss(pred, target, student_feats, teacher_feats)
    print(f"Total Loss (with feat distill): {total.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
