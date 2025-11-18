import os
import argparse
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
import sys
sys.path.append('src')
from models.student import create_student_model
from losses.losses import FusionLoss
from dataset import create_dataloaders
from utils import set_seed, save_checkpoint, load_checkpoint


# ============================================
# 训练配置
# ============================================

def get_args():
    parser = argparse.ArgumentParser(description='Train Fusion Student Model')
    
    # 数据路径参数（适配您的数据结构）
    parser.add_argument('--ir_root', type=str, default='./data/ir',
                       help='Path to IR images directory')
    parser.add_argument('--vis_root', type=str, default='./data/vi',
                       help='Path to VIS images directory')
    parser.add_argument('--teacher_root', type=str, default='./DDAEFuse_MSRS',
                       help='Path to Teacher fused images directory')
    
    # 数据集划分参数
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio')
    
    parser.add_argument('--input_size', type=int, default=512,
                       help='Input image size')
    
    # 模型参数
    parser.add_argument('--in_channels', type=int, default=1,
                       help='Number of input channels (1 for grayscale IR, 3 for RGB)')
    parser.add_argument('--out_channels', type=int, default=3,
                       help='Number of output channels')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=120,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # 损失权重
    parser.add_argument('--lambda_pix', type=float, default=1.0,
                       help='Weight for pixel loss')
    parser.add_argument('--lambda_perc', type=float, default=0.1,
                       help='Weight for perceptual loss')
    parser.add_argument('--lambda_ssim', type=float, default=1.0,
                       help='Weight for SSIM loss')
    parser.add_argument('--lambda_grad', type=float, default=0.5,
                       help='Weight for gradient loss')
    parser.add_argument('--lambda_feat', type=float, default=0.5,
                       help='Weight for feature distillation')
    
    # 增强
    parser.add_argument('--misalign_prob', type=float, default=0.5,
                       help='Probability of applying misalignment augmentation')
    
    # 优化
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    
    # 日志和保存
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log_freq', type=int, default=50,
                       help='Log metrics every N iterations')
    parser.add_argument('--vis_freq', type=int, default=200,
                       help='Visualize results every N iterations')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


# ============================================
# 时间估算工具
# ============================================

class TrainingTimer:
    """训练时间估算器"""
    def __init__(self, total_epochs, steps_per_epoch):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.start_time = time.time()
        self.epoch_times = []
        
    def update_epoch(self, epoch, epoch_time):
        """更新epoch时间"""
        self.epoch_times.append(epoch_time)
        
        # 计算剩余时间
        avg_epoch_time = np.mean(self.epoch_times[-5:])  # 使用最近5个epoch的平均值
        remaining_epochs = self.total_epochs - epoch - 1
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        # 计算总用时和预计结束时间
        elapsed = time.time() - self.start_time
        estimated_total = elapsed + estimated_remaining
        eta = datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining)
        
        return {
            'elapsed': elapsed,
            'epoch_time': epoch_time,
            'avg_epoch_time': avg_epoch_time,
            'estimated_remaining': estimated_remaining,
            'estimated_total': estimated_total,
            'eta': eta
        }
    
    @staticmethod
    def format_time(seconds):
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}min"


# ============================================
# 训练一个epoch
# ============================================

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, 
                   device, epoch, args, writer, global_step):
    """训练一个epoch"""
    model.train()
    
    epoch_losses = {
        'total': [],
        'pix': [],
        'perc': [],
        'ssim': [],
        'grad': [],
        'feat': []
    }
    
    epoch_start_time = time.time()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for batch_idx, (ir, vis, teacher) in enumerate(pbar):
        batch_start_time = time.time()
        
        ir = ir.to(device)
        vis = vis.to(device)
        teacher = teacher.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        with autocast(enabled=args.use_amp):
            # Get student prediction and features
            pred, student_feats = model(ir, vis, return_features=True)
            
            # Compute loss (teacher features can be None if not available)
            loss, loss_dict = criterion(pred, teacher, student_feats, None)
        
        # Backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # Update metrics
        for key in epoch_losses.keys():
            if key in loss_dict:
                epoch_losses[key].append(loss_dict[key])
        
        # 计算速度
        batch_time = time.time() - batch_start_time
        samples_per_sec = ir.size(0) / batch_time
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'pix': f"{loss_dict['pix']:.4f}",
            'ssim': f"{loss_dict['ssim']:.4f}",
            'speed': f"{samples_per_sec:.1f} img/s"
        })
        
        # Logging
        if (batch_idx + 1) % args.log_freq == 0:
            for key, value in loss_dict.items():
                writer.add_scalar(f'train/{key}_loss', value, global_step)
            writer.add_scalar('train/speed', samples_per_sec, global_step)
        
        # Visualization
        if (batch_idx + 1) % args.vis_freq == 0:
            with torch.no_grad():
                # 显示IR, VIS, Teacher, Pred
                n_show = min(4, ir.size(0))
                vis_images = torch.cat([
                    ir[:n_show].repeat(1, 3, 1, 1) if ir.size(1) == 1 else ir[:n_show],  # IR可能是单通道
                    vis[:n_show], 
                    teacher[:n_show], 
                    pred[:n_show]
                ], dim=0)
                writer.add_images('train/visualization', vis_images, global_step, 
                                dataformats='NCHW')
        
        global_step += 1
    
    # Calculate average losses
    avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
    epoch_time = time.time() - epoch_start_time
    
    return avg_losses, global_step, epoch_time


# ============================================
# 验证函数
# ============================================

@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, writer):
    """验证模型"""
    model.eval()
    
    val_losses = {
        'total': [],
        'pix': [],
        'perc': [],
        'ssim': [],
        'grad': [],
        'feat': []
    }
    
    pbar = tqdm(val_loader, desc='Validation')
    
    for batch_idx, (ir, vis, teacher) in enumerate(pbar):
        ir = ir.to(device)
        vis = vis.to(device)
        teacher = teacher.to(device)
        
        # Forward pass
        pred, student_feats = model(ir, vis, return_features=True)
        
        # Compute loss
        loss, loss_dict = criterion(pred, teacher, student_feats, None)
        
        # Update metrics
        for key in val_losses.keys():
            if key in loss_dict:
                val_losses[key].append(loss_dict[key])
        
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'ssim': f"{loss_dict['ssim']:.4f}"
        })
        
        # Visualize first batch
        if batch_idx == 0:
            n_show = min(4, ir.size(0))
            vis_images = torch.cat([
                ir[:n_show].repeat(1, 3, 1, 1) if ir.size(1) == 1 else ir[:n_show],
                vis[:n_show], 
                teacher[:n_show], 
                pred[:n_show]
            ], dim=0)
            writer.add_images('val/visualization', vis_images, epoch, 
                            dataformats='NCHW')
    
    # Calculate average losses
    avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
    
    # Log to tensorboard
    for key, value in avg_losses.items():
        writer.add_scalar(f'val/{key}_loss', value, epoch)
    
    return avg_losses


# ============================================
# 主训练循环
# ============================================

def main():
    args = get_args()
    
    # 打印配置
    print("="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Data paths:")
    print(f"  IR:      {args.ir_root}")
    print(f"  VIS:     {args.vis_root}")
    print(f"  Teacher: {args.teacher_root}")
    print(f"\nTraining settings:")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Input size:  {args.input_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Mixed precision: {args.use_amp}")
    print("="*80)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create dataloaders (自动划分数据集)
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        args.ir_root,
        args.vis_root,
        args.teacher_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        misalign_prob=args.misalign_prob,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"\nDataloader info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # 初始化时间估算器
    timer = TrainingTimer(args.epochs, len(train_loader))
    
    # 估算单个epoch时间（基于第一个batch）
    print("\n" + "="*80)
    print("ESTIMATING TRAINING TIME")
    print("="*80)
    
    # Create model
    print("\nCreating model...")
    model = create_student_model(
        in_channels=args.in_channels,
        out_channels=args.out_channels
    ).to(device)
    
    # 快速测试一个batch来估算时间
    test_batch = next(iter(train_loader))
    ir_test = test_batch[0].to(device)
    vis_test = test_batch[1].to(device)
    
    # Warmup
    model.train()
    with torch.no_grad():
        _ = model(ir_test, vis_test)
    
    # 测量时间
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        _ = model(ir_test, vis_test)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    single_batch_time = time.time() - start
    estimated_epoch_time = single_batch_time * len(train_loader) * 1.5  # 1.5x buffer for backward pass
    estimated_total_time = estimated_epoch_time * args.epochs
    
    print(f"\nTime estimates:")
    print(f"  Single batch (forward): {single_batch_time*1000:.2f}ms")
    print(f"  Estimated epoch time:   {TrainingTimer.format_time(estimated_epoch_time)}")
    print(f"  Estimated total time:   {TrainingTimer.format_time(estimated_total_time)}")
    print(f"  Expected completion:    {(datetime.datetime.now() + datetime.timedelta(seconds=estimated_total_time)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create loss criterion
    print("\nInitializing loss functions...")
    criterion = FusionLoss(
        lambda_pix=args.lambda_pix,
        lambda_perc=args.lambda_perc,
        lambda_ssim=args.lambda_ssim,
        lambda_grad=args.lambda_grad,
        lambda_feat=args.lambda_feat,
        device=device
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Create gradient scaler for AMP
    scaler = GradScaler(enabled=args.use_amp)
    
    # Tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Training loop
    print("\n" + "="*80)
    print(f"STARTING TRAINING")
    print("="*80)
    print(f"Training from epoch {start_epoch} to {args.epochs}")
    print("="*80 + "\n")
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_losses, global_step, epoch_train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args, writer, global_step
        )
        
        # Validate
        val_losses = validate(model, val_loader, criterion, device, epoch, writer)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        
        # Update timer and get time estimates
        epoch_time = time.time() - epoch_start_time
        time_info = timer.update_epoch(epoch, epoch_time)
        
        # Print epoch summary with detailed timing
        print("\n" + "="*80)
        print(f"EPOCH {epoch+1}/{args.epochs} SUMMARY")
        print("="*80)
        print(f"Time:")
        print(f"  This epoch:    {TrainingTimer.format_time(time_info['epoch_time'])}")
        print(f"  Avg epoch:     {TrainingTimer.format_time(time_info['avg_epoch_time'])}")
        print(f"  Total elapsed: {TrainingTimer.format_time(time_info['elapsed'])}")
        print(f"  Est. remaining: {TrainingTimer.format_time(time_info['estimated_remaining'])}")
        print(f"  Est. total:    {TrainingTimer.format_time(time_info['estimated_total'])}")
        print(f"  ETA:           {time_info['eta'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nLearning Rate: {current_lr:.6f}")
        print(f"\nLosses:")
        print(f"  Train Total: {train_losses['total']:.4f}")
        print(f"  Train SSIM:  {train_losses['ssim']:.4f}")
        print(f"  Val Total:   {val_losses['total']:.4f}")
        print(f"  Val SSIM:    {val_losses['ssim']:.4f}")
        
        # Save checkpoint
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']
            print(f"\n✓ New best validation loss: {best_val_loss:.4f}")
        
        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_val_loss,
                train_losses,
                val_losses
            )
            
            if is_best:
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    best_val_loss,
                    train_losses,
                    val_losses
                )
                print(f"✓ Best model saved to {best_path}")
        
        print("="*80 + "\n")
    
    # Training completed
    total_training_time = time.time() - training_start_time
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Total training time: {TrainingTimer.format_time(total_training_time)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print(f"TensorBoard logs: {args.log_dir}")
    print("="*80)
    
    writer.close()


if __name__ == '__main__':
    main()
