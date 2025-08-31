import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

class TimeEmbedding(nn.Module):
    """时间步嵌入模块"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = torch.exp(torch.arange(self.half_dim) * -self.emb)

    def forward(self, x):
        x = x.unsqueeze(1)
        emb = x * self.emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb

class AudioEncoder(nn.Module):
    """音频特征编码器"""
    def __init__(self, input_dim=80, hidden_dim=256, output_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

class FaceEncoder(nn.Module):
    """面部特征编码器"""
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

class UNetBlock(nn.Module):
    """UNet基本模块"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        
        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # 加入时间嵌入
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        # 残差连接
        return h + self.residual_conv(x)

class DownBlock(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.block = UNetBlock(in_channels, out_channels, time_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        
    def forward(self, x, t):
        x = self.block(x, t)
        x_down = self.downsample(x)
        return x, x_down

class UpBlock(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.block = UNetBlock(in_channels + out_channels, out_channels, time_dim)
        
    def forward(self, x, skip_x, t):
        x = self.upsample(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.block(x, t)
        return x

class LatentDiffusionModel(nn.Module):
    """音频驱动的潜在扩散模型，与LatentSync技术路线一致"""
    def __init__(self, 
                 image_size=64, 
                 in_channels=3,
                 out_channels=3,
                 base_channels=64,
                 time_emb_dim=256,
                 audio_dim=512,
                 face_dim=512):
        super().__init__()
        
        self.image_size = image_size
        self.base_channels = base_channels
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 条件编码器
        self.audio_encoder = AudioEncoder(output_dim=audio_dim)
        self.face_encoder = FaceEncoder(output_dim=face_dim)
        
        # 条件投影
        self.audio_proj = nn.Linear(audio_dim, time_emb_dim)
        self.face_proj = nn.Linear(face_dim, time_emb_dim)
        
        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList([
            DownBlock(base_channels, base_channels * 2, time_emb_dim),
            DownBlock(base_channels * 2, base_channels * 4, time_emb_dim),
            DownBlock(base_channels * 4, base_channels * 8, time_emb_dim),
        ])
        
        # 中间块
        self.mid_block = UNetBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList([
            UpBlock(base_channels * 8, base_channels * 4, time_emb_dim),
            UpBlock(base_channels * 4, base_channels * 2, time_emb_dim),
            UpBlock(base_channels * 2, base_channels, time_emb_dim),
        ])
        
        # 最终输出
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
        # 时间一致性损失模块 (TREPA)
        self.temporal_aligner = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        )
        
        # SyncNet监督头
        self.syncnet_head = nn.Sequential(
            nn.Conv2d(out_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x, t, audio_feat, face_feat):
        # 计算时间嵌入
        time_emb = self.time_embedding(t)
        time_emb = self.time_mlp(time_emb)
        
        # 计算条件嵌入
        batch_size, seq_len = audio_feat.shape[0], audio_feat.shape[1]
        
        # 对序列特征取平均
        audio_avg = torch.mean(audio_feat, dim=1)
        face_avg = torch.mean(face_feat, dim=1)
        
        # 投影到时间嵌入维度
        audio_emb = self.audio_proj(audio_avg)
        face_emb = self.face_proj(face_avg)
        
        # 合并所有条件
        cond_emb = time_emb + audio_emb + face_emb
        
        # 初始卷积
        x = self.init_conv(x)
        
        # 下采样
        skips = []
        for down in self.down_blocks:
            h, x = down(x, cond_emb)
            skips.append(h)
        
        # 中间块
        x = self.mid_block(x, cond_emb)
        
        # 上采样
        for i, up in enumerate(self.up_blocks):
            x = up(x, skips[-(i+1)], cond_emb)
        
        # 最终输出
        x = self.final_conv(x)
        
        return x
    
    def temporal_consistency_loss(self, frames):
        # 调整形状以适应3D卷积
        frames = frames.permute(0, 2, 1, 3, 4)
        
        # 应用时间卷积
        aligned = self.temporal_aligner(frames)
        
        # 计算与原始帧的差异
        loss = F.l1_loss(aligned, frames)
        return loss
    
    def syncnet_loss(self, generated_frames, audio_feat):
        batch_size, seq_len = generated_frames.shape[0], generated_frames.shape[1]
        
        # 计算每帧的同步分数
        sync_scores = []
        for i in range(seq_len):
            frame = generated_frames[:, i]
            score = self.syncnet_head(frame)
            sync_scores.append(score)
        
        sync_scores = torch.stack(sync_scores, dim=1)
        
        # 计算与音频特征的相关性损失
        audio_feat_flat = audio_feat.mean(dim=2, keepdim=True)
        loss = F.mse_loss(sync_scores, audio_feat_flat)
        
        return loss
    
    @torch.no_grad()
    def sample(self, audio_feat, face_feat, num_steps=30, guidance_scale=3.0):
        batch_size, seq_len = audio_feat.shape[0], audio_feat.shape[1]
        device = audio_feat.device
        
        # 初始化噪声
        x = torch.randn(batch_size, seq_len, 3, self.image_size, self.image_size, device=device)
        x = x.view(-1, 3, self.image_size, self.image_size)
        
        # 时间步
        timesteps = torch.linspace(1, 0, num_steps, device=device) * 1000
        
        # 生成进度
        for step in reversed(range(num_steps)):
            t = torch.full((batch_size * seq_len,), timesteps[step], device=device, dtype=torch.float32)
            
            # 扩展特征以匹配批处理大小
            audio_expanded = audio_feat.repeat_interleave(seq_len, dim=0)
            face_expanded = face_feat.repeat_interleave(seq_len, dim=0)
            
            # 双重采样用于分类器引导
            x_input = torch.cat([x, x], dim=0)
            t_input = torch.cat([t, t], dim=0)
            audio_input = torch.cat([audio_expanded, torch.zeros_like(audio_expanded)], dim=0)
            face_input = torch.cat([face_expanded, torch.zeros_like(face_expanded)], dim=0)
            
            # 预测噪声
            noise_pred = self.forward(x_input, t_input, audio_input, face_input)
            
            # 拆分预测结果
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # 应用扩散步骤
            alpha = 1.0 - (step / num_steps)
            x = (x - (1 - alpha) * noise_pred) / torch.sqrt(alpha)
        
        # 重塑为视频序列
        x = x.view(batch_size, seq_len, 3, self.image_size, self.image_size)
        # 转换为0-1范围
        x = torch.sigmoid(x)
        
        return x
