# lip-sync-full-framework
# 潜在扩散模型对口型系统

这是一个基于与LatentSync相同技术路线的对口型系统，采用音频驱动的潜在扩散模型实现唇形同步。

## 核心技术

1. **音频驱动的潜在扩散模型** - 在潜在空间中直接生成唇形动作
2. **TREPA时间一致性优化** - 确保视频帧间的连贯性
3. **SyncNet监督机制** - 保证音视频同步精度

## 安装步骤

1. 克隆仓库并安装依赖：pip install -r requirements.txt
2. 下载预训练模型：
   - 主模型：[latent_sync_model.pth](https://example.com/models/latent_sync_model.pth)
   - 面部特征点模型：[shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

3. 将模型放入pretrained文件夹

## 使用方法

启动Web界面：python app.py --mode web --port 7860
启动API服务：python app.py --mode api --port 8000
## 技术细节

系统主要由以下部分组成：
- 音频处理器：提取MFCC特征
- 视频处理器：提取面部特征点和处理视频帧
- 潜在扩散模型：核心生成模型，基于音频特征生成唇形
- Web/API接口：提供用户交互界面
