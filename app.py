import os
import torch
import numpy as np
import gradio as gr
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
import tempfile
import cv2
from datetime import datetime
import asyncio
from models.latent_diffusion import LatentDiffusionModel
from utils.audio_processor import AudioProcessor
from utils.video_processor import VideoProcessor

# 初始化应用
app = FastAPI(title="LatentSync风格对口型系统")
processor = gr.Blocks(title="智能唇形同步系统")

# 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "pretrained/latent_sync_model.pth"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型和处理器
audio_processor = AudioProcessor()
video_processor = VideoProcessor()

# 加载预训练模型
print(f"加载模型到 {DEVICE}...")
model = LatentDiffusionModel(image_size=64)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()
print("模型加载完成")

# 处理函数
def process_lip_sync(video_path, audio_path, progress=gr.Progress()):
    try:
        # 1. 预处理
        progress(0, desc="处理输入数据")
        video_frames, fps = video_processor.load_video(video_path)
        audio_features = audio_processor.extract_features(audio_path)
        
        # 2. 提取面部特征
        progress(0.3, desc="提取面部特征")
        face_features = video_processor.extract_face_features(video_frames)
        
        # 3. 模型推理
        progress(0.5, desc="生成唇形同步视频")
        with torch.no_grad():
            # 转换为张量
            audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            face_tensor = torch.tensor(face_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # 生成视频
            output_tensor = model.sample(
                audio_feat=audio_tensor,
                face_feat=face_tensor,
                num_steps=30
            )
        
        # 4. 后处理
        progress(0.8, desc="处理输出视频")
        output_frames = output_tensor.squeeze(0).cpu().numpy()
        output_frames = video_processor.postprocess_frames(output_frames, video_frames)
        
        # 5. 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.mp4")
        video_processor.save_video(output_frames, fps, audio_path, output_path)
        
        progress(1.0, desc="完成")
        return output_path
        
    except Exception as e:
        print(f"处理错误: {str(e)}")
        raise gr.Error(f"处理失败: {str(e)}")

# 构建Gradio界面
with processor:
    gr.Markdown("# 🎭 智能唇形同步系统")
    gr.Markdown("基于潜在扩散模型的音频驱动唇形生成，与LatentSync采用相同技术路线")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="输入视频")
            audio_input = gr.Audio(label="输入音频", type="filepath")
            generate_btn = gr.Button("生成同步视频", variant="primary")
        
        with gr.Column(scale=1):
            video_output = gr.Video(label="输出视频")
    
    generate_btn.click(
        fn=process_lip_sync,
        inputs=[video_input, audio_input],
        outputs=video_output
    )
    
    gr.Markdown("""
    ### 技术说明
    - 采用音频驱动的潜在扩散模型
    - 集成TREPA时间一致性优化
    - 使用SyncNet监督机制确保音视频同步
    """)

# API端点
@app.post("/api/generate")
async def api_generate(video: UploadFile = File(...), audio: UploadFile = File(...)):
    # 保存上传文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as af:
        
        vf.write(await video.read())
        af.write(await audio.read())
        video_path = vf.name
        audio_path = af.name
    
    # 处理
    output_path = process_lip_sync(video_path, audio_path)
    
    # 清理临时文件
    os.unlink(video_path)
    os.unlink(audio_path)
    
    return FileResponse(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["web", "api"], default="web")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    if args.mode == "web":
        processor.launch(server_name="0.0.0.0", server_port=args.port)
    else:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
