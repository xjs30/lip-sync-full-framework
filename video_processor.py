import cv2
import numpy as np
import dlib
import torch
from PIL import Image
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip, AudioFileClip

class VideoProcessor:
    def __init__(self):
        # 面部检测器和特征点预测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("pretrained/shape_predictor_68_face_landmarks.dat")
        
        # 图像转换
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def load_video(self, video_path):
        """加载视频并提取帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
        cap.release()
        return np.array(frames), fps
    
    def detect_face_landmarks(self, frame):
        """检测面部特征点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
            
        # 取第一个检测到的脸
        shape = self.predictor(gray, faces[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        
        # 只提取唇部相关特征点（48-67）
        lip_landmarks = landmarks[48:68]
        return lip_landmarks
    
    def extract_face_features(self, frames):
        """提取视频帧中的面部特征"""
        features = []
        for frame in frames:
            landmarks = self.detect_face_landmarks(frame)
            if landmarks is not None:
                # 标准化
                mean = np.mean(landmarks, axis=0, keepdims=True)
                std = np.std(landmarks, axis=0, keepdims=True) + 1e-6
                norm_landmarks = (landmarks - mean) / std
                features.append(norm_landmarks.flatten())
            else:
                # 如果没有检测到脸，使用上一帧的特征
                if features:
                    features.append(features[-1])
                else:
                    features.append(np.zeros(40))  # 20个点 × 2个坐标
        
        # 确保特征长度与音频匹配
        return np.array(features)
    
    def postprocess_frames(self, generated_frames, original_frames):
        """后处理生成的帧，与原始视频融合"""
        processed = []
        generated_frames = (generated_frames * 255).astype(np.uint8)
        
        for gen_frame, orig_frame in zip(generated_frames, original_frames):
            # 调整生成帧大小以匹配原始帧
            gen_resized = cv2.resize(gen_frame.transpose(1, 2, 0), 
                                    (orig_frame.shape[1], orig_frame.shape[0]))
            
            # 简单融合（实际应用中可使用更复杂的面部融合技术）
            mask = self.detect_face_mask(orig_frame)
            if mask is not None:
                # 只替换面部区域
                result = orig_frame.copy()
                result[mask] = gen_resized[mask]
                processed.append(result)
            else:
                processed.append(gen_resized)
                
        return np.array(processed)
    
    def detect_face_mask(self, frame):
        """生成面部掩码用于融合"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
            
        # 创建掩码
        mask = np.zeros_like(gray, dtype=bool)
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            mask[y1:y2, x1:x2] = True
            
        return mask
    
    def save_video(self, frames, fps, audio_path, output_path):
        """保存视频并添加音频"""
        # 先保存无音频的视频
        temp_video = "temp_video.mp4"
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        for frame in frames:
            # 转换回BGR格式
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        
        out.release()
        
        # 添加音频
        video_clip = VideoFileClip(temp_video)
        audio_clip = AudioFileClip(audio_path)
        
        # 确保音频和视频长度一致
        final_clip = video_clip.set_audio(audio_clip.set_duration(video_clip.duration))
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        # 清理临时文件
        video_clip.close()
        audio_clip.close()
        if os.path.exists(temp_video):
            os.remove(temp_video)
