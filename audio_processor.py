import librosa
import numpy as np
import torch
import soundfile as sf

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mfcc=80, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        
    def load_audio(self, audio_path):
        """加载音频并转换为指定采样率"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio
    
    def extract_mfcc(self, audio):
        """提取MFCC特征"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        # 转置为 (时间步, 特征维度)
        return mfcc.T
    
    def extract_features(self, audio_path):
        """提取用于模型的音频特征"""
        audio = self.load_audio(audio_path)
        mfcc = self.extract_mfcc(audio)
        
        # 标准化
        mean = np.mean(mfcc, axis=0, keepdims=True)
        std = np.std(mfcc, axis=0, keepdims=True) + 1e-6
        mfcc = (mfcc - mean) / std
        
        return mfcc
    
    def resample_audio(self, audio_path, target_path, target_sr=16000):
        """重采样音频用于输出视频"""
        audio, sr = librosa.load(audio_path, sr=target_sr)
        sf.write(target_path, audio, target_sr)
        return target_path
