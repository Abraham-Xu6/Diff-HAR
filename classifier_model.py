# # In classifier_model.py

# import torch
# import torch.nn as nn

# class SignalClassifier(nn.Module):
#     """一个用于时间序列分类的简单Transformer编码器模型。"""
#     def __init__(self, input_dim=36, model_dim=96, num_heads=4, num_layers=4, num_classes=12, dropout=0.1):
#         super().__init__()
#         self.input_proj = nn.Linear(input_dim, model_dim)
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=model_dim, 
#             nhead=num_heads, 
#             dropout=dropout, 
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer, 
#             num_layers=num_layers
#         )
        
#         self.pool = lambda x: x.mean(dim=1)
#         self.output_layer = nn.Linear(model_dim, num_classes)

#     def forward(self, x):
#         # x shape: (batch_size, seq_len, input_dim)
#         x = self.input_proj(x)
#         # x shape: (batch_size, seq_len, model_dim)
        
#         x = self.transformer_encoder(x)
#         # x shape: (batch_size, seq_len, model_dim)
        
#         # ★★★ 核心修改：在池化后，我们得到了用于分类的特征向量 ★★★
#         features = self.pool(x)
#         # features shape: (batch_size, model_dim)
        
#         logits = self.output_layer(features)
#         # logits shape: (batch_size, num_classes)
        
#         # ★★★ 返回 logits 和 features ★★★
#         return logits, features


# classifier_model.py (通用增强版)，升级模型架构（引入 CNN 主干）
# 由于 mmWave 和 USC-HAR 数据维度较低，单纯 Transformer 表现不佳
import torch
import torch.nn as nn

class SignalClassifier(nn.Module):
    """
    升级版 Transformer 编码器。
    引入 Conv1d 主干，同时兼容 PAMAP2 (高维) 和 mmWave (低维)。
    """
    def __init__(self, input_dim=36, model_dim=96, num_heads=4, num_layers=4, num_classes=12, dropout=0.1):
        super().__init__()
        
        # ★★★ 核心升级：双层卷积主干 ★★★
        # 这一层能像人眼看波形一样，捕捉局部的时间变化
        self.input_proj = nn.Sequential(
            # 第一层卷积：从原始维度映射到模型维度，提取初级波形
            nn.Conv1d(input_dim, model_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 第二层卷积：进一步整合特征
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU()
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=model_dim * 2, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.pool = lambda x: x.mean(dim=1)
        self.output_layer = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # x input shape: (batch_size, seq_len, input_dim)
        
        # 1. 调整维度以适配 Conv1d (batch, dim, seq)
        x = x.permute(0, 2, 1) 
        
        # 2. 通过卷积主干提取波形特征
        x = self.input_proj(x)
        
        # 3. 调整回 Transformer 需要的维度 (batch, seq, dim)
        x = x.permute(0, 2, 1)
        
        x = self.transformer_encoder(x)
        
        features = self.pool(x)
        logits = self.output_layer(features)
        
        return logits, features