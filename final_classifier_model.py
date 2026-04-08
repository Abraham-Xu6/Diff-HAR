# In final_classifier_model.py

import torch.nn as nn

class MLPClassifier(nn.Module):
    """
    一个简单的MLP分类器，用于替代nn.Linear。
    具备更强的非线性学习能力。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        
        # ★★★ 修改点：替换为单层线性网络 ★★★
        # 直接从输入维度映射到输出维度，没有隐藏层，没有ReLU
        # self.main = nn.Linear(input_dim, output_dim)

        # ★★★ 提升点 3: 使用更深、更宽的MLP ★★★
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), # 例如，将第一层加宽
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), # 增加一个中间层
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
 
        # # ★★★ 基础版: 双层MLP ★★★
        # self.main = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim), # 恢复
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, output_dim) # 恢复
        # )
    
    def forward(self, x):
        return self.main(x)