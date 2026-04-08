# In train_feature_extractor.py (FINAL FIX v2)

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import dill
from classifier_model import SignalClassifier

class RealSeenTrainDataset(Dataset):
    def __init__(self, preprocessed_data_dir):
        train_meta_path = os.path.join(preprocessed_data_dir, 'train_meta.pkl')
        data_path = os.path.join(preprocessed_data_dir, 'processed_data.npy')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"在 '{preprocessed_data_dir}' 中未找到 'processed_data.npy'。请先运行步骤1。")
            
        self.all_real_data = np.load(data_path)
        with open(train_meta_path, 'rb') as f:
            self.meta = dill.load(f)
        
        # ★★★ 核心修复：采用更稳健的标准化方法 ★★★
        # 无论输入是 2D 还是 3D，都将其视为 (批次数, 特征数) 进行标准化
        if self.all_real_data.ndim == 3:
            # 对于3D数据 (N, L, E)，我们先 reshape 成 (N*L, E)
            original_shape = self.all_real_data.shape
            num_features = original_shape[2]
            data_reshaped = self.all_real_data.reshape(-1, num_features)
            
            mean = np.mean(data_reshaped, axis=0)
            std = np.std(data_reshaped, axis=0)
            std[std == 0] = 1.0
            
            # 标准化后 reshape 回原始 3D 形状
            self.all_real_data = ((data_reshaped - mean) / std).reshape(original_shape)
        elif self.all_real_data.ndim == 2:
             # 对于已经是 2D 的数据 (N, E)，直接标准化
            mean = np.mean(self.all_real_data, axis=0)
            std = np.std(self.all_real_data, axis=0)
            std[std == 0] = 1.0
            self.all_real_data = (self.all_real_data - mean) / std
        else:
            print("警告：数据维度不为2或3，跳过标准化。")


    def __len__(self):
        return len(self.meta['data_list'])

    def __getitem__(self, idx):
        item = self.meta['data_list'][idx]
        signal_np = self.all_real_data[item['index']].astype(np.float32)
        
        # ★★★ 核心修复：确保 signal 总是 2D 张量 (Seq_Len, Features) ★★★
        # Transformer模型期望至少2D的输入（序列），即使序列长度为1
        if signal_np.ndim == 1:
            signal_np = np.expand_dims(signal_np, axis=0) # 将 (features,) 转换为 (1, features)

        signal = torch.from_numpy(signal_np)
        
        local_label = item['label']
        text_label = item['text']
        
        return {
            'signal': signal,
            'label': torch.tensor(local_label, dtype=torch.long),
            'text': text_label
        }

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = RealSeenTrainDataset(args.preprocessed_data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    if len(train_dataset.all_real_data) == 0:
        raise ValueError("预处理后的数据为空，无法确定模型输入维度。")
    
    # 动态获取输入维度
    input_dim = train_dataset[0]['signal'].shape[-1]
    
    num_seen_classes = len(train_dataset.meta['label_text_dict'])
    # 使用从数据推断出的 input_dim 初始化模型
    model = SignalClassifier(input_dim=input_dim, num_classes=num_seen_classes).to(device)
    
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Training feature extractor on {num_seen_classes} REAL SEEN classes...")
    for epoch in range(args.epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            signals, labels = batch['signal'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            logits, _ = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_model_path)
    print(f"\nFeature Extractor saved to {args.output_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_data_dir', default='./data_preprocessed/PAMAP2')
    parser.add_argument('--output_model_path', default='results_GZSL_Feature/feature_extractor.pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)