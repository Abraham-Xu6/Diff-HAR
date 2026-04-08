# In train_final_classifier.py (FIXED CUDA ASSERT ERROR)

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
import numpy as np

# 导入新的MLP模型
from final_classifier_model import MLPClassifier

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 加载最终的混合特征数据集 .npz
    print(f"Loading final hybrid feature dataset from: {args.data_path}")
    data = np.load(args.data_path)
    
    # 转换为 Tensor
    all_features = torch.from_numpy(data['features']).float()
    all_labels = torch.from_numpy(data['labels']).long() # 确保是 long 类型
    
    feature_dim = all_features.shape[1]
    
    # =========================================================================
    # ★★★ 核心修复：更安全的类别数计算方式 ★★★
    # 不要使用 len(np.unique(labels))，因为它在缺少某类样本时会计算错误。
    # 使用 (max_label_index + 1) 可以确保涵盖所有标签索引。
    # =========================================================================
    num_classes = int(torch.max(all_labels).item()) + 1
    
    print(f"Model Configuration: Input Dim={feature_dim}, Output Classes={num_classes} (Max Label ID: {num_classes-1})")

    feature_dataset = TensorDataset(all_features, all_labels)
    
    # 2. 使用加权采样器来确保平衡训练
    # 注意：bincount 需要 minlength=num_classes 确保维度对齐
    class_counts = torch.bincount(all_labels, minlength=num_classes)
    class_weights = 1. / (class_counts.float() + 1e-8)
    sample_weights = class_weights[all_labels]
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), True)
    
    train_loader = DataLoader(feature_dataset, batch_size=args.batch_size, sampler=sampler)

    # 3. 实例化MLP分类器
    final_classifier = MLPClassifier(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim, 
        output_dim=num_classes # 使用修正后的类别数
    ).to(device)

    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(final_classifier.parameters(), lr=args.lr)

    print("Training the final MLP classifier on hybrid features...")
    for epoch in range(args.epochs):
        # 这里的tqdm可能会因为前面报错而显示异常，可以加上 leave=False
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = final_classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    # 4. 保存最终分类器
    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    torch.save(final_classifier.state_dict(), args.output_model_path)
    print(f"\nFinal classifier saved to {args.output_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data_preprocessed/PAMAP2/final_hybrid_features.npz')
    parser.add_argument('--output_model_path', default='results_GZSL_Feature/final_classifier.pth')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dimension for the MLP classifier.")
    args = parser.parse_args()
    main(args)