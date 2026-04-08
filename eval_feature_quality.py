import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from torch.utils.data import DataLoader
from tqdm import tqdm
import dill
import pandas as pd

# 导入你的模块
from classifier_model import SignalClassifier
from evaluate_gzsl import GZSLTestDataset, infer_model_shapes

# ================= 核心计算函数 =================

def calculate_mmd(x, y, sigma=1.0):
    """计算 MMD (Maximum Mean Discrepancy)"""
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    
    # 降采样防止显存爆炸
    if x_size > 1000:
        idx = torch.randperm(x_size)[:1000]
        x = x[idx]
        x_size = 1000
    if y_size > 1000:
        idx = torch.randperm(y_size)[:1000]
        y = y[idx]
        y_size = 1000

    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
    
    x_kernel = torch.exp(-torch.mean((x.unsqueeze(1) - x.unsqueeze(0)) ** 2, dim=2) / sigma)
    y_kernel = torch.exp(-torch.mean((y.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=2) / sigma)
    xy_kernel = torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / sigma)
    
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd

def calculate_kl_divergence(p_features, q_features, num_bins=30):
    """计算平均 KL 散度"""
    kl_sum = 0.0
    dims = p_features.shape[1]
    
    if len(p_features) > 2000: p_features = p_features[np.random.choice(len(p_features), 2000, replace=False)]
    if len(q_features) > 2000: q_features = q_features[np.random.choice(len(q_features), 2000, replace=False)]

    valid_dims = 0
    for i in range(dims):
        p_col = p_features[:, i]
        q_col = q_features[:, i]
        
        min_val = min(p_col.min(), q_col.min())
        max_val = max(p_col.max(), q_col.max())
        
        if max_val - min_val < 1e-6: continue 
        
        p_hist, _ = np.histogram(p_col, bins=num_bins, range=(min_val, max_val), density=True)
        q_hist, _ = np.histogram(q_col, bins=num_bins, range=(min_val, max_val), density=True)
        
        p_hist = p_hist + 1e-7
        q_hist = q_hist + 1e-7
        
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()
        
        kl = stats.entropy(p_hist, q_hist)
        if np.isfinite(kl):
            kl_sum += kl
            valid_dims += 1
            
    return kl_sum / max(valid_dims, 1)

def plot_feature_histograms(real, gen, save_path):
    """绘制特征分布直方图"""
    plt.figure(figsize=(12, 4))
    
    # 挑选 3 个方差最大的维度
    variances = np.var(real, axis=0)
    top_indices = np.argsort(variances)[::-1][:3]
    
    for i, dim_idx in enumerate(top_indices):
        plt.subplot(1, 3, i+1)
        plt.hist(real[:, dim_idx], bins=30, alpha=0.5, label='Real', density=True, color='#1f77b4', edgecolor='black')
        plt.hist(gen[:, dim_idx], bins=30, alpha=0.5, label='Gen', density=True, color='#ff7f0e', edgecolor='#ff7f0e', hatch='//')
        plt.title(f'Feature Dim {dim_idx}', fontsize=10)
        if i == 0: plt.legend(loc='upper right', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Distribution plot saved to {save_path}")

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 提取真实特征 (Real Unseen)
    print("Step 1: Extracting REAL Unseen Features...")
    dataset = GZSLTestDataset(args.preprocessed_data_dir)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    test_unseen_meta_path = os.path.join(args.preprocessed_data_dir, 'test_unseen_meta.pkl')
    with open(test_unseen_meta_path, 'rb') as f: unseen_meta = dill.load(f)
    # 这里的 values 就是 class index
    unseen_label_indices = list(unseen_meta['label_text_dict'].values())
    print(f"   Target Unseen Labels: {unseen_label_indices}")

    fe_state = torch.load(args.feature_extractor_path, map_location=device)
    input_dim, model_dim, num_classes_fe = infer_model_shapes(fe_state)
    
    # ★★★ 核心修复：使用关键字参数防止传参错误 ★★★
    feature_extractor = SignalClassifier(
        input_dim=input_dim, 
        model_dim=model_dim, 
        num_classes=num_classes_fe
    ).to(device)
    
    feature_extractor.load_state_dict(fe_state)
    feature_extractor.eval()
    
    real_unseen_features = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Real"):
            labels = batch['label']
            mask = torch.tensor([l.item() in unseen_label_indices for l in labels], dtype=torch.bool)
            
            if mask.sum() > 0:
                signals = batch['signal'][mask].to(device)
                _, feats = feature_extractor(signals)
                real_unseen_features.append(feats.cpu())
                
    real_unseen_features = torch.cat(real_unseen_features, dim=0).numpy()
    print(f"   -> Extracted {len(real_unseen_features)} Real Unseen samples.")

    # 2. 加载生成特征 (Generated Unseen)
    print("Step 2: Loading GENERATED Unseen Features...")
    
    # 自动寻找最佳文件，找不到则用默认
    if not os.path.exists(args.generated_path):
        fallback = './data_preprocessed/PAMAP2/final_hybrid_features_cfg.npz'
        if os.path.exists(fallback):
             print(f"⚠️ Warning: {args.generated_path} not found. Using {fallback}...")
             args.generated_path = fallback
        else:
             print(f"❌ Error: Cannot find generated features at {args.generated_path}")
             return

    gen_data = np.load(args.generated_path)
    all_gen_feats = gen_data['features']
    all_gen_labels = gen_data['labels']
    
    gen_unseen_list = []
    for lbl in unseen_label_indices:
        # 假设混合文件里 label 为 Unseen 的就是生成的 (且排除了 Real Seen)
        # 更好的策略：Unseen类的样本，通常位于数组后半部分
        # 这里简单起见，只要 label 匹配就拿
        mask = (all_gen_labels == lbl)
        if mask.sum() > 0:
            gen_unseen_list.append(all_gen_feats[mask])
            
    if not gen_unseen_list:
        print("❌ Error: No Unseen labels found in the hybrid file!")
        return
        
    gen_unseen_features = np.concatenate(gen_unseen_list, axis=0)
    print(f"   -> Found {len(gen_unseen_features)} Generated Unseen samples.")
    
    # 3. 计算指标
    print("Step 3: Calculating Metrics...")
    
    n_sample = min(1000, len(real_unseen_features), len(gen_unseen_features))
    idx_r = np.random.choice(len(real_unseen_features), n_sample, replace=False)
    idx_g = np.random.choice(len(gen_unseen_features), n_sample, replace=False)
    
    real_tensor = torch.from_numpy(real_unseen_features[idx_r]).to(device)
    gen_tensor = torch.from_numpy(gen_unseen_features[idx_g]).to(device)
    
    mmd_val = calculate_mmd(real_tensor, gen_tensor).item()
    print(f"🏆 MMD Score: {mmd_val:.5f} (Target: < 0.1)")
    
    kl_val = calculate_kl_divergence(real_unseen_features, gen_unseen_features)
    print(f"🏆 KL Divergence: {kl_val:.5f} (Target: < 0.5)")
    
    # 4. 绘图
    print("Step 4: Plotting Histograms...")
    plot_feature_histograms(real_unseen_features, gen_unseen_features, 'exp_feature_quality_dist.png')
    
    df = pd.DataFrame({'Metric': ['MMD', 'KL Divergence'], 'Score': [mmd_val, kl_val]})
    df.to_csv('exp_feature_quality_metrics.csv', index=False)
    print("✅ Done. Results saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_data_dir', default='./data_preprocessed/PAMAP2')
    parser.add_argument('--feature_extractor_path', default='results_GZSL_Feature/PAMAP2/feature_extractor.pth')
    # 默认尝试加载 best，如果不存在，脚本里会自动降级加载 final
    parser.add_argument('--generated_path', default='./data_preprocessed/PAMAP2/best_hybrid_features.npz')
    args = parser.parse_args()
    
    # 路径容错
    if not os.path.exists(args.feature_extractor_path):
         if os.path.exists('results_GZSL_Feature/feature_extractor.pth'):
             args.feature_extractor_path = 'results_GZSL_Feature/feature_extractor.pth'
             
    main(args)