# In generate_virtual_features.py (FIXED: Adaptive Noise & CFG)

import torch
import numpy as np
import os
import argparse
import dill
from sentence_transformers import SentenceTransformer
from train_feature_diffusion import FeatureDiffusion 

@torch.no_grad()
def generate_with_cfg(model, text_embeddings, num_samples, device, guidance_scale):
    """
    使用 Classifier-Free Guidance (CFG) 进行特征生成
    """
    model.eval()
    shape = (num_samples, model.feature_dim)
    xt = torch.randn(shape, device=device)
    
    # 准备无条件嵌入 (全0)
    uncond_embedding = torch.zeros_like(text_embeddings)
    
    for i in reversed(range(model.betas.shape[0])):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        
        # 进行两次预测：一次有条件(文本)，一次无条件(空)
        pred_noise_cond = model(xt, t, text_embeddings)
        pred_noise_uncond = model(xt, t, uncond_embedding)
        
        # CFG公式: pred = pred_uncond + w * (pred_cond - pred_uncond)
        pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
        
        alphas_t = model._extract(1. - model.betas, t, xt.shape)
        alphas_cumprod_t = model._extract(model.alphas_cumprod, t, xt.shape)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / alphas_t)
        
        model_mean = sqrt_recip_alphas_t * (xt - (1. - alphas_t) / torch.sqrt(1. - alphas_cumprod_t) * pred_noise)
        
        if i != 0:
            noise = torch.randn_like(xt)
            posterior_variance_t = model._extract(model.betas, t, xt.shape)
            xt = model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            xt = model_mean
            
    return xt.cpu().numpy()

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载真实特征数据 (用于获取维度和计算噪声基准)
    print(f"Loading real features from: {args.real_features_path}")
    temp_data = np.load(args.real_features_path)
    real_features = temp_data['features']
    real_labels = temp_data['labels']
    
    feature_dim = real_features.shape[1]
    
    # 2. 加载模型和文本编码器
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    text_dim = text_encoder.get_sentence_embedding_dimension()
    
    model = FeatureDiffusion(feature_dim, text_dim).to(device)
    print(f"Loading diffusion model from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 3. 加载标签映射
    gzsl_meta_path = os.path.join(args.preprocessed_data_dir, 'test_gzsl_meta.pkl')
    with open(gzsl_meta_path, 'rb') as f:
        gzsl_meta = dill.load(f)
    global_text_to_idx = {v: k for k, v in gzsl_meta['label_text_dict'].items()}

    unseen_meta_path = os.path.join(args.preprocessed_data_dir, 'test_unseen_meta.pkl')
    with open(unseen_meta_path, 'rb') as f:
        unseen_meta = dill.load(f)
    unseen_texts = list(unseen_meta['label_text_dict'].keys())
    
    print(f"Generating features for {len(unseen_texts)} unseen classes. Scale={args.guidance_scale}")
    
    # 4. 生成虚拟unseen特征
    all_virtual_features, all_virtual_labels = [], []
    for text in unseen_texts:
        global_label = global_text_to_idx[text]
        
        # 编码文本
        text_embeddings = text_encoder.encode([text] * args.num_per_class, convert_to_tensor=True, device=device)
        
        # 使用CFG生成
        virtual_features = generate_with_cfg(model, text_embeddings, args.num_per_class, device, args.guidance_scale)
        
        all_virtual_features.append(virtual_features)
        all_virtual_labels.extend([global_label] * args.num_per_class)
        
    all_virtual_features = np.concatenate(all_virtual_features, axis=0)

    # ==================== ★★★ 核心修改：自适应噪声注入 ★★★ ====================
    print("Calculating adaptive noise levels...")
    
    # 计算真实 Seen 类别的平均类内标准差 (Average Intra-class STD)
    std_list = []
    unique_labels = np.unique(real_labels)
    for label in unique_labels:
        cls_feats = real_features[real_labels == label]
        if len(cls_feats) > 1:
            std_list.append(np.std(cls_feats))
            
    if std_list:
        avg_real_std = np.mean(std_list)
    else:
        avg_real_std = 1.0 # 兜底
        
    # 设置噪声强度为真实类内离散度的一定比例 (例如 0.5 倍)
    # 这个系数越小，生成的特征越紧凑；越大，特征越发散
    # noise_scale_factor = 0.5

    noise_scale_factor = 0.0
    adaptive_noise_std = avg_real_std * noise_scale_factor

    print(f"   -> Real Data Avg Std: {avg_real_std:.4f}")
    print(f"   -> Injecting Noise Std: {adaptive_noise_std:.4f} (Factor: {noise_scale_factor})")

    noise = np.random.normal(0, 1, all_virtual_features.shape) * adaptive_noise_std
    all_virtual_features = all_virtual_features + noise
    # ==================== ★★★ 修改结束 ★★★ ====================

    # ==================== [Fixed] Spectral Calibration with Energy Restoration ====================
    print("正在执行物理惯性约束 (Spectral Calibration)...")

    # 1. 准备数据
    # 假设 real_features 是真实 Seen 数据，用来做能量参考
    # all_virtual_features 是生成的 Unseen 数据
    # 务必保证此时 all_virtual_features 还没和 real_features 拼接！

    # --- 步骤 A: 计算真实数据的平均能量 (模长) ---
    # 这一步至关重要！我们要保证生成的数据和真实数据在同一个量级
    real_norms = np.linalg.norm(real_features, axis=1)
    target_avg_norm = np.mean(real_norms)
    print(f"Target Norm (from Real Data): {target_avg_norm:.4f}")

    # --- 步骤 B: SVD 分解与衰减 ---
    U, S, Vh = np.linalg.svd(all_virtual_features, full_matrices=False)

    n_components = len(S)
    # 稍微温和一点的衰减：从 1.0 -> 0.4 (之前是 0.15，太狠了)
    # 既能让红线弯曲，又不至于损失太多信息
    decay_factor = np.logspace(0, np.log10(0.4), n_components)

    # 保护前 15 个主成分 (语义核心) 不受影响
    decay_factor[:15] = 1.0 
    # 平滑过渡
    decay_factor[15:25] = np.linspace(1.0, decay_factor[25], 10)

    S_calibrated = S * decay_factor
    all_virtual_features_calibrated = U @ np.diag(S_calibrated) @ Vh

    # --- 步骤 C: 能量恢复 (Energy Restoration) ---
    # 计算校准后的当前模长
    current_norms = np.linalg.norm(all_virtual_features_calibrated, axis=1, keepdims=True)
    # 重新缩放：把每个样本的模长拉回到 target_avg_norm
    # 加上 1e-8 防止除以零
    all_virtual_features_final = all_virtual_features_calibrated * (target_avg_norm / (current_norms + 1e-8))

    print(f"Calibration Done. Norm restored to ~{target_avg_norm:.4f}")

    # 5. 更新变量
    all_virtual_features = all_virtual_features_final
    # ======================================================================================
    
    # 5. 合并并保存
    final_features = np.concatenate([real_features, all_virtual_features], axis=0)
    final_labels = np.concatenate([real_labels, all_virtual_labels], axis=0)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez(args.output_path, features=final_features, labels=final_labels)
    print(f"\nSaved {len(final_labels)} total features to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_data_dir', default='./data_preprocessed/PAMAP2')
    parser.add_argument('--real_features_path', default='./data_preprocessed/PAMAP2/real_seen_features.npz')
    parser.add_argument('--model_path', default='results_GZSL_Feature/PAMAP2/feature_diffusion_cfg.pth') 
    parser.add_argument('--output_path', default='./data_preprocessed/PAMAP2/final_hybrid_features_cfg.npz')
    parser.add_argument('--num_per_class', type=int, default=500, help="每类生成的样本数")
    parser.add_argument('--guidance_scale', type=float, default=3.0, help="CFG引导强度")
    args = parser.parse_args()
    main(args)