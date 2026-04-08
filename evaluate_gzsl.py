# In evaluate_gzsl.py (COMPLETE FINAL VERSION)

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import dill
import pandas as pd  # 新增 pandas 用于保存 CSV
from sklearn.metrics import accuracy_score, f1_score

from classifier_model import SignalClassifier
from final_classifier_model import MLPClassifier

def _standardize_data(data):
    """辅助函数，用于统一标准化逻辑"""
    if data.size == 0: return data
    if data.ndim == 3:
        original_shape = data.shape
        num_features = original_shape[2]
        data_reshaped = data.reshape(-1, num_features)
        mean = np.mean(data_reshaped, axis=0)
        std = np.std(data_reshaped, axis=0)
        std[std == 0] = 1.0
        return ((data_reshaped - mean) / std).reshape(original_shape)
    elif data.ndim == 2:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1.0
        return (data - mean) / std
    else:
        return data

class GZSLTestDataset(Dataset):
    def __init__(self, preprocessed_data_dir):
        gzsl_meta_path = os.path.join(preprocessed_data_dir, 'test_gzsl_meta.pkl')
        data_path = os.path.join(preprocessed_data_dir, 'processed_data.npy')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"在 '{preprocessed_data_dir}' 中未找到 'processed_data.npy'。")
        self.all_real_data = np.load(data_path)
        with open(gzsl_meta_path, 'rb') as f: self.meta = dill.load(f)
        self.all_real_data = _standardize_data(self.all_real_data)

    def __len__(self): return len(self.meta['data_list'])

    def __getitem__(self, idx):
        item = self.meta['data_list'][idx]
        signal_np = self.all_real_data[item['index']].astype(np.float32)
        if signal_np.ndim == 1: signal_np = np.expand_dims(signal_np, axis=0)
        return {
            'signal': torch.from_numpy(signal_np),
            'label': torch.tensor(item['label'], dtype=torch.long),
            'is_seen': torch.tensor(item['is_seen'], dtype=torch.long)
        }

class RealSeenTrainDataset(Dataset):
    def __init__(self, preprocessed_data_dir):
        train_meta_path = os.path.join(preprocessed_data_dir, 'train_meta.pkl')
        data_path = os.path.join(preprocessed_data_dir, 'processed_data.npy')
        if not os.path.exists(data_path): raise FileNotFoundError(f"未找到 'processed_data.npy'")
        self.all_real_data = np.load(data_path)
        with open(train_meta_path, 'rb') as f: self.meta = dill.load(f)
        self.all_real_data = _standardize_data(self.all_real_data)

    def __len__(self): return len(self.meta['data_list'])
    def __getitem__(self, idx):
        item = self.meta['data_list'][idx]
        signal_np = self.all_real_data[item['index']].astype(np.float32)
        if signal_np.ndim == 1: signal_np = np.expand_dims(signal_np, axis=0)
        return {'signal': torch.from_numpy(signal_np)}

def infer_model_shapes(state_dict):
    """自动推断模型维度的辅助函数"""
    # 1. 推断输入层权重
    if 'input_proj.0.weight' in state_dict:
        weight = state_dict['input_proj.0.weight']
        model_dim = weight.shape[0]
        input_dim = weight.shape[1]
    elif 'input_proj.weight' in state_dict:
        weight = state_dict['input_proj.weight']
        model_dim = weight.shape[0]
        input_dim = weight.shape[1]
    else:
        raise KeyError("无法在权重文件中找到 input_proj 的权重")
    
    # 2. 推断类别数
    if 'output_layer.weight' in state_dict:
        num_seen_classes = state_dict['output_layer.weight'].shape[0]
    else:
        raise KeyError("无法找到 output_layer.weight")
        
    return input_dim, model_dim, num_seen_classes

def find_openset_threshold(feature_extractor, final_classifier, seen_train_dir, device):
    print("Finding optimal threshold for open-set detection...")
    seen_dataset = RealSeenTrainDataset(seen_train_dir)
    seen_loader = DataLoader(seen_dataset, batch_size=128)
    all_max_softmax = []
    feature_extractor.eval(); final_classifier.eval()
    with torch.no_grad():
        for batch in tqdm(seen_loader, desc="Calibrating threshold"):
            signals = batch['signal'].to(device)
            _, features = feature_extractor(signals)
            logits = final_classifier(features)
            softmax_probs = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(softmax_probs, dim=1)
            all_max_softmax.extend(max_probs.cpu().numpy())
    threshold = np.percentile(all_max_softmax, 5) if all_max_softmax else 0.5
    print(f"Threshold set to {threshold:.4f} (95% recall on seen training data)")
    return threshold

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataset = GZSLTestDataset(args.preprocessed_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"Loading feature extractor from: {args.feature_extractor_path}")
    feature_extractor_state = torch.load(args.feature_extractor_path, map_location=device)
    
    input_dim, model_dim, num_seen_classes_in_fe = infer_model_shapes(feature_extractor_state)
    print(f"Inferred feature extractor shape: input_dim={input_dim}, model_dim={model_dim}, num_classes={num_seen_classes_in_fe}")
    
    feature_extractor = SignalClassifier(input_dim=input_dim, model_dim=model_dim, num_classes=num_seen_classes_in_fe).to(device)
    feature_extractor.load_state_dict(feature_extractor_state)
    feature_extractor.eval()
    
    print(f"Loading final MLP classifier from: {args.model_path}")
    num_total_classes = len(test_dataset.meta['label_text_dict'])
    final_classifier = MLPClassifier(input_dim=model_dim, hidden_dim=args.hidden_dim, output_dim=num_total_classes).to(device)
    final_classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    final_classifier.eval()
    
    # ==================== ★★★ 核心部分：自动搜索最佳 Gamma 并保存曲线数据 ★★★ ====================
    print("\n" + "="*50 + "\n      Searching for Best Gamma (Scanning 0.0 - 100.0)\n" + "="*50)

    # 1. 准备校准掩码需要的索引
    train_meta_path = os.path.join(args.preprocessed_data_dir, 'train_meta.pkl')
    with open(train_meta_path, 'rb') as f: train_meta = dill.load(f)
    seen_text_pool = list(train_meta['label_text_dict'].keys())
    global_text_to_idx = {v: k for k, v in test_dataset.meta['label_text_dict'].items()}
    seen_class_indices = [global_text_to_idx[text] for text in seen_text_pool]
    seen_class_indices = torch.tensor(seen_class_indices, device=device)

    # 2. 预先计算所有样本的 Logits (加速搜索，避免重复推理)
    all_logits = []
    all_labels = []
    all_is_seen = []
    all_max_softmax = [] 

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference for Search"):
            signals = batch['signal'].to(device)
            
            _, features = feature_extractor(signals)
            logits = final_classifier(features)
            
            all_logits.append(logits)
            all_labels.append(batch['label'])
            all_is_seen.append(batch['is_seen'])
            
            # 用于 OpenSet
            softmax_probs = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(softmax_probs, dim=1)
            all_max_softmax.extend(max_probs.cpu().numpy())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_is_seen = torch.cat(all_is_seen, dim=0).numpy()
    
    seen_mask = (all_is_seen == 1)
    unseen_mask = (all_is_seen == 0)

    # 3. 循环搜索最佳 Gamma 并记录详细数据
    best_h = 0.0
    best_gamma = 0.0
    
    # 准备列表存储曲线数据
    history_gamma = []
    history_seen = []
    history_unseen = []
    history_h = []
    
    # 扩大搜索范围到 100，步长 1.0
    gamma_range = np.arange(0.0, 100.0, 1.0)
    
    for g in gamma_range:
        # Calibrated Softmax: Logits_seen = Logits_seen - gamma
        calibration_mask = torch.zeros_like(all_logits)
        calibration_mask[:, seen_class_indices] = 1.0
        calibrated_logits = all_logits - calibration_mask * g
        
        preds = torch.argmax(calibrated_logits, dim=1).cpu().numpy()
        
        acc_s = accuracy_score(all_labels[seen_mask], preds[seen_mask]) if seen_mask.sum() > 0 else 0.0
        acc_u = accuracy_score(all_labels[unseen_mask], preds[unseen_mask]) if unseen_mask.sum() > 0 else 0.0
        
        if (acc_s + acc_u) > 0:
            h_mean = (2 * acc_s * acc_u) / (acc_s + acc_u)
        else:
            h_mean = 0.0
            
        # 记录数据
        history_gamma.append(g)
        history_seen.append(acc_s)
        history_unseen.append(acc_u)
        history_h.append(h_mean)

        if h_mean > best_h:
            best_h = h_mean
            best_gamma = g

    # 4. 保存曲线数据到 CSV (关键修改)
    # 自动从路径中提取数据集名称 (例如 PAMAP2, USC-HAD)
    dataset_name = os.path.basename(os.path.normpath(args.preprocessed_data_dir))
    csv_filename = f'gamma_curve_{dataset_name}.csv'
    
    df = pd.DataFrame({
        'gamma': history_gamma,
        'seen_acc': history_seen,
        'unseen_acc': history_unseen,
        'h_mean': history_h
    })
    df.to_csv(csv_filename, index=False)
    print(f"✅ Gamma curve data saved to: {csv_filename}")

    # 5. 打印最佳结果
    # 重新计算一次最佳 Gamma 对应的具体 Seen/Unseen 值以便打印
    best_idx = history_gamma.index(best_gamma)
    print(f"\n🏆 Best Result Found:")
    print(f"   Gamma: {best_gamma}")
    print(f"   Seen Acc: {history_seen[best_idx]:.4f}")
    print(f"   Unseen Acc: {history_unseen[best_idx]:.4f}")
    print(f"   H-mean: {best_h:.4f}")
    
    # ==================== Open Set Evaluation ====================
    threshold = find_openset_threshold(feature_extractor, final_classifier, args.preprocessed_data_dir, device)
    openset_true = all_is_seen
    openset_pred = (np.array(all_max_softmax) > threshold).astype(int)
    openset_f1 = f1_score(openset_true, openset_pred, pos_label=1, zero_division=0)
    print(f"\nOpen-Set F1: {openset_f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_data_dir', default='./data_preprocessed/PAMAP2')
    parser.add_argument('--model_path', default='results_GZSL_Feature/PAMAP2/final_classifier_mlp.pth')
    parser.add_argument('--feature_extractor_path', default='results_GZSL_Feature/PAMAP2/feature_extractor.pth')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.0) # 占位符，实际会搜索
    args = parser.parse_args()
    main(args)