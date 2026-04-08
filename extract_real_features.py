# # In extract_real_features.py (FIXED)

# import torch
# import numpy as np
# import os
# import dill
# import argparse
# from classifier_model import SignalClassifier
# from train_feature_extractor import RealSeenTrainDataset
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# def main(args):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # 1. 加载训练好的特征提取器
#     print(f"Loading feature extractor from: {args.feature_extractor_path}")
#     feature_extractor_state = torch.load(args.feature_extractor_path, map_location=device)
    
#     # ★★★ 核心修复：从 state_dict 中动态推断模型结构 ★★★
#     # 'input_proj.weight' 的形状是 [model_dim, input_dim]
#     model_dim = feature_extractor_state['input_proj.weight'].shape[0]
#     input_dim = feature_extractor_state['input_proj.weight'].shape[1] 
#     # 'output_layer.weight' 的形状是 [num_classes, model_dim]
#     num_seen_classes = feature_extractor_state['output_layer.weight'].shape[0]

#     print(f"Inferred model shape: input_dim={input_dim}, model_dim={model_dim}, num_classes={num_seen_classes}")
    
#     # 使用推断出的维度来创建模型实例，确保结构完全匹配
#     feature_extractor = SignalClassifier(
#         input_dim=input_dim,
#         model_dim=model_dim, 
#         num_classes=num_seen_classes
#     ).to(device)
    
#     # 现在加载状态字典就不会有尺寸不匹配的问题了
#     feature_extractor.load_state_dict(feature_extractor_state)
#     feature_extractor.eval()

#     # 2. 加载更新后的数据集
#     print(f"Loading real seen dataset from: {args.preprocessed_data_dir}")
#     # 注意：这个Dataset的__init__里也加载了数据文件，确保它也使用了通用的文件名
#     dataset = RealSeenTrainDataset(args.preprocessed_data_dir)
#     loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
#     # 3. 提取特征
#     print("Extracting features, labels, and texts from real seen data...")
#     all_features, all_global_labels, all_text_labels = [], [], []

#     gzsl_meta_path = os.path.join(args.preprocessed_data_dir, 'test_gzsl_meta.pkl')
#     with open(gzsl_meta_path, 'rb') as f:
#         gzsl_meta = dill.load(f)
#     global_text_to_idx = {v: k for k, v in gzsl_meta['label_text_dict'].items()}
    
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Extracting"):
#             signals = batch['signal'].to(device)
#             texts = batch['text']
#             global_labels = [global_text_to_idx[t] for t in texts]
#             _, features = feature_extractor(signals)
            
#             all_features.append(features.cpu())
#             all_global_labels.extend(global_labels)
#             all_text_labels.extend(texts)

#     all_features = torch.cat(all_features, dim=0).numpy()
    
#     # 4. 保存所有必需的信息
#     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
#     np.savez(args.output_path, 
#              features=all_features, 
#              labels=np.array(all_global_labels),
#              texts=np.array(all_text_labels))    
             
#     print(f"\nExtracted {len(all_global_labels)} real features and saved to {args.output_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Extract features, global labels, and texts from real seen data.")
#     parser.add_argument('--preprocessed_data_dir', default='./data_preprocessed/PAMAP2')
#     parser.add_argument('--feature_extractor_path', default='results_GZSL_Feature/feature_extractor.pth')
#     parser.add_argument('--output_path', default='./data_preprocessed/PAMAP2/real_seen_features.npz')
#     parser.add_argument('--batch_size', type=int, default=256)
#     args = parser.parse_args()
#     main(args)

# classifier_model.py (FIXED)
# In extract_real_features.py (FIXED FOR CONV MODEL)

import torch
import numpy as np
import os
import dill
import argparse
from classifier_model import SignalClassifier
from train_feature_extractor import RealSeenTrainDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def infer_model_shapes(state_dict):
    """自动推断模型维度的辅助函数，兼容 Linear 和 Sequential(Conv1d)"""
    # 1. 推断输入层权重
    if 'input_proj.0.weight' in state_dict:
        # 新版：Sequential 结构 (Conv1d)
        weight = state_dict['input_proj.0.weight']
        # Conv1d weight shape: [out_channels, in_channels, kernel_size]
        model_dim = weight.shape[0]
        input_dim = weight.shape[1]
        print(f"Detected Conv1d architecture (New).")
    elif 'input_proj.weight' in state_dict:
        # 旧版：Linear 结构
        weight = state_dict['input_proj.weight']
        # Linear weight shape: [out_features, in_features]
        model_dim = weight.shape[0]
        input_dim = weight.shape[1]
        print(f"Detected Linear architecture (Old).")
    else:
        raise KeyError("无法在权重文件中找到 input_proj 的权重，请检查模型结构。")

    # 2. 推断类别数
    if 'output_layer.weight' in state_dict:
        num_seen_classes = state_dict['output_layer.weight'].shape[0]
    else:
        raise KeyError("无法找到 output_layer.weight")

    return input_dim, model_dim, num_seen_classes

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载训练好的特征提取器状态
    print(f"Loading feature extractor from: {args.feature_extractor_path}")
    feature_extractor_state = torch.load(args.feature_extractor_path, map_location=device)
    
    # ★★★ 修复点：使用更健壮的推断逻辑 ★★★
    input_dim, model_dim, num_seen_classes = infer_model_shapes(feature_extractor_state)

    print(f"Inferred model shape: input_dim={input_dim}, model_dim={model_dim}, num_classes={num_seen_classes}")
    
    # 2. 创建模型实例
    feature_extractor = SignalClassifier(
        input_dim=input_dim,
        model_dim=model_dim, 
        num_classes=num_seen_classes
    ).to(device)
    
    feature_extractor.load_state_dict(feature_extractor_state)
    feature_extractor.eval()

    # 3. 加载数据集
    print(f"Loading real seen dataset from: {args.preprocessed_data_dir}")
    dataset = RealSeenTrainDataset(args.preprocessed_data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. 提取特征
    print("Extracting features, labels, and texts from real seen data...")
    all_features, all_global_labels, all_text_labels = [], [], []

    gzsl_meta_path = os.path.join(args.preprocessed_data_dir, 'test_gzsl_meta.pkl')
    with open(gzsl_meta_path, 'rb') as f:
        gzsl_meta = dill.load(f)
    global_text_to_idx = {v: k for k, v in gzsl_meta['label_text_dict'].items()}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            signals = batch['signal'].to(device)
            texts = batch['text']
            global_labels = [global_text_to_idx[t] for t in texts]
            _, features = feature_extractor(signals)
            
            all_features.append(features.cpu())
            all_global_labels.extend(global_labels)
            all_text_labels.extend(texts)

    all_features = torch.cat(all_features, dim=0).numpy()
    
    # 5. 保存
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez(args.output_path, 
             features=all_features, 
             labels=np.array(all_global_labels),
             texts=np.array(all_text_labels))    
             
    print(f"\nExtracted {len(all_global_labels)} real features and saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract features, global labels, and texts from real seen data.")
    parser.add_argument('--preprocessed_data_dir', default='./data_preprocessed/PAMAP2')
    parser.add_argument('--feature_extractor_path', default='results_GZSL_Feature/feature_extractor.pth')
    parser.add_argument('--output_path', default='./data_preprocessed/PAMAP2/real_seen_features.npz')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    main(args)