# In data_utils.py (COMPLETE AND FIXED FOR MMWAVE DIMENSIONALITY)

import os
import numpy as np
import pandas as pd
import random
import dill
from collections import Counter
from sklearn.preprocessing import StandardScaler
import scipy.io as io

# --- 通用辅助函数 ---

def sliding_window(array, windowsize, overlaprate):
    stride = int(windowsize * (1 - overlaprate))
    if stride == 0: stride = 1
    windows = []
    array_len = len(array)
    for start_idx in range(0, array_len - windowsize + 1, stride):
        end_idx = start_idx + windowsize
        windows.append(array[start_idx:end_idx])
    return windows

def z_score_standard_single(data):
    # 增加健壮性检查，防止对空数组或维度不正确的数组进行操作
    if data.ndim < 2 or data.shape[0] == 0:
        print(f"警告: z_score_standard_single 接收到维度不正确或为空的数组 (shape: {data.shape})，跳过标准化。")
        return data
    shape = data.shape
    # -1 会自动计算需要被展平的维度的大小
    data = data.reshape(-1, shape[-1])
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data.reshape(shape)

# =============================================================================
#                       数据集专属预处理函数区域
# =============================================================================

def _preprocess_pamap_data(dataset_dir, window_size, overlap_rate):
    """
    加载并预处理原始的 PAMAP2 数据集。
    """
    use_cols = [1] + [*range(4, 16)] + [*range(21, 33)] + [*range(38, 50)]
    category_dict = dict(zip(range(12), [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]))
    all_data, all_labels = [], []
    
    filelist = [f for f in os.listdir(dataset_dir) if f.startswith('subject') and f.endswith('.dat')]
    if not filelist:
        raise FileNotFoundError(f"在 '{dataset_dir}' 中未找到数据文件。请检查 configs.py 中的 'raw_data_dir' 路径。")

    print('正在加载并预处理 PAMAP2 subject 数据...')
    for filename in sorted(filelist):
        subject_id = int(filename.split('.')[0][-3:])
        print(f'     正在处理 subject: 【{subject_id}】')
        file_path = os.path.join(dataset_dir, filename)
        content = pd.read_csv(file_path, sep=' ', header=None, usecols=use_cols)
        content = content.interpolate(method='linear', limit_direction='forward', axis=0).to_numpy()
        data = content[::3, 1:]
        labels = content[::3, 0]
        active_mask = labels != 0
        data, labels = data[active_mask], labels[active_mask]
        for label_id in range(12):
            true_label = category_dict[label_id]
            activity_data = data[labels == true_label]
            if len(activity_data) > window_size:
                cur_data_windows = sliding_window(activity_data, window_size, overlap_rate)
                all_data.extend(cur_data_windows)
                all_labels.extend([label_id] * len(cur_data_windows))

    all_data = np.array(all_data, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    
    print_summary("PAMAP2", all_data, all_labels)
    return all_data, all_labels

def _preprocess_usc_had_data(dataset_dir, window_size, overlap_rate):
    """
    加载并预处理 USC-HAD 数据集。
    """
    print('正在加载并预处理 USC-HAD subject 数据...')
    result_by_class = [[] for _ in range(12)]
    
    subject_list = [d for d in os.listdir(dataset_dir) if d.startswith('Subject') and os.path.isdir(os.path.join(dataset_dir, d))]
    if not subject_list:
        raise FileNotFoundError(f"在 '{dataset_dir}' 中未找到 Subject 文件夹。请检查 configs.py 中的 'raw_data_dir' 路径。")
    
    for subject in sorted(subject_list):
        print(f'     正在处理: 【{subject}】')
        subject_path = os.path.join(dataset_dir, subject)
        mat_list = [f for f in os.listdir(subject_path) if f.endswith('.mat')]
        for mat_file in mat_list:
            class_id = int(mat_file.split('t')[0][1:]) - 1
            file_path = os.path.join(subject_path, mat_file)
            content = io.loadmat(file_path)['sensor_readings']
            windows = sliding_window(content, window_size, overlap_rate)
            result_by_class[class_id].extend(windows)
            
    all_data_list, all_labels_list = [], []
    for class_id, data_windows in enumerate(result_by_class):
        if data_windows:
            all_data_list.extend(data_windows)
            all_labels_list.extend([class_id] * len(data_windows))

    all_data = np.array(all_data_list, dtype=np.float32)
    all_labels = np.array(all_labels_list, dtype=np.int64)

    print_summary("USC-HAD", all_data, all_labels)
    return all_data, all_labels

def _preprocess_mmwave_data(dataset_dir, window_size, overlap_rate):
    """
    加载并预处理 mmWave (MMFi) 数据集。
    (已修复：增强特征提取，保留均值、标准差和极差)
    """
    print('正在加载并预处理 mmWave (MMFi) 数据 (增强版)...')
    all_data_list, all_labels_list = [], []
    
    # 检查目录是否存在
    if not os.path.exists(dataset_dir):
         raise FileNotFoundError(f"路径不存在: {dataset_dir}。请检查 configs.py。")

    env_list = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not env_list:
         raise FileNotFoundError(f"在 '{dataset_dir}' 中未找到环境文件夹 (e.g., 'E1')。")

    for env in env_list:
        env_path = os.path.join(dataset_dir, env)
        subject_list = [d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d))]
        for subject in subject_list:
            subject_path = os.path.join(env_path, subject)
            action_list = [d for d in os.listdir(subject_path) if d.startswith('A') and os.path.isdir(os.path.join(subject_path, d))]
            print(f'     正在处理: {env}/{subject}')
            
            for action in action_list:
                action_id = int(action[1:]) - 1
                action_path = os.path.join(subject_path, action)
                
                if not os.path.exists(action_path): continue

                frame_list = sorted([f for f in os.listdir(action_path) if f.endswith('.bin')])
                current_sequence = []
                
                for frame_file in frame_list:
                    frame_path = os.path.join(action_path, frame_file)
                    
                    # 1. 读取原始点云数据
                    with open(frame_path, 'rb') as f:
                        raw_data = f.read()
                        # 这里的 shape 是 (N, 5)，N是点的数量
                        frame_data = np.frombuffer(raw_data, dtype=np.float64).copy().reshape(-1, 5)

                    # ==================== ★★★ 核心修改开始 ★★★ ====================
                    # 改动：提取更丰富的统计特征 (均值 + 标准差 + 极差)
                    # 维度从 5 变为 15 (5+5+5)
                    
                    if frame_data.shape[0] > 0:
                        # 1. 均值 (重心)
                        feat_mean = np.mean(frame_data, axis=0) # shape (5,)
                        # 2. 标准差 (反映肢体张开程度)
                        feat_std = np.std(frame_data, axis=0)   # shape (5,)
                        # 3. 极差 (反映动作幅度和轮廓大小)
                        feat_range = np.ptp(frame_data, axis=0) # shape (5,)
                        
                        # 拼接所有特征: 5 + 5 + 5 = 15维
                        frame_feature = np.concatenate([feat_mean, feat_std, feat_range], axis=0).reshape(1, -1)
                    else:
                        # 如果这一帧没有检测到点，用全0填充，注意维度是 15
                        frame_feature = np.zeros((1, 15), dtype=np.float64)
                    
                    current_sequence.append(frame_feature)
                    # ==================== ★★★ 核心修改结束 ★★★ ====================

                if not current_sequence: continue

                full_sequence_data = np.concatenate(current_sequence, axis=0)
                windows = sliding_window(full_sequence_data, window_size, overlap_rate)
                all_data_list.extend(windows)
                all_labels_list.extend([action_id] * len(windows))
    
    all_data = np.array(all_data_list, dtype=np.float32) if all_data_list else np.array([])
    all_labels = np.array(all_labels_list, dtype=np.int64)

    print_summary("mmWave (Enhanced)", all_data, all_labels)
    return all_data, all_labels

def print_summary(name, data, labels):
    """一个辅助函数，用于打印预处理摘要"""
    print('\n' + '-'*60)
    print(f'{name} 预处理完成。')
    print(f'总数据形态: {data.shape}')
    print(f'总标签形态: {labels.shape}')
    print(f'标签分布: {Counter(labels)}')
    print('-'*60 + '\n')

# =============================================================================
#                       通用的 ZSL/GZSL 划分器
# =============================================================================
class ZeroShotSplitter:
    def __init__(self, all_data, all_labels, log_dir, train_ratio=0.8, val_ratio=0.1):
        self.all_data = all_data
        self.all_labels = all_labels
        self.log_dir = log_dir
        self.label_text_dict = {}
        self.text_label_dict = {}
        if not (train_ratio + val_ratio < 1.0):
            raise ValueError("train_ratio 和 val_ratio 的总和必须小于 1.0。")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        os.makedirs(self.log_dir, exist_ok=True)

    def bind_label_text_dict(self, label_text_dict):
        self.label_text_dict = label_text_dict
        self.text_label_dict = {v: k for k, v in self.label_text_dict.items()}

    def split_data(self, seen_num):
        random.seed(123)
        np.random.seed(123)
        if not self.label_text_dict:
            raise ValueError("未初始化标签-文本字典。请先调用 bind_label_text_dict。")

        all_text_labels = sorted(list(self.text_label_dict.keys()))
        random.shuffle(all_text_labels)
        
        seen_text_pool = all_text_labels[:seen_num]
        unseen_text_pool = all_text_labels[seen_num:]
        
        seen_text_label_dict = {text: i for i, text in enumerate(seen_text_pool)}
        unseen_text_label_dict = {text: i for i, text in enumerate(unseen_text_pool)}

        seen_label_indices = [self.text_label_dict[text] for text in seen_text_pool]
        unseen_label_indices = [self.text_label_dict[text] for text in unseen_text_pool]
        
        seen_meta_class_dict = {label_idx: [] for label_idx in seen_label_indices}
        unseen_meta = []

        for i in range(len(self.all_labels)):
            original_label = self.all_labels[i]
            corres_text = self.label_text_dict[original_label]
            item = {"text": corres_text, "index": i}
            if original_label in seen_label_indices:
                item["label"] = seen_text_label_dict[corres_text]
                seen_meta_class_dict[original_label].append(item)
            elif original_label in unseen_label_indices:
                item["label"] = unseen_text_label_dict[corres_text]
                unseen_meta.append(item)

        train_meta = {"type": "train", "label_text_dict": seen_text_label_dict, "data_list": []}
        val_meta = {"type": "val", "label_text_dict": seen_text_label_dict, "data_list": []}
        test_seen_meta = {"type": "test_seen", "label_text_dict": seen_text_label_dict, "data_list": []}
        
        for class_data in seen_meta_class_dict.values():
            random.shuffle(class_data)
            n = len(class_data)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)
            train_meta["data_list"].extend(class_data[:train_end])
            val_meta["data_list"].extend(class_data[train_end:val_end])
            test_seen_meta["data_list"].extend(class_data[val_end:])

        random.shuffle(train_meta["data_list"])

        gzsl_label_text_dict = {self.text_label_dict[text]: text for text in all_text_labels}
        gzsl_test_meta = {"type": "test_gzsl", "label_text_dict": gzsl_label_text_dict, "data_list": []}
        
        for item in test_seen_meta["data_list"]:
            item_copy = item.copy()
            item_copy['label'] = self.text_label_dict[item['text']]
            item_copy['is_seen'] = 1
            gzsl_test_meta['data_list'].append(item_copy)

        for item in unseen_meta:
            item_copy = item.copy()
            item_copy['label'] = self.text_label_dict[item['text']]
            item_copy['is_seen'] = 0
            gzsl_test_meta['data_list'].append(item_copy)

        random.shuffle(gzsl_test_meta['data_list'])
        
        files_to_save = {
            "train_meta.pkl": train_meta,
            "val_meta.pkl": val_meta,
            "test_seen_meta.pkl": test_seen_meta,
            "test_unseen_meta.pkl": {"type": "test_unseen", "label_text_dict": unseen_text_label_dict, "data_list": unseen_meta},
            "test_gzsl_meta.pkl": gzsl_test_meta
        }
        for filename, meta_dict in files_to_save.items():
            meta_dict["samples"] = len(meta_dict["data_list"])
            dill.dump(meta_dict, open(os.path.join(self.log_dir, filename), "wb"))

        print("[***] 已创建新的训练/测试划分:")
        print(f"Seen 类别 ({len(seen_text_pool)}): {seen_text_pool}")
        print(f"Unseen 类别 ({len(unseen_text_pool)}): {unseen_text_pool}")
        print(f"\n元数据文件已成功保存在 '{self.log_dir}'")

# =============================================================================
#                                主执行函数
# =============================================================================
def main(config):
    save_dir = config['preprocessed_data_dir']
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir, 'processed_data.npy')
    label_path = os.path.join(save_dir, 'processed_labels.npy')

    if os.path.exists(data_path) and os.path.exists(label_path):
        print("发现已预处理的数据，从缓存加载...")
        all_data, all_labels = np.load(data_path), np.load(label_path)
    else:
        print("未找到预处理数据，开始预处理流程...")
        if config['name'].upper() == 'PAMAP2':
            all_data, all_labels = _preprocess_pamap_data(dataset_dir=config['raw_data_dir'], window_size=config['window_size'], overlap_rate=config['overlap_rate'])
        elif config['name'].upper() == 'USC-HAD':
            all_data, all_labels = _preprocess_usc_had_data(dataset_dir=config['raw_data_dir'], window_size=config['window_size'], overlap_rate=config['overlap_rate'])
        elif config['name'].upper() == 'MMWAVE':
            all_data, all_labels = _preprocess_mmwave_data(dataset_dir=config['raw_data_dir'], window_size=config['window_size'], overlap_rate=config['overlap_rate'])
        else:
            raise ValueError(f"没有为数据集 '{config['name']}' 定义预处理函数。")
        
        if all_data.shape[0] == 0:
            raise RuntimeError(f"预处理后未生成任何数据样本！\n请检查 configs.py 中的 'raw_data_dir' 路径 ('{config['raw_data_dir']}') 是否正确，并确认该路径下 '{config['name']}' 数据集的目录结构是否符合预期。")

        print("正在对整个数据集进行 Z-score 标准化...")
        all_data = z_score_standard_single(all_data)
        np.save(data_path, all_data)
        np.save(label_path, all_labels)
        print(f"预处理数据已保存至 '{save_dir}'")

    splitter = ZeroShotSplitter(all_data, all_labels, log_dir=save_dir)
    splitter.bind_label_text_dict(config['label_text_dict'])
    splitter.split_data(seen_num=config['num_seen_classes'])