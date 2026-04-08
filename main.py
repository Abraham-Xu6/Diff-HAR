import argparse
import os
import sys
import time
from argparse import Namespace

# --- 导入每个步骤的主函数 ---
from data_utils import main as data_main
from train_feature_extractor import main as train_extractor_main
from extract_real_features import main as extract_features_main
from train_feature_diffusion import main as train_diffusion_main
from generate_virtual_features import main as generate_features_main
from train_final_classifier import main as train_classifier_main
from evaluate_gzsl import main as evaluate_main

# --- 导入配置加载器 ---
from configs import get_config

def generate_paths(config):
    """根据数据集配置动态生成所有路径"""
    preprocessed_dir = config['preprocessed_data_dir']
    results_dir = config['results_dir']
    
    os.makedirs(results_dir, exist_ok=True)
    
    return {
        "real_seen_features": os.path.join(preprocessed_dir, 'real_seen_features.npz'),
        "hybrid_features_cfg": os.path.join(preprocessed_dir, 'final_hybrid_features_cfg.npz'),
        "feature_extractor": os.path.join(results_dir, 'feature_extractor.pth'),
        "feature_diffusion_cfg": os.path.join(results_dir, 'feature_diffusion_cfg.pth'),
        "final_classifier_mlp": os.path.join(results_dir, 'final_classifier_mlp.pth')
    }

def print_header(step, description, dataset_name):
    print("\n" + "="*80)
    print(f"🚀 [步骤 {step} for {dataset_name.upper()}] {description}")
    print("="*80)

def parse_steps(step_str):
    if not step_str: return []
    steps = set()
    for part in step_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            steps.update(range(start, end + 1))
        else:
            steps.add(int(part))
    return sorted(list(steps))

# =================================================================================
#                             流水线步骤定义
# =================================================================================

def run_step_1(config, paths, args_main):
    print_header(1, "数据预处理 & ZSL划分", config['name'])
    try:
        data_main(config)
    except FileNotFoundError as e:
        print(f"\n[错误] 文件未找到: {e}")
        print(f"请确保您已将 {config['name']} 数据集放入 'configs.py' 中配置的 'raw_data_dir' 路径下！")
        sys.exit(1)
    print("\n✅ 步骤 1 执行完毕。")

def run_step_2(config, paths, args_main):
    print_header(2, "训练通用特征提取器", config['name'])
    args = Namespace(
        preprocessed_data_dir=config['preprocessed_data_dir'],
        output_model_path=paths["feature_extractor"],
        epochs=40, batch_size=64, lr=1e-3
    )
    train_extractor_main(args)
    print("\n✅ 步骤 2 执行完毕。")

def run_step_3(config, paths, args_main):
    print_header(3, "提取真实的Seen类特征", config['name'])
    args = Namespace(
        preprocessed_data_dir=config['preprocessed_data_dir'],
        feature_extractor_path=paths["feature_extractor"],
        output_path=paths["real_seen_features"],
        batch_size=256
    )
    extract_features_main(args)
    print("\n✅ 步骤 3 执行完毕。")

def run_step_4(config, paths, args_main):
    print_header(4, "训练特征空间扩散模型 (带CFG)", config['name'])
    args = Namespace(
        data_path=paths["real_seen_features"],
        output_model_path=paths["feature_diffusion_cfg"],
        epochs=args_main.diffusion_epochs,
        batch_size=128, lr=1e-4, p_uncond=0.1
    )
    print(f"配置: epochs={args.epochs}")
    train_diffusion_main(args)
    print("\n✅ 步骤 4 执行完毕。")

def run_step_5(config, paths, args_main):
    print_header(5, "生成虚拟Unseen类特征", config['name'])
    
    current_noise_scale = getattr(args_main, 'noise_scale', 0.5)
    
    args = Namespace(
        preprocessed_data_dir=config['preprocessed_data_dir'],
        real_features_path=paths["real_seen_features"],
        model_path=paths["feature_diffusion_cfg"],
        # ★★★ 修复：统一使用 paths["hybrid_features_cfg"]，避免上下游文件断层 ★★★
        output_path=paths["hybrid_features_cfg"], 
        num_per_class=500,
        guidance_scale=args_main.guidance_scale,
        noise_scale=current_noise_scale 
    )
    
    print(f"配置: guidance_scale={args.guidance_scale}, noise_scale={args.noise_scale}")
    print(f"📂 [Target] 特征将保存至: {paths['hybrid_features_cfg']}") 
    
    generate_features_main(args)
    print("\n✅ 步骤 5 执行完毕。")

def run_step_6(config, paths, args_main):
    print_header(6, "训练最终的MLP分类器", config['name'])
    args = Namespace(
        data_path=paths["hybrid_features_cfg"],
        output_model_path=paths["final_classifier_mlp"],
        epochs=100, batch_size=256, lr=1e-3,
        hidden_dim=args_main.mlp_hidden_dim
    )
    print(f"配置: hidden_dim={args.hidden_dim}")
    train_classifier_main(args)
    print("\n✅ 步骤 6 执行完毕。")

def run_step_7(config, paths, args_main):
    print_header(7, "最终评估与校准", config['name'])
    args = Namespace(
        preprocessed_data_dir=config['preprocessed_data_dir'],
        model_path=paths["final_classifier_mlp"],
        feature_extractor_path=paths["feature_extractor"],
        batch_size=128,
        hidden_dim=args_main.mlp_hidden_dim,
        gamma=args_main.gamma
    )
    print(f"配置: hidden_dim={args.hidden_dim}, gamma={args.gamma}")
    evaluate_main(args)
    print("\n✅ 步骤 7 执行完毕。")

def main():
    parser = argparse.ArgumentParser(
        description="基于生成式特征增强的 GZSL 项目总控脚本",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--config_choose', 
        type=str, 
        default='PAMAP2',
        help="选择要使用的数据集配置 (例如 'PAMAP2', 'USC-HAD', 'mmWave')。\n"
             "配置定义在 configs.py 文件中。"
    )
    parser.add_argument(
        '--steps', type=str, default='1-7',
        help="指定要运行的步骤 (例如: '1-7', '3,5-7')"
    )
    # 关键超参数
    parser.add_argument('--diffusion_epochs', type=int, default=300, help="步骤4: 扩散模型的训练轮数")
    parser.add_argument('--mlp_hidden_dim', type=int, default=128, help="步骤6/7: MLP分类器的隐藏层维度")
    parser.add_argument('--gamma', type=float, default=13.0, help="步骤7: 评估时的校准强度 (默认占位)")
    parser.add_argument('--guidance_scale', type=float, default=3.0, help="步骤5: CFG引导强度")
    parser.add_argument('--noise_scale', type=float, default=0.5, help="步骤5: 自适应噪声注入强度 (Lambda)")
    
    args = parser.parse_args()

    try:
        config = get_config(args.config_choose)
        print(f"成功加载数据集配置: {config['name'].upper()}")
    except (ValueError, NotImplementedError) as e:
        print(f"[错误] 加载配置失败: {e}")
        sys.exit(1)

    paths = generate_paths(config)

    step_functions = {
        1: run_step_1, 2: run_step_2, 3: run_step_3, 4: run_step_4,
        5: run_step_5, 6: run_step_6, 7: run_step_7,
    }

    steps_to_run = parse_steps(args.steps)
    if not steps_to_run:
        print("未指定要运行的步骤。使用 --steps 参数。")
        return

    print(f"将要执行的步骤: {steps_to_run}")
    
    start_time = time.time()
    for step in steps_to_run:
        if step in step_functions:
            step_functions[step](config, paths, args)
        else:
            print(f"警告：未知的步骤 '{step}'，将被跳过。")
    
    total_time = time.time() - start_time
    print("\n" + "#"*80)
    print(f"🎉 针对数据集 '{config['name']}' 的所有指定步骤已成功完成！")
    print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print("#"*80)

if __name__ == '__main__':
    main()