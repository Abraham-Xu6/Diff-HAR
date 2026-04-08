# In train_feature_diffusion.py (FINAL CORRECTED VERSION WITH CFG)

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --- 模型定义 (保持不变) ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.main = nn.Sequential(*layers)
    def forward(self, x): return self.main(x)

class FeatureDiffusion(nn.Module):
    def __init__(self, feature_dim, text_dim, time_emb_dim=128, timesteps=1000):
        super().__init__()
        self.feature_dim = feature_dim
        self.model = MLP(feature_dim + time_emb_dim + text_dim, 512, feature_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        betas = torch.linspace(0.0001, 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def sinusoidal_embedding(self, t, dim):
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=1)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self._extract(self.alphas_cumprod.sqrt(), t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract((1. - self.alphas_cumprod).sqrt(), t, x0.shape)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, xt, t, text_embeddings):
        time_emb = self.sinusoidal_embedding(t, 128)
        time_emb = self.time_mlp(time_emb)
        model_input = torch.cat([xt, time_emb, text_embeddings], dim=1)
        return self.model(model_input)

# --- 数据集定义 (保持不变) ---
class FeatureDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.features = torch.from_numpy(data['features'])
        self.text_labels = data['texts']

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'feature': self.features[idx],
            'text': self.text_labels[idx]
        }

# --- 训练逻辑 (最终修正版) ---
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = FeatureDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    text_dim = text_encoder.get_sentence_embedding_dimension()
    feature_dim = dataset.features.shape[1]

    model = FeatureDiffusion(feature_dim, text_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    print("Training Feature Diffusion Model with Classifier-Free Guidance...")
    for epoch in range(args.epochs):
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            real_features = batch['feature'].to(device)
            texts = list(batch['text'])
            batch_size = real_features.shape[0]
            
            # 文本编码仍在no_grad上下文中进行，以节省资源
            with torch.no_grad():
                text_embeddings = text_encoder.encode(texts, convert_to_tensor=True, device=device)

            # 克隆张量，使其变为可写的“普通”张量
            text_embeddings = text_embeddings.clone()

            # 现在可以安全地在克隆出的张量上进行就地修改
            uncond_mask = torch.rand(batch_size, device=device) < args.p_uncond
            text_embeddings[uncond_mask] = 0

            t = torch.randint(0, model.betas.shape[0], (batch_size,), device=device)
            noise = torch.randn_like(real_features)
            xt = model.q_sample(real_features, t, noise)
            
            pred_noise = model(xt, t, text_embeddings)
            loss = criterion(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_model_path)
    print(f"\nFeature Diffusion model saved to {args.output_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data_preprocessed/PAMAP2/real_seen_features.npz')
    parser.add_argument('--output_model_path', default='results_GZSL_Feature/feature_diffusion_cfg.pth')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--p_uncond', type=float, default=0.1, help="Probability of dropping text condition for CFG.")
    args = parser.parse_args()
    main(args)