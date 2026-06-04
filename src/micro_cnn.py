# -*- coding: utf-8 -*-
# micro_cnn.py — 微观结构时空CNN (Bottleneck Embedding Extractor)
#
# 核心理念: 用小型1D-CNN从微观结构特征的时序窗口中提取稠密嵌入向量，
#          作为"学习到的微观因子"注入LightGBM。
#
# 类比: 订单簿CNN (10档×4特征 → 2D Conv → Bottleneck)
# 适配: 30bar微观特征窗口 → 1D Temporal Conv → 64维嵌入
#       (受限于仅L1盘口数据，用时间维度补偿空间维度不足)
#
# 架构:
#   Input:  (batch, 30 bars, 25 micro_features)
#   Conv1D: 32 filters, kernel=5 → (batch, 26, 32)
#   Conv1D: 64 filters, kernel=3 → (batch, 24, 64)
#   GlobalAvgPool → (batch, 64)
#   Dense(128) + Dropout(0.5)
#   Bottleneck: Dense(64) ← 输出嵌入 (注入LightGBM)
#   (训练时) Prediction Head: Dense(1) → 30min forward return
#
# 训练策略:
#   1. 仅在tick数据覆盖期内训练 (2023-04 ~ 2025-10, 微观特征=真实值)
#   2. 训练后提取全体嵌入 (tick缺失区用0填充特征 → 嵌入仍有信息)
#   3. 保存64维嵌入列 → df_factors.pkl 追加新列
#   4. 后续重训练时可选微调CNN或保持冻结

import os
import pickle
import json
import warnings
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# PyTorch CNN 模型定义
# ═══════════════════════════════════════════════════════════════

_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    pass


if _HAS_TORCH:
    class MicroCNN(nn.Module):
        """微观结构时序CNN — 提取瓶颈嵌入"""
        
        def __init__(
            self,
            n_features: int = 25,
            lookback: int = 30,
            bottleneck_dim: int = 64,
            dropout: float = 0.5,
        ):
            super().__init__()
            
            # Temporal convolution stack
            self.conv1 = nn.Conv1d(
                in_channels=n_features,
                out_channels=32,
                kernel_size=5,
                padding=2,  # same-length output
            )
            self.conv2 = nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            )
            self.conv3 = nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            )
            
            # Normalization
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(64)
            
            # Activation
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            
            # Bottleneck
            self.bottleneck = nn.Linear(64, bottleneck_dim)
            self.bn_bottleneck = nn.BatchNorm1d(bottleneck_dim)
            
            # Prediction head (used only during training)
            self.fc1 = nn.Linear(bottleneck_dim, 128)
            self.bn_fc = nn.BatchNorm1d(128)
            self.pred_head = nn.Linear(128, 1)
            
            # Store dims for reference
            self.n_features = n_features
            self.lookback = lookback
            self.bottleneck_dim = bottleneck_dim
        
        def encode(self, x: 'torch.Tensor') -> 'torch.Tensor':
            """
            前向传播 → 瓶颈嵌入 (不经过预测头)
            
            Args:
                x: (batch, n_features, lookback) 或 (batch, n_features, T)
            Returns:
                embedding: (batch, bottleneck_dim)
            """
            # Conv stack
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            
            # Global average pooling over time dimension
            x = x.mean(dim=2)  # (batch, 64)
            
            # Bottleneck
            x = self.dropout(x)
            x = self.bn_bottleneck(self.bottleneck(x))
            x = self.relu(x)
            
            return x
        
        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            """
            完整前向传播 (训练时使用)
            
            Args:
                x: (batch, n_features, lookback)
            Returns:
                pred: (batch, 1) 预测的30min前向收益
            """
            embedding = self.encode(x)
            
            # Prediction head
            out = self.relu(self.bn_fc(self.fc1(embedding)))
            out = self.dropout(out)
            out = self.pred_head(out)
            
            return out.squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# 数据准备
# ═══════════════════════════════════════════════════════════════

class MicroCNNPipeline:
    """
    CNN嵌入提取管道。
    
    使用流程:
    1. pipeline = MicroCNNPipeline(base_dir)
    2. pipeline.prepare_data()        # 构建训练数据
    3. pipeline.train(epochs=50)      # 训练CNN
    4. pipeline.extract_embeddings()  # 提取全体嵌入
    5. pipeline.inject_to_factors()   # 注入df_factors.pkl
    """
    
    # 默认微观结构特征列表 (与 tick_data_processor 输出对齐)
    DEFAULT_MICRO_FEATURES = [
        'Spread_Mean', 'Spread_Max', 'Spread_Std',
        'Depth_Imbalance_Mean', 'Depth_Imbalance_Std', 'Depth_Imbalance_Last',
        'Total_Depth_Mean', 'Total_Depth_Min',
        'HF_Return_Sum', 'HF_Return_Std',
        'Signed_Volume_Sum',
        'Trade_Count', 'Open_Count', 'Close_Count',
        'Up_Tick_Ratio', 'Effective_Spread_Mean',
        # 来自 factor_extraction 的微观因子
        'Cum_Imbalance_15', 'Cum_Imbalance_30', 'Imbalance_ZScore',
        'Signed_Vol_5', 'Signed_Vol_15',
        'VPIN_5', 'VPIN_15',
        'HF_RV_5', 'HF_RV_30', 'HF_Vol_Ratio',
        'Cum_Net_Open_15', 'Cum_Net_Open_30',
        'Close_Pressure', 'Open_Price_Push',
        'Trade_Intensity', 'Vol_Disconnect',
        'Spread_Ratio',
    ]
    
    def __init__(
        self,
        base_dir: str,
        lookback: int = 30,
        bottleneck_dim: int = 64,
    ):
        self.base_dir = base_dir
        self.lookback = lookback
        self.bottleneck_dim = bottleneck_dim
        
        # Paths
        self.factors_path = os.path.join(base_dir, "outputs/df_factors.pkl")
        self.tick_features_path = os.path.join(base_dir, "outputs/tick_minute_features.pkl")
        self.model_path = os.path.join(base_dir, "models/micro_cnn.pt")
        self.embeddings_path = os.path.join(base_dir, "outputs/cnn_embeddings.pkl")
        
        self.model: Optional['MicroCNN'] = None
        self.micro_features: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._train_stats: dict = {}
    
    def _detect_micro_features(self, df: pd.DataFrame) -> List[str]:
        """自动检测df_factors中的微观结构特征列"""
        # Use predefined list, filter to columns that exist
        available = []
        for f in self.DEFAULT_MICRO_FEATURES:
            if f in df.columns:
                available.append(f)
        
        if len(available) < 10:
            # Fallback: auto-detect columns matching microstructure patterns
            micro_patterns = [
                'Spread', 'Imbalance', 'VPIN', 'HF_', 'Signed',
                'Trade_', 'Open_', 'Close_', 'Cum_', 'Vol_Disconnect',
                'Depth', 'Tick', 'Effective',
            ]
            available = [
                c for c in df.columns
                if any(p in c for p in micro_patterns)
                and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
            ]
        
        return available[:30]  # Cap at 30 features
    
    def prepare_data(
        self,
        prediction_horizon: int = 30,
        tick_only: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        准备CNN训练/推理数据。
        
        Args:
            prediction_horizon: 预测时域 (bar数, 默认30min)
            tick_only: True=仅用tick覆盖期数据训练, False=用全部数据
        
        Returns:
            X_windows: (n_samples, n_features, lookback) 或 None
            y_targets: (n_samples,) 或 None
            aligned_dates: (n_samples,) 时间戳数组
        """
        if not os.path.exists(self.factors_path):
            print(f"[MicroCNN] 因子文件不存在: {self.factors_path}")
            return None, None, None
        
        df = pd.read_pickle(self.factors_path)
        
        # Detect microstructure features
        self.micro_features = self._detect_micro_features(df)
        if len(self.micro_features) < 5:
            print(f"[MicroCNN] 微观特征不足 ({len(self.micro_features)}), 需要 >= 5")
            return None, None, None
        
        print(f"[MicroCNN] 检测到 {len(self.micro_features)} 个微观特征")
        
        # Fill NaN with 0
        df_micro = df[self.micro_features].fillna(0).values.astype(np.float32)
        
        # Filter to tick-data period for training
        if tick_only:
            # Tick data available until ~2025-10-31
            tick_cutoff = pd.Timestamp('2025-10-31')
            if 'date' in df.columns:
                tick_mask = pd.to_datetime(df['date']) <= tick_cutoff
                df_micro = df_micro[tick_mask]
                df_filtered = df[tick_mask].copy()
            else:
                df_filtered = df.copy()
            print(f"[MicroCNN] 训练数据 (tick覆盖期): {len(df_micro)} 行")
        else:
            df_filtered = df.copy()
        
        n_samples = len(df_micro)
        if n_samples < self.lookback + prediction_horizon + 100:
            print(f"[MicroCNN] 数据不足: {n_samples} < {self.lookback + prediction_horizon + 100}")
            return None, None, None
        
        # Build sliding windows
        n_windows = n_samples - self.lookback - prediction_horizon
        X_windows = np.zeros((n_windows, len(self.micro_features), self.lookback), dtype=np.float32)
        y_targets = np.zeros(n_windows, dtype=np.float32)
        aligned_dates = np.zeros(n_windows, dtype=object)
        
        # Construct Target_Ret if not present
        if 'Target_Ret' not in df_filtered.columns:
            df_filtered['next_close'] = df_filtered['close'].shift(-1)
            df_filtered['future_close'] = df_filtered['close'].shift(-(prediction_horizon + 1))
            df_filtered['Target_Ret'] = (
                df_filtered['future_close'] / df_filtered['next_close'] - 1
            ) * 100
        
        targets = df_filtered['Target_Ret'].fillna(0).values.astype(np.float32)
        dates = df_filtered['date'].values if 'date' in df_filtered.columns else np.arange(n_samples)
        
        for i in range(n_windows):
            # Input: rows [i, i+lookback) → transpose to (features, lookback)
            window = df_micro[i : i + self.lookback].T  # (features, lookback)
            X_windows[i] = window
            
            # Target: 30min forward return at i + lookback + horizon
            y_targets[i] = targets[i + self.lookback + prediction_horizon - 1]
            aligned_dates[i] = dates[i + self.lookback + prediction_horizon - 1]
        
        # Normalize each feature channel to zero-mean unit-variance
        for f_idx in range(X_windows.shape[1]):
            channel = X_windows[:, f_idx, :]
            mean = channel.mean()
            std = channel.std()
            if std > 1e-8:
                X_windows[:, f_idx, :] = (channel - mean) / std
        
        # Remove rows with NaN targets
        valid = ~np.isnan(y_targets)
        X_windows = X_windows[valid]
        y_targets = y_targets[valid]
        aligned_dates = aligned_dates[valid]
        
        # Clip extreme targets (beyond 5σ)
        target_std = np.std(y_targets)
        clip_bound = 5 * target_std
        y_targets = np.clip(y_targets, -clip_bound, clip_bound)
        
        print(f"[MicroCNN] 准备数据: X={X_windows.shape}, y={y_targets.shape}")
        print(f"[MicroCNN] Target mean={y_targets.mean():.6f}, std={y_targets.std():.6f}")
        
        return X_windows, y_targets, aligned_dates
    
    # ═══ 训练 ═══════════════════════════════════════════════
    
    def train(
        self,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = 'auto',
    ) -> bool:
        """
        训练微观CNN。
        
        Args:
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            weight_decay: L2正则化强度
            device: 'auto'/'cpu'/'cuda'
        """
        if not _HAS_TORCH:
            print("[MicroCNN] ❌ PyTorch 未安装, 无法训练CNN")
            print("[MicroCNN]     安装: pip install torch --index-url https://download.pytorch.org/whl/cpu")
            return False
        
        X, y, dates = self.prepare_data(tick_only=True)
        if X is None:
            return False
        
        # Device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"[MicroCNN] 训练设备: {device}")
        
        # Train/val split (80/20 by time)
        n = len(X)
        split = int(n * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]
        
        # Convert to torch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
        
        # Model
        self.model = MicroCNN(
            n_features=len(self.micro_features),
            lookback=self.lookback,
            bottleneck_dim=self.bottleneck_dim,
        ).to(device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[MicroCNN] 模型参数量: {n_params:,}")
        
        # Optimizer + loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.L1Loss()  # MAE — 对异常值鲁棒
        
        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item() * len(batch_X)
            
            epoch_loss /= len(X_train)
            train_losses.append(epoch_loss)
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            val_preds_all = []
            val_ys_all = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    preds = self.model(batch_X)
                    val_loss += criterion(preds, batch_y).item() * len(batch_X)
                    val_preds_all.append(preds.cpu().numpy())
                    val_ys_all.append(batch_y.cpu().numpy())
            
            val_loss /= len(X_val)
            val_losses.append(val_loss)
            
            val_preds_np = np.concatenate(val_preds_all)
            val_ys_np = np.concatenate(val_ys_all)
            if len(val_ys_np) > 1:
                val_ic = np.corrcoef(val_preds_np, val_ys_np)[0, 1]
            else:
                val_ic = 0
            
            scheduler.step()
            
            # Print every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | "
                      f"Train MAE={epoch_loss:.6f} | "
                      f"Val MAE={val_loss:.6f} | "
                      f"Val IC={val_ic:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                # Save best model
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'n_features': len(self.micro_features),
                    'lookback': self.lookback,
                    'bottleneck_dim': self.bottleneck_dim,
                    'micro_features': self.micro_features,
                    'train_stats': {
                        'best_val_loss': best_val_loss,
                        'best_epoch': best_epoch,
                        'val_ic': val_ic,
                        'n_params': n_params,
                    },
                }, self.model_path)
        
        # Load best model
        checkpoint = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._train_stats = checkpoint.get('train_stats', {})
        
        print(f"\n[MicroCNN] 训练完成: best_val_MAE={best_val_loss:.6f} (epoch {best_epoch})")
        print(f"[MicroCNN] 验证IC: {self._train_stats.get('val_ic', 0):.4f}")
        print(f"[MicroCNN] 模型已保存: {self.model_path}")
        
        return True
    
    # ═══ 嵌入提取 ═══════════════════════════════════════════
    
    def extract_embeddings(self, device: str = 'auto') -> Optional[np.ndarray]:
        """
        为全体因子数据提取CNN瓶颈嵌入。
        
        Returns:
            embeddings: (n_samples, bottleneck_dim) 或 None
        """
        if not _HAS_TORCH:
            print("[MicroCNN] ❌ PyTorch 未安装")
            return None
        
        # Load model if not in memory
        if self.model is None:
            if not os.path.exists(self.model_path):
                print(f"[MicroCNN] 模型文件不存在: {self.model_path}")
                print("[MicroCNN] 请先运行 train()")
                return None
            
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            checkpoint = torch.load(self.model_path, map_location=device)
            self.micro_features = checkpoint.get('micro_features', self.micro_features)
            
            self.model = MicroCNN(
                n_features=checkpoint['n_features'],
                lookback=checkpoint['lookback'],
                bottleneck_dim=checkpoint['bottleneck_dim'],
            ).to(device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self._train_stats = checkpoint.get('train_stats', {})
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load ALL factor data (not filtered to tick period)
        if not os.path.exists(self.factors_path):
            print(f"[MicroCNN] 因子文件不存在")
            return None
        
        df = pd.read_pickle(self.factors_path)
        
        # Ensure micro features match
        available_features = [f for f in self.micro_features if f in df.columns]
        if len(available_features) < 5:
            print(f"[MicroCNN] 微观特征不足: {len(available_features)}")
            return None
        
        df_micro = df[available_features].fillna(0).values.astype(np.float32)
        n_samples = len(df_micro)
        
        print(f"[MicroCNN] 提取嵌入: {n_samples} 行 × {len(available_features)} 特征 → {self.bottleneck_dim} 维")
        
        # Build windows for ALL samples
        n_windows = n_samples - self.lookback
        embeddings = np.zeros((n_samples, self.bottleneck_dim), dtype=np.float32)
        
        # First <lookback> rows: fill with 0 or nearest
        # For i >= lookback, compute embedding from window
        self.model.eval()
        
        batch_size = 1024
        with torch.no_grad():
            for start in range(self.lookback, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = list(range(start, end))
                
                # Build batch windows
                batch_X = np.zeros((len(batch_indices), len(available_features), self.lookback), dtype=np.float32)
                for j, idx in enumerate(batch_indices):
                    window = df_micro[idx - self.lookback : idx].T
                    batch_X[j] = window
                
                # Normalize (same as training)
                for f_idx in range(batch_X.shape[1]):
                    channel = batch_X[:, f_idx, :]
                    mean_c = channel.mean()
                    std_c = channel.std()
                    if std_c > 1e-8:
                        batch_X[:, f_idx, :] = (channel - mean_c) / std_c
                
                batch_t = torch.FloatTensor(batch_X).to(device)
                emb = self.model.encode(batch_t).cpu().numpy()
                
                for j, idx in enumerate(batch_indices):
                    embeddings[idx] = emb[j]
        
        # Fill first <lookback> rows with first valid embedding
        if self.lookback < n_samples:
            embeddings[:self.lookback] = embeddings[self.lookback]
        
        self._embeddings = embeddings
        
        # Save
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'bottleneck_dim': self.bottleneck_dim,
                'micro_features': self.micro_features,
                'extracted_at': datetime.now().isoformat(),
            }, f)
        
        print(f"[MicroCNN] 嵌入已保存: {self.embeddings_path} ({embeddings.shape})")
        
        return embeddings
    
    # ═══ 因子注入 ═══════════════════════════════════════════
    
    def inject_to_factors(self, force: bool = False) -> bool:
        """
        将CNN嵌入作为新列注入 df_factors.pkl。
        
        Args:
            force: True=覆盖已有CNN列, False=跳过(默认)
        
        Returns:
            True if injection succeeded
        """
        if self._embeddings is None:
            if not os.path.exists(self.embeddings_path):
                print("[MicroCNN] 嵌入文件不存在, 请先运行 extract_embeddings()")
                return False
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
            self._embeddings = data['embeddings']
            self.bottleneck_dim = data.get('bottleneck_dim', 64)
        
        if not os.path.exists(self.factors_path):
            print(f"[MicroCNN] 因子文件不存在: {self.factors_path}")
            return False
        
        df = pd.read_pickle(self.factors_path)
        
        n_emb = len(self._embeddings)
        n_df = len(df)
        
        if n_emb != n_df:
            print(f"[MicroCNN] ⚠ 嵌入行数({n_emb}) != 因子行数({n_df})")
            if n_emb < n_df:
                # Pad with zeros at the beginning
                padded = np.zeros((n_df, self.bottleneck_dim), dtype=np.float32)
                padded[-n_emb:] = self._embeddings
                self._embeddings = padded
                print(f"[MicroCNN] 嵌入已填充至 {n_df} 行")
            else:
                self._embeddings = self._embeddings[:n_df]
                print(f"[MicroCNN] 嵌入已截断至 {n_df} 行")
        
        # Check if CNN columns already exist
        cnn_cols = [f'CNN_Emb_{i:02d}' for i in range(self.bottleneck_dim)]
        existing = [c for c in cnn_cols if c in df.columns]
        
        if existing and not force:
            print(f"[MicroCNN] CNN嵌入列已存在 ({len(existing)}/{self.bottleneck_dim}), 跳过")
            print("[MicroCNN] 使用 force=True 强制覆盖")
            return True  # Already injected, not an error
        
        # Remove old CNN columns if force
        if force:
            old_cnn = [c for c in df.columns if c.startswith('CNN_Emb_')]
            df = df.drop(columns=old_cnn, errors='ignore')
        
        # Inject
        for i in range(self.bottleneck_dim):
            col_name = f'CNN_Emb_{i:02d}'
            df[col_name] = self._embeddings[:, i]
        
        # Save
        backup_path = self.factors_path + '.pre_cnn.backup'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(self.factors_path, backup_path)
            print(f"[MicroCNN] 原始因子文件已备份: {backup_path}")
        
        df.to_pickle(self.factors_path)
        
        print(f"[MicroCNN] ✓ {self.bottleneck_dim} 维CNN嵌入已注入 df_factors.pkl")
        print(f"[MicroCNN] 新增列: CNN_Emb_00 ~ CNN_Emb_{self.bottleneck_dim-1:02d}")
        print(f"[MicroCNN] 因子总数: {len(df.columns)} (原{len(df.columns) - self.bottleneck_dim} + {self.bottleneck_dim} CNN)")
        
        # Write flag file
        flag_path = os.path.join(self.base_dir, "outputs/.cnn_injected")
        with open(flag_path, 'w') as f:
            f.write(datetime.now().isoformat())
        
        return True
    
    # ═══ 状态查询 ═══════════════════════════════════════════
    
    def get_status(self) -> dict:
        """返回CNN模块当前状态 (不加载大文件)"""
        model_exists = os.path.exists(self.model_path)
        embeddings_exist = os.path.exists(self.embeddings_path)
        
        # Check injection via flag file or backup existence
        injected = False
        flag_path = os.path.join(self.base_dir, "outputs/.cnn_injected")
        backup_path = self.factors_path + '.pre_cnn.backup'
        if os.path.exists(flag_path) or os.path.exists(backup_path):
            injected = True
        
        return {
            'model_exists': model_exists,
            'embeddings_exist': embeddings_exist,
            'injected': injected,
            'n_features': len(self.micro_features) if self.micro_features else 0,
            'bottleneck_dim': self.bottleneck_dim,
            'lookback': self.lookback,
            'train_stats': self._train_stats,
            'has_torch': _HAS_TORCH,
        }
    
    def get_status_text(self) -> str:
        """生成可读状态文本"""
        s = self.get_status()
        lines = [
            "=" * 50,
            "  MicroCNN 微观结构嵌入提取器",
            "=" * 50,
            f"  PyTorch: {'✓ 可用' if s['has_torch'] else '✗ 未安装'}",
            f"  模型: {'✓ 已训练' if s['model_exists'] else '✗ 未训练'}",
            f"  嵌入: {'✓ 已提取' if s['embeddings_exist'] else '✗ 未提取'}",
            f"  注入: {'✓ 已注入' if s['injected'] else '✗ 未注入'}",
            f"  微观特征数: {s['n_features']}",
            f"  瓶颈维度: {s['bottleneck_dim']}",
            f"  回看窗口: {s['lookback']} bars",
        ]
        
        stats = s['train_stats']
        if stats:
            lines.append(f"\n  训练统计:")
            lines.append(f"    最佳验证MAE: {stats.get('best_val_loss', 'N/A')}")
            lines.append(f"    最佳轮次: {stats.get('best_epoch', 'N/A')}")
            lines.append(f"    验证IC: {stats.get('val_ic', 'N/A')}")
            lines.append(f"    参数量: {stats.get('n_params', 'N/A'):,}")
        
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════

def run_status(base_dir: str) -> int:
    pipeline = MicroCNNPipeline(base_dir)
    print(pipeline.get_status_text())
    return 0


def run_train(base_dir: str, epochs: int = 50) -> int:
    pipeline = MicroCNNPipeline(base_dir)
    ok = pipeline.train(epochs=epochs)
    return 0 if ok else 1


def run_extract(base_dir: str) -> int:
    pipeline = MicroCNNPipeline(base_dir)
    emb = pipeline.extract_embeddings()
    return 0 if emb is not None else 1


def run_inject(base_dir: str, force: bool = False) -> int:
    pipeline = MicroCNNPipeline(base_dir)
    ok = pipeline.inject_to_factors(force=force)
    return 0 if ok else 1


def run_full_pipeline(base_dir: str, epochs: int = 50) -> int:
    """一键完整流程: 训练 → 提取 → 注入"""
    pipeline = MicroCNNPipeline(base_dir)
    
    print("\n" + "=" * 55)
    print("  MicroCNN 完整管道")
    print("=" * 55)
    
    # Step 1: Train
    print("\n[1/3] 训练CNN...")
    ok = pipeline.train(epochs=epochs)
    if not ok:
        return 1
    
    # Step 2: Extract embeddings
    print("\n[2/3] 提取嵌入...")
    emb = pipeline.extract_embeddings()
    if emb is None:
        return 1
    
    # Step 3: Inject to factors
    print("\n[3/3] 注入因子...")
    ok = pipeline.inject_to_factors(force=True)
    if not ok:
        return 1
    
    print("\n" + "=" * 55)
    print("  MicroCNN 管道完成!")
    print("  CNN嵌入已注入 df_factors.pkl")
    print("  下一步: 重新训练LightGBM以使用CNN特征")
    print("  python main.py --mode train")
    print("=" * 55)
    
    return 0


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python micro_cnn.py <command> [base_dir] [options]")
        print("  status    — 查看CNN模块状态")
        print("  train     — 训练CNN模型")
        print("  extract   — 提取全体嵌入")
        print("  inject    — 注入df_factors.pkl")
        print("  pipeline  — 一键完整流程 (训练+提取+注入)")
        print("  pipeline --epochs 100  — 自定义训练轮数")
        print()
        print("推荐流程:")
        print("  1. python micro_cnn.py pipeline")
        print("  2. python main.py --mode train     # 用CNN特征重训练LightGBM")
        print("  3. python self_evolution.py weekly  # 适配器也会用到CNN特征")
        sys.exit(1)
    
    command = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else r"D:\桌面\F_Agent"
    
    # Parse extra options
    epochs = 50
    force = False
    for i, arg in enumerate(sys.argv):
        if arg == '--epochs' and i + 1 < len(sys.argv):
            epochs = int(sys.argv[i + 1])
        if arg == '--force':
            force = True
    
    commands = {
        'status': lambda: run_status(base),
        'train': lambda: run_train(base, epochs),
        'extract': lambda: run_extract(base),
        'inject': lambda: run_inject(base, force),
        'pipeline': lambda: run_full_pipeline(base, epochs),
    }
    
    if command not in commands:
        print(f"未知命令: {command}")
        sys.exit(1)
    
    sys.exit(commands[command]())
