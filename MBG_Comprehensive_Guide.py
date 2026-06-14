# ============================================================
# SETUP: Install Dependencies for MBG
# ============================================================
# mamba-ssm membutuhkan PyTorch + CUDA 12.4 (cu124).
# Google Colab default PyTorch terlalu baru, jadi kita install
# versi yang compatible secara otomatis.
#
# Runtime version: APAPUN (Latest, 2025.10, dll) - semua OK.
# Script ini otomatis handle compatibility.
# ============================================================

# !nvidia-smi

import sys, os

# Step 1: Install PyTorch 2.5.1 + CUDA 12.4 (proven compatible with mamba-ssm)
print('\n[1/3] Installing PyTorch 2.5.1+cu124...')
# !pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --quiet

# Step 2: Install mamba-ssm + dependencies
# CATATAN: mamba-ssm compile CUDA kernel dari source (~5-15 menit).
# Jika terlihat diam/tidak ada output, itu NORMAL (sedang compile).
print('[2/3] Installing mamba-ssm + causal-conv1d + einops...')
print('      Kompilasi CUDA kernel ~5-15 menit. Harap tunggu.\n')
# !pip install einops scipy scikit-learn --quiet
print('      einops + scipy + sklearn: OK')
print('      Compiling causal-conv1d...')
# !pip install causal-conv1d>=1.4.0 --no-build-isolation 2>&1 | grep -E '(Successfully|ERROR|error:)'
print('      Compiling mamba-ssm (ini yang lama)...')
# !pip install mamba-ssm --no-build-isolation 2>&1 | grep -E '(Successfully|ERROR|error:)'

# Step 3: Verifikasi
print('[3/3] Verifikasi...\n')
import torch
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA:          {torch.version.cuda}')
print(f'  GPU:           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')
print(f'  VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')

import mamba_ssm
print(f'  mamba-ssm:     {mamba_ssm.__version__} [OK]')

import einops
print(f'  einops:        {einops.__version__} [OK]')

# Clone repo
if not os.path.exists('/home/adb/awangga/mbg'):
    pass
    # !git clone https://github.com/neuraldecoding/mbg.git
else:
    print('\n  mbg repo: already cloned')

if '/home/adb/awangga/mbg' not in sys.path:
    sys.path.insert(0, '/home/adb/awangga/mbg')

print('\n' + '=' * 50)
print('  SETUP COMPLETE - Real Mamba SSM ready')
print('=' * 50)

# Verifikasi instalasi
import torch
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"NumPy version: {np.__version__}")

# Verify Mamba is REAL (not fallback)
from modules.mixer_seq_simple import MAMBA_AVAILABLE
print(f"\nMAMBA_AVAILABLE = {MAMBA_AVAILABLE}")
assert MAMBA_AVAILABLE, 'FATAL: mamba-ssm not loaded! Restart runtime and re-run setup.'
print('Model menggunakan Mamba2 SSM yang asli.')

import torch

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Enable TF32 for faster computation on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import torch
import torch.nn as nn
import numpy as np

# Import model MBG
from models.mbg import MBG
from models.gem_block import GEMBlock
from models.graph_attention import GraphAttentionLayer
from models.graph_construction import AdaptiveAdjacency, get_prior_adjacency
from utils.electrode_positions import get_electrode_positions, compute_distance_matrix
from utils.graph_utils import bfs_ordering, dfs_ordering, gaussian_adjacency

print("Semua modul berhasil di-import!")

# Inisialisasi model MBG
model = MBG(
    in_dim=200,          # Dimensi input per patch (jumlah sample points)
    out_dim=200,         # Dimensi output per patch
    d_model=200,         # Hidden dimension
    dim_feedforward=800, # FFN intermediate dimension
    seq_len=30,          # Maksimum jumlah time segments
    n_layer=12,          # Jumlah GEM blocks
    nhead=8,             # Jumlah attention heads
    num_channels=22      # Jumlah channel EEG
).to(device)

print("Model MBG berhasil diinisialisasi!")
print(f"\nArsitektur Model:")
print(f"  - Input dim: 200")
print(f"  - Hidden dim: 200")
print(f"  - Layers: 12 GEM Blocks")
print(f"  - Attention heads: 8")
print(f"  - Channels: 22")
print(f"  - Device: {device}")

# Tampilkan struktur model secara detail
print("=" * 70)
print("STRUKTUR MODEL MBG")
print("=" * 70)
print(model)

# Dapatkan posisi elektroda untuk 22-channel (BCI Competition IV 2a)
positions_22 = get_electrode_positions(22)

print("Posisi Elektroda 22-Channel (BCI Competition IV 2a):")
print("=" * 50)
for name, (x, y) in positions_22.items():
    print(f"  {name:6s}: ({x:6.2f}, {y:6.2f})")

# Hitung distance matrix
dist_matrix = compute_distance_matrix(positions_22)

print("Distance Matrix (22x22):")
print(f"Shape: {dist_matrix.shape}")
print(f"Min distance (non-zero): {dist_matrix[dist_matrix > 0].min():.4f}")
print(f"Max distance: {dist_matrix.max():.4f}")
print(f"Mean distance: {dist_matrix[dist_matrix > 0].mean():.4f}")

# Tampilkan beberapa entri
electrode_names = list(positions_22.keys())
print(f"\nContoh jarak antar elektroda:")
print(f"  Fz - FCz:  {dist_matrix[0, 3]:.4f} (dekat, same midline)")
print(f"  Fz - POz:  {dist_matrix[0, 21]:.4f} (jauh, anterior-posterior)")
print(f"  C3 - C4:   {dist_matrix[7, 11]:.4f} (jauh, bilateral)")
print(f"  C3 - C1:   {dist_matrix[7, 8]:.4f} (dekat, neighbors)")

# Hitung adjacency matrix menggunakan Gaussian kernel
adj_matrix = gaussian_adjacency(dist_matrix, sigma=1.0)

print("Adjacency Matrix (Gaussian Kernel, sigma=1.0):")
print(f"Shape: {adj_matrix.shape}")
print(f"Min value (non-zero): {adj_matrix[adj_matrix > 0].min():.4f}")
print(f"Max value: {adj_matrix.max():.4f}")
print(f"Mean value: {adj_matrix[adj_matrix > 0].mean():.4f}")

# Contoh: adjacency weights untuk Cz (index 9, pusat kepala)
cz_idx = list(positions_22.keys()).index('Cz')
print(f"\nAdjacency weights dari Cz ke elektroda lain:")
for i, name in enumerate(electrode_names):
    if i != cz_idx:
        print(f"  Cz -> {name:6s}: {adj_matrix[cz_idx, i]:.4f}")

# Visualisasi sederhana adjacency matrix (text-based)
print("Adjacency Matrix Heatmap (discretized):")
print("  Tinggi = channel dekat secara fisik")
print()

# Discretize ke simbol
symbols = [' ', '.', 'o', 'O', '#']
thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]

header = "      " + "".join([f"{n[:3]:>4s}" for n in electrode_names])
print(header)
print("      " + "-" * (4 * len(electrode_names)))

for i, name_i in enumerate(electrode_names):
    row = f"{name_i:5s}|"
    for j in range(len(electrode_names)):
        val = adj_matrix[i, j]
        sym = ' '
        for k in range(len(thresholds) - 1, -1, -1):
            if val >= thresholds[k]:
                sym = symbols[k]
                break
        row += f"   {sym}"
    print(row)

# Demonstrasi BFS ordering dari Cz (pusat kepala)
bfs_order = bfs_ordering(adj_matrix, seed_idx=cz_idx)

print("BFS Ordering mulai dari Cz (pusat kepala):")
print("=" * 50)
print("Urutan kunjungan (dari nearest neighbors ke furthest):")
for step, idx in enumerate(bfs_order):
    name = electrode_names[idx]
    dist = dist_matrix[cz_idx, idx] if idx != cz_idx else 0.0
    print(f"  Step {step:2d}: {name:6s} (jarak dari Cz: {dist:.3f})")

print(f"\nUrutan BFS: {[electrode_names[i] for i in bfs_order]}")

# Demonstrasi DFS ordering dari Cz
dfs_order = dfs_ordering(adj_matrix, seed_idx=cz_idx)

print("DFS Ordering mulai dari Cz (pusat kepala):")
print("=" * 50)
print("Urutan kunjungan (depth-first, mengikuti strongest connections):")
for step, idx in enumerate(dfs_order):
    name = electrode_names[idx]
    dist = dist_matrix[cz_idx, idx] if idx != cz_idx else 0.0
    print(f"  Step {step:2d}: {name:6s} (jarak dari Cz: {dist:.3f})")

print(f"\nUrutan DFS: {[electrode_names[i] for i in dfs_order]}")

# Perbandingan BFS vs DFS
print("Perbandingan BFS vs DFS Ordering:")
print("=" * 60)
print(f"{'Step':<6}{'BFS':<12}{'DFS':<12}{'Keterangan'}")
print("-" * 60)
for step in range(len(bfs_order)):
    bfs_name = electrode_names[bfs_order[step]]
    dfs_name = electrode_names[dfs_order[step]]
    note = "SAMA" if bfs_order[step] == dfs_order[step] else "BEDA"
    print(f"{step:<6}{bfs_name:<12}{dfs_name:<12}{note}")

print("\nInterpretasi:")
print("  - BFS: Menjelajah SELURUH neighbors terlebih dahulu (breadth)")
print("  - DFS: Menelusuri SATU path sampai habis sebelum backtrack (depth)")
print("  - Kedua strategi memberikan scan order yang topology-aware")

# Buat model versi kecil untuk demo (hemat memory)
model_small = MBG(
    in_dim=200,
    out_dim=200,
    d_model=200,
    dim_feedforward=800,
    seq_len=30,
    n_layer=4,       # 4 layers saja untuk demo
    nhead=8,
    num_channels=22
).to(device)
model_small.eval()

print("Model MBG (versi kecil, 4 layers) untuk demonstrasi")
print("=" * 50)

# Simulasi data EEG
# Format: (batch_size, num_channels, time_segments, points_per_patch)
# Contoh: 8 samples, 22 channels, 4 time segments (0.8 detik di 200Hz per patch = 4 patches dari 3.2s)
batch_size = 8
num_channels = 22
time_segments = 4
points_per_patch = 200  # 1 detik di 200 Hz sampling rate

# Generate mock EEG data (random normal, simulating preprocessed EEG)
x = torch.randn(batch_size, num_channels, time_segments, points_per_patch).to(device)

print(f"Input shape: {x.shape}")
print(f"  - Batch size: {batch_size}")
print(f"  - Channels: {num_channels} (sistem 10-20)")
print(f"  - Time segments: {time_segments}")
print(f"  - Points per patch: {points_per_patch} (1 detik @ 200Hz)")
print(f"  - Total durasi: {time_segments * points_per_patch / 200:.1f} detik")
print(f"  - Device: {x.device}")

# Forward pass
with torch.no_grad():
    output = model_small(x)

print(f"\nForward Pass Berhasil!")
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"\n  Input == Output dimensi: {x.shape == output.shape}")
print(f"\nInterpretasi:")
print(f"  Model melakukan encoding-decoding per patch.")
print(f"  Untuk pre-training: output = rekonstruksi patch yang di-mask")
print(f"  Untuk fine-tuning: output digunakan sebagai feature representation")

def count_parameters(model):
    """Hitung jumlah parameter model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def format_params(n):
    """Format jumlah parameter."""
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)

# Model kecil (4 layers)
total_small, trainable_small = count_parameters(model_small)
print(f"Model MBG (4 layers):")
print(f"  Total parameters:     {format_params(total_small)} ({total_small:,})")
print(f"  Trainable parameters: {format_params(trainable_small)} ({trainable_small:,})")

# Model full (12 layers)
total_full, trainable_full = count_parameters(model)
print(f"\nModel MBG (12 layers - full):")
print(f"  Total parameters:     {format_params(total_full)} ({total_full:,})")
print(f"  Trainable parameters: {format_params(trainable_full)} ({trainable_full:,})")

# Breakdown per komponen
print("\nParameter Breakdown per Komponen (model 12 layers):")
print("=" * 60)

# Patch Embedding
pe_params = sum(p.numel() for p in model.patch_embedding.parameters())
print(f"  Patch Embedding:     {format_params(pe_params):>10s} ({pe_params:,})")

# Encoder (GEM Blocks)
enc_params = sum(p.numel() for p in model.encoder.parameters())
print(f"  Encoder (12 GEM):    {format_params(enc_params):>10s} ({enc_params:,})")

# Per GEM block breakdown
gem_block = model.encoder[0]
gem_total = sum(p.numel() for p in gem_block.parameters())
gat_params = sum(p.numel() for p in gem_block.graph_attention.parameters())
mamba_params = sum(p.numel() for p in gem_block.mamba_temporal.parameters())
fusion_params = sum(p.numel() for p in gem_block.fusion_gate.parameters())
ffn_params = sum(p.numel() for p in gem_block.ffn.parameters())
norm_params = sum(p.numel() for p in gem_block.norm1.parameters()) + \
              sum(p.numel() for p in gem_block.norm2.parameters()) + \
              sum(p.numel() for p in gem_block.norm3.parameters())

print(f"\n  Per GEM Block:       {format_params(gem_total):>10s}")
print(f"    - Graph Attention: {format_params(gat_params):>10s}")
print(f"    - Mamba Temporal:  {format_params(mamba_params):>10s}")
print(f"    - Fusion Gate:     {format_params(fusion_params):>10s}")
print(f"    - FFN:             {format_params(ffn_params):>10s}")
print(f"    - Layer Norms:     {format_params(norm_params):>10s}")

# Output projection
proj_params = sum(p.numel() for p in model.proj_out.parameters())
print(f"\n  Output Projection:   {format_params(proj_params):>10s} ({proj_params:,})")

print(f"\n  TOTAL:               {format_params(total_full):>10s}")

# Demonstrasi pattern task-specific model (backbone + classifier head)
# Ini menunjukkan bagaimana MBG digunakan sebagai backbone untuk downstream tasks

from einops.layers.torch import Rearrange

class MBGForClassification(nn.Module):
    """Task-specific model: MBG backbone + classification head."""
    def __init__(self, num_channels=22, num_segments=4):
        super().__init__()
        self.backbone = MBG(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=4, nhead=8, num_channels=num_channels  # 4 layers for demo
        )
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Sequential(
            Rearrange('b c s d -> b d c s'),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(200, 1),
            Rearrange('b 1 -> (b 1)'),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)

tuab_model = MBGForClassification(num_channels=22, num_segments=4).to(device)

print("Task-Specific Classification Model (TUAB pattern):")
print("=" * 50)
print("Backbone: MBG (4 layers for demo)")
print("Task: Binary classification (normal vs abnormal EEG)")
print("Classifier: AdaptiveAvgPool2d + Linear")

# Forward pass untuk classification
# Demo: 22 channels, 4 time segments, 200 points per patch
x_tuab = torch.randn(4, 22, 4, 200).to(device)  # Batch of 4

tuab_model.eval()
with torch.no_grad():
    output_tuab = tuab_model(x_tuab)

print(f"Input shape:  {x_tuab.shape} (B=4, C=22, T=4, P=200)")
print(f"Output shape: {output_tuab.shape} (B*1 = 4 predictions)")
print(f"\nOutput values (logits): {output_tuab[:4].cpu().numpy()}")
print(f"Predictions (sigmoid): {torch.sigmoid(output_tuab[:4]).cpu().numpy()}")
print(f"\nInterpretasi:")
print(f"  - Output > 0.5: Abnormal EEG")
print(f"  - Output < 0.5: Normal EEG")
print(f"\nNote: Pada implementasi penuh (model_for_tuab.py), model menggunakan")
print(f"  12 layers dan konfigurasi channel sesuai dataset target.")

# Demonstrasi AdaptiveAdjacency module
adaptive_adj = AdaptiveAdjacency(num_channels=22, d_model=200)

print("Adaptive Adjacency Module:")
print("=" * 50)
print(f"  Num channels: 22")
print(f"  D model: 200")
print(f"  Alpha (initial): {adaptive_adj.alpha.item():.4f}")
print(f"  Alpha (sigmoid): {torch.sigmoid(adaptive_adj.alpha).item():.4f}")
print(f"  A_prior shape: {adaptive_adj.A_prior.shape}")
print(f"  A_prior is fixed (buffer): True")

# Simulate input: node features (B*T, C, D)
bt = 8 * 4  # batch * time
x_nodes = torch.randn(bt, 22, 200)

with torch.no_grad():
    A = adaptive_adj(x_nodes)

print(f"\n  Input shape:  ({bt}, 22, 200) = (B*T, C, D)")
print(f"  Output shape: {A.shape} = (B*T, C, C)")
print(f"\n  A[0] sample (first 5x5):")
print(A[0, :5, :5].cpu().numpy().round(3))

# Mount Google Drive dan buat direktori project
# from google.colab import drive
# drive.mount('/content/drive')

# Create project directory on Drive
import os
DRIVE_ROOT = './MBG_Project'
CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(DRIVE_ROOT, 'results')
LOG_DIR = os.path.join(DRIVE_ROOT, 'logs')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
print(f"Project directory: {DRIVE_ROOT}")
print(f"Checkpoints: {CHECKPOINT_DIR}")
print(f"Results: {RESULTS_DIR}")
print(f"Logs: {LOG_DIR}")

# Demonstrasi CheckpointManager
from utils.training_utils import CheckpointManager, EarlyStopping, MetricLogger

# Inisialisasi CheckpointManager
checkpoint_manager = CheckpointManager(
    checkpoint_dir=CHECKPOINT_DIR,
    save_every_n_epochs=10,    # Auto-save setiap 10 epoch
    keep_top_k=3,              # Simpan hanya 3 checkpoint terakhir per fold
    metric_name='balanced_accuracy',  # Metrik untuk best model
    metric_mode='max',         # Higher is better
)

# Inisialisasi EarlyStopping
early_stopping = EarlyStopping(
    patience=15,       # Stop jika tidak ada improvement selama 15 epoch
    min_delta=1e-4,    # Minimum improvement yang dianggap signifikan
    mode='max',        # Maximize metric
)

# Inisialisasi MetricLogger (log ke CSV di Drive)
metric_logger = MetricLogger(
    log_dir=LOG_DIR,
    fold=0,
)

print("Checkpoint Manager, EarlyStopping, dan MetricLogger siap!")

# Demonstrasi: Auto-resume dari checkpoint
# Ini berguna ketika Colab session terputus

def resume_training_from_checkpoint(model, optimizer, scheduler=None, scaler=None, fold=0):
    """Resume training dari checkpoint terakhir jika ada."""
    latest_ckpt = checkpoint_manager.get_latest_checkpoint(fold=fold)
    
    if latest_ckpt is not None:
        print(f"Ditemukan checkpoint: {latest_ckpt}")
        ckpt = checkpoint_manager.load_checkpoint(
            path=latest_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device = "cpu", # Forced to CPU because RTX 5060 Ti (sm_120) lacks PyTorch binary support
        )
        start_epoch = ckpt['epoch'] + 1
        metrics_history = ckpt.get('metrics_history', [])
        print(f"Melanjutkan training dari epoch {start_epoch}")
        return start_epoch, metrics_history
    else:
        print("Tidak ada checkpoint. Memulai training dari awal.")
        return 0, []

print("Fungsi resume_training_from_checkpoint siap digunakan.")

# Training loop dengan checkpoint saving (template)

def train_one_fold_with_checkpointing(
    model, train_loader, val_loader, optimizer, scheduler,
    n_epochs=100, fold=0, use_amp=True
):
    """Training loop lengkap dengan checkpoint dan early stopping."""
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    criterion = torch.nn.CrossEntropyLoss()
    
    # Resume dari checkpoint jika ada
    start_epoch, metrics_history = resume_training_from_checkpoint(
        model, optimizer, scheduler, scaler, fold
    )
    
    # Reset early stopping
    early_stopping.reset()
    logger = MetricLogger(log_dir=LOG_DIR, fold=fold)
    best_metric = 0.0
    
    for epoch in range(start_epoch, n_epochs):
        # === Training ===
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
        
        if scheduler is not None:
            scheduler.step()
        
        train_loss /= len(train_loader)
        
        # === Validation ===
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Compute metrics
        import numpy as np
        from utils.kfold_cv import compute_metrics
        metrics = compute_metrics(
            np.array(val_targets), np.array(val_preds), task_type='classification'
        )
        metrics['train_loss'] = train_loss
        metrics['val_loss'] = val_loss
        
        # Log metrics
        logger.log(epoch, metrics)
        metrics_history.append(metrics)
        
        # Print progress
        bal_acc = metrics.get('balanced_accuracy', 0)
        print(f"Epoch {epoch}/{n_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Balanced Acc: {bal_acc:.4f}")
        
        # Save best model
        if bal_acc > best_metric:
            best_metric = bal_acc
            checkpoint_manager.save_best_model(model, metrics, fold=fold)
        
        # Auto-save checkpoint
        if checkpoint_manager.should_save(epoch):
            checkpoint_manager.save_checkpoint(
                model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler,
                epoch=epoch, fold=fold,
                metrics=metrics, metrics_history=metrics_history,
            )
        
        # Early stopping
        if early_stopping(bal_acc):
            print(f"Early stopping di epoch {epoch}!")
            break
    
    return metrics_history, best_metric

print("Training loop template siap!")

# Demo: 5-Fold Subject-Independent Cross Validation
import numpy as np
from utils.kfold_cv import KFoldCrossValidator, compute_metrics, aggregate_fold_results

# Simulasi dataset EEG
n_subjects = 20
samples_per_subject = 50
n_total = n_subjects * samples_per_subject

# Buat label dan group arrays
np.random.seed(42)
labels = np.random.randint(0, 2, n_total)  # Binary classification
subjects = np.repeat(np.arange(n_subjects), samples_per_subject)  # Subject IDs

print(f"Total samples: {n_total}")
print(f"Jumlah subjek: {n_subjects}")
print(f"Samples per subjek: {samples_per_subject}")
print(f"Distribusi kelas: {np.bincount(labels)}")

# Inisialisasi 5-Fold Subject-Independent CV
cv = KFoldCrossValidator(
    n_folds=5,
    split_type='group',  # GroupKFold - subject-independent!
    random_seed=42,
    results_dir=RESULTS_DIR,
)

# Dapatkan splits
splits = cv.get_splits(labels, groups=subjects)

print(f"\nJumlah folds: {len(splits)}")
for i, (train_idx, test_idx) in enumerate(splits):
    train_subj = np.unique(subjects[train_idx])
    test_subj = np.unique(subjects[test_idx])
    print(f"Fold {i}: Train={len(train_subj)} subjek ({len(train_idx)} samples), "
          f"Test={len(test_subj)} subjek ({len(test_idx)} samples)")
    # Verifikasi tidak ada overlap
    overlap = set(train_subj) & set(test_subj)
    assert len(overlap) == 0, f"DATA LEAKAGE pada fold {i}!"

print("\nVerifikasi: Tidak ada subject overlap di semua fold!")

# Demo: Leave-One-Subject-Out (LOSO) Cross Validation
# Cocok untuk dataset kecil (<20 subjek)

cv_loso = KFoldCrossValidator(
    split_type='loso',
    random_seed=42,
    results_dir=RESULTS_DIR,
)

splits_loso = cv_loso.get_splits(labels, groups=subjects)

print(f"LOSO: {len(splits_loso)} folds (= {n_subjects} subjek)")
print("\nSetiap fold: 1 subjek untuk test, sisanya untuk train")
print(f"Test set per fold: {samples_per_subject} samples")
print(f"Train set per fold: {(n_subjects-1) * samples_per_subject} samples")

# Cek beberapa fold
for i in range(min(3, len(splits_loso))):
    train_idx, test_idx = splits_loso[i]
    test_subj = np.unique(subjects[test_idx])
    print(f"\nFold {i}: Test subjek = {test_subj[0]}, "
          f"Train = {len(train_idx)} samples, Test = {len(test_idx)} samples")

# Demo: Agregasi hasil dan pelaporan statistik
# Simulasi hasil 5-fold CV

# Contoh hasil per-fold (simulasi)
simulated_fold_results = [
    {'accuracy': 0.85, 'balanced_accuracy': 0.84, 'f1': 0.83, 'auc_roc': 0.91, 'cohen_kappa': 0.69},
    {'accuracy': 0.87, 'balanced_accuracy': 0.86, 'f1': 0.85, 'auc_roc': 0.93, 'cohen_kappa': 0.73},
    {'accuracy': 0.82, 'balanced_accuracy': 0.81, 'f1': 0.80, 'auc_roc': 0.89, 'cohen_kappa': 0.63},
    {'accuracy': 0.89, 'balanced_accuracy': 0.88, 'f1': 0.87, 'auc_roc': 0.94, 'cohen_kappa': 0.77},
    {'accuracy': 0.86, 'balanced_accuracy': 0.85, 'f1': 0.84, 'auc_roc': 0.92, 'cohen_kappa': 0.71},
]

# Agregasi
aggregated = aggregate_fold_results(simulated_fold_results, confidence_level=0.95)

print("=" * 70)
print("  HASIL K-FOLD CROSS VALIDATION (5-Fold, Subject-Independent)")
print("=" * 70)
print(f"{'Metric':<22} {'Mean':>8} {'Std':>8} {'95% CI':>20}")
print("-" * 70)
for metric_name, stats in aggregated.items():
    ci_str = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
    print(f"{metric_name:<22} {stats['mean']:>8.4f} {stats['std']:>8.4f} {ci_str:>20}")
print("=" * 70)

# Demo: Statistical Testing (Paired t-test)
# Membandingkan dua metode secara statistik
from utils.kfold_cv import statistical_test

# Simulasi: MBG vs Baseline (CBraMod)
mbg_results = [0.85, 0.87, 0.82, 0.89, 0.86]      # Balanced accuracy per fold
baseline_results = [0.80, 0.82, 0.78, 0.84, 0.81]  # Baseline per fold

test_result = statistical_test(mbg_results, baseline_results, test_type='both')

print("=" * 60)
print("  STATISTICAL SIGNIFICANCE TEST")
print("  MBG vs Baseline (CBraMod)")
print("=" * 60)
print(f"\nMBG:      {test_result['mean_a']:.4f} +/- {test_result['std_a']:.4f}")
print(f"Baseline: {test_result['mean_b']:.4f} +/- {test_result['std_b']:.4f}")
print(f"Mean difference: {test_result['mean_difference']:.4f}")

print(f"\n--- Paired t-test ---")
ttest = test_result['paired_ttest']
print(f"  t-statistic: {ttest['t_statistic']:.4f}")
print(f"  p-value: {ttest['p_value']:.6f}")
print(f"  Significant (p<0.05): {'YA' if ttest['significant_005'] else 'TIDAK'}")
print(f"  Significant (p<0.01): {'YA' if ttest['significant_001'] else 'TIDAK'}")

print(f"\n--- Wilcoxon signed-rank test ---")
if 'error' not in test_result.get('wilcoxon', {}):
    wilcox = test_result['wilcoxon']
    print(f"  W-statistic: {wilcox['w_statistic']:.4f}")
    print(f"  p-value: {wilcox['p_value']:.6f}")
    print(f"  Significant (p<0.05): {'YA' if wilcox['significant_005'] else 'TIDAK'}")
else:
    print(f"  {test_result['wilcoxon']['note']}")

print(f"\n--- Kesimpulan untuk Paper ---")
if ttest['significant_005']:
    print("  MBG secara SIGNIFIKAN lebih baik dari baseline (p<0.05).")
    print("  Bisa dilaporkan: 'MBG significantly outperforms the baseline")
    print(f"  (p={ttest['p_value']:.4f}, paired t-test, 5-fold CV).'")
else:
    print("  Perbedaan TIDAK signifikan secara statistik.")

# Generate LaTeX table untuk paper
cv.fold_results = simulated_fold_results

latex_table = cv.generate_latex_table(
    method_name='MBG (Ours)',
    metrics_to_show=['accuracy', 'balanced_accuracy', 'f1', 'auc_roc', 'cohen_kappa'],
)

print("LaTeX table untuk paper:")
print("=" * 60)
print(latex_table)
print("=" * 60)
print("\nCopy-paste ke file .tex Anda!")

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

from utils.training_utils import CheckpointManager, EarlyStopping, MetricLogger
from utils.kfold_cv import (
    KFoldCrossValidator, compute_metrics, aggregate_fold_results, statistical_test
)


def train_mbg_kfold(
    dataset,
    subjects,
    labels,
    n_folds=5,
    n_epochs=100,
    batch_size=32,
    lr=5e-5,
    checkpoint_dir=CHECKPOINT_DIR,
    results_dir=RESULTS_DIR,
    log_dir=LOG_DIR,
    use_amp=True,
    patience=15,
    task_type='classification',
    model_fn=None,
    num_classes=2,
    use_gradient_checkpointing=True,
    save_every_n_epochs=10,
    keep_top_k=3,
):
    """Complete training pipeline with K-fold CV, checkpointing, and early stopping.
    
    Args:
        dataset: Dataset or tensor of EEG data.
        subjects: Array of subject IDs for each sample.
        labels: Array of labels for each sample.
        n_folds: Number of cross-validation folds.
        n_epochs: Maximum number of training epochs per fold.
        batch_size: Training batch size.
        lr: Learning rate.
        checkpoint_dir: Directory for checkpoints on Drive.
        results_dir: Directory for results on Drive.
        log_dir: Directory for training logs on Drive.
        use_amp: Whether to use mixed precision training.
        patience: Early stopping patience.
        task_type: 'classification' or 'regression'.
        model_fn: Function that returns a new model instance.
        num_classes: Number of output classes.
        use_gradient_checkpointing: Enable gradient checkpointing.
        save_every_n_epochs: Auto-save interval.
        keep_top_k: Keep top-K checkpoints per fold.
    
    Returns:
        Dictionary with per-fold and aggregated results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup K-Fold CV (subject-independent)
    cv = KFoldCrossValidator(
        n_folds=n_folds,
        split_type='group',
        random_seed=42,
        results_dir=results_dir,
    )
    
    subjects_arr = np.array(subjects)
    labels_arr = np.array(labels)
    splits = cv.get_splits(labels_arr, groups=subjects_arr)
    
    all_fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx + 1}/{len(splits)}")
        print(f"{'='*60}")
        
        # Verify subject independence
        train_subjects = set(subjects_arr[train_idx])
        test_subjects = set(subjects_arr[test_idx])
        assert len(train_subjects & test_subjects) == 0, "DATA LEAKAGE!"
        print(f"  Train: {len(train_idx)} samples from {len(train_subjects)} subjects")
        print(f"  Test:  {len(test_idx)} samples from {len(test_subjects)} subjects")
        
        # Create data loaders
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        # Create fresh model for each fold
        if model_fn is not None:
            model = model_fn().to(device)
        else:
            from models.model_for_tuab import MBGForTUAB
            model = MBGForTUAB(num_classes=num_classes).to(device)
        
        # Enable gradient checkpointing
        if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
        criterion = nn.CrossEntropyLoss()
        
        # Checkpoint and early stopping
        ckpt_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            save_every_n_epochs=save_every_n_epochs,
            keep_top_k=keep_top_k,
            metric_name='balanced_accuracy',
            metric_mode='max',
        )
        early_stop = EarlyStopping(patience=patience, mode='max')
        logger = MetricLogger(log_dir=log_dir, fold=fold_idx)
        
        # Resume from checkpoint if available
        latest_ckpt = ckpt_manager.get_latest_checkpoint(fold=fold_idx)
        start_epoch = 0
        metrics_history = []
        best_metric = 0.0
        
        if latest_ckpt is not None:
            ckpt = ckpt_manager.load_checkpoint(
                latest_ckpt, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, device=str(device),
            )
            start_epoch = ckpt['epoch'] + 1
            metrics_history = ckpt.get('metrics_history', [])
            best_metric = ckpt.get('best_metric', 0.0) or 0.0
            print(f"  Resumed from epoch {start_epoch}")
        
        # Training loop
        for epoch in range(start_epoch, n_epochs):
            # Train
            model.train()
            train_loss = 0.0
            
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
            
            scheduler.step()
            train_loss /= max(len(train_loader), 1)
            
            # Evaluate
            model.eval()
            all_preds, all_targets, all_probs = [], [], []
            val_loss = 0.0
            
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(device), targets.to(device)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = model(data)
                    else:
                        outputs = model(data)
                    
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    probs = torch.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            val_loss /= max(len(test_loader), 1)
            
            # Compute metrics
            y_true = np.array(all_targets)
            y_pred = np.array(all_preds)
            y_prob = np.array(all_probs)
            
            metrics = compute_metrics(y_true, y_pred, y_prob, task_type=task_type)
            metrics['train_loss'] = train_loss
            metrics['val_loss'] = val_loss
            metrics['lr'] = optimizer.param_groups[0]['lr']
            
            logger.log(epoch, metrics)
            metrics_history.append(metrics)
            
            bal_acc = metrics.get('balanced_accuracy', 0)
            
            # Print progress (every 10 epochs)
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch:3d}/{n_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Bal.Acc: {bal_acc:.4f}")
            
            # Save best model
            if bal_acc > best_metric:
                best_metric = bal_acc
                ckpt_manager.save_best_model(model, metrics, fold=fold_idx)
            
            # Auto-save checkpoint
            if ckpt_manager.should_save(epoch):
                ckpt_manager.save_checkpoint(
                    model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler,
                    epoch=epoch, fold=fold_idx,
                    metrics=metrics, metrics_history=metrics_history,
                )
            
            # Early stopping check
            if early_stop(bal_acc):
                print(f"  Early stopping at epoch {epoch}!")
                break
        
        # Store fold results (best epoch metrics)
        best_epoch_metrics = logger.get_best_epoch('balanced_accuracy', mode='max')
        fold_result = {k: v for k, v in best_epoch_metrics.items() 
                       if k not in ('epoch', 'timestamp', 'train_loss', 'val_loss', 'lr')}
        all_fold_results.append(fold_result)
        
        print(f"\n  Fold {fold_idx + 1} Best: Balanced Accuracy = {best_metric:.4f}")
        
        # Clean up GPU memory
        del model, optimizer, scheduler, scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Aggregate results
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS ({len(splits)}-Fold Subject-Independent CV)")
    print(f"{'='*60}")
    
    aggregated = aggregate_fold_results(all_fold_results)
    for name, stats in aggregated.items():
        print(f"  {name}: {stats['mean']:.4f} +/- {stats['std']:.4f} "
              f"(95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])")
    
    # Save results
    cv.fold_results = all_fold_results
    results = {
        'split_type': 'group',
        'n_folds': len(splits),
        'task_type': task_type,
        'fold_results': all_fold_results,
        'aggregated': aggregated,
    }
    cv._save_results(results)
    
    # Generate LaTeX table
    print(f"\n--- LaTeX Table ---")
    print(cv.generate_latex_table(method_name='MBG (Ours)'))
    
    return results


print("train_mbg_kfold() siap digunakan!")
print("\nContoh penggunaan:")
print("  results = train_mbg_kfold(dataset, subjects, labels, n_folds=5, n_epochs=100)")

# Contoh penggunaan dengan mock data (untuk verifikasi pipeline berjalan)

# Buat mock dataset
n_subjects_demo = 10
samples_per_subj = 20
n_channels = 22
n_timepoints = 30
patch_size = 200
n_total_demo = n_subjects_demo * samples_per_subj

# Mock EEG data: (N, C, T, P)
mock_data = torch.randn(n_total_demo, n_channels, n_timepoints, patch_size)
mock_labels = torch.randint(0, 2, (n_total_demo,))
mock_subjects = np.repeat(np.arange(n_subjects_demo), samples_per_subj)

# Buat TensorDataset
mock_dataset = TensorDataset(mock_data, mock_labels)

print(f"Mock dataset: {n_total_demo} samples, {n_subjects_demo} subjects")
print(f"Data shape: {mock_data.shape}")
print(f"Labels distribution: {torch.bincount(mock_labels).tolist()}")
print("\nUntuk menjalankan training penuh:")
print("  results = train_mbg_kfold(mock_dataset, mock_subjects, mock_labels.numpy(), n_folds=5)")
print("\n(Jalankan cell di atas jika ingin test pipeline -- membutuhkan beberapa menit)")

# GPU Memory Monitoring
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved()/1e6:.1f} MB")
    torch.cuda.empty_cache()

from torch.cuda.amp import autocast, GradScaler

# Mixed precision forward pass
with torch.no_grad():
    with autocast(dtype=torch.float16):
        output_fp16 = model_small(x)
print(f"FP16 output shape: {output_fp16.shape}")

import time

# Benchmark function
def benchmark_inference(model, x, n_runs=10, device='cuda'):
    model = model.to(device)
    x = x.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed / n_runs

# Run benchmarks
x_bench = torch.randn(8, 22, 4, 200)
model_bench = MBG(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, 
                  seq_len=30, n_layer=4, nhead=8, num_channels=22)

gpu_time = benchmark_inference(model_bench, x_bench, device='cuda')
# Mamba SSM does not support CPU inference natively (needs CUDA)
print(f"GPU inference: {gpu_time*1000:.2f} ms")
print("CPU inference benchmark skipped (Mamba requires CUDA)")

# For training with limited VRAM, use gradient checkpointing
from torch.utils.checkpoint import checkpoint

# The model supports gradient checkpointing for memory-efficient training
model_train = MBG(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                  seq_len=30, n_layer=12, nhead=8, num_channels=22).to(device)

# Enable gradient checkpointing
model_train.gradient_checkpointing = True
print(f"Gradient checkpointing enabled: {model_train.gradient_checkpointing}")

# Auto-adjust model size based on available GPU memory
def get_optimal_config(device):
    """Determine optimal model config based on GPU memory."""
    if not torch.cuda.is_available():
        return {'n_layer': 4, 'batch_size': 4, 'note': 'CPU mode - reduced config'}
    
    total_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    
    if total_mem_gb >= 40:  # A100
        return {'n_layer': 12, 'batch_size': 64, 'note': f'A100 ({total_mem_gb:.0f}GB) - full config'}
    elif total_mem_gb >= 15:  # T4/V100
        return {'n_layer': 12, 'batch_size': 32, 'note': f'T4/V100 ({total_mem_gb:.0f}GB) - full model, reduced batch'}
    else:  # Older GPUs
        return {'n_layer': 8, 'batch_size': 16, 'note': f'Limited GPU ({total_mem_gb:.0f}GB) - reduced config'}

config = get_optimal_config(device)
print(f"Optimal config: {config}")

print("="  * 60)
print("  MBG: Mamba-Based Graph Foundation Model for EEG")
print("  Panduan Komprehensif - SELESAI")
print("=" * 60)
print()
print("Notebook ini mencakup:")
print("  [x] Setup dan instalasi")
print("  [x] Background (CBraMod, EEGMamba)")
print("  [x] Gap analysis")
print("  [x] Arsitektur MBG")
print("  [x] 4 Key novelties")
print("  [x] Tabel perbandingan")
print("  [x] Code walkthrough lengkap")
print("  [x] 14 Downstream tasks")
print("  [x] Potential experiments")
print("  [x] Paper writing support")
print("  [x] GPU Optimization (AMP, Gradient Checkpointing)")
print("  [x] Google Drive Checkpoint Management")
print("  [x] K-Fold Cross Validation (Subject-Independent)")
print("  [x] Complete Training Pipeline (Production-Ready)")

