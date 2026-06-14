# Panduan Setup & Reproduksi MBG (Mamba-Based Graph)

Dokumen ini menjelaskan langkah-langkah untuk melakukan *setup environment* dan memodifikasi konfigurasi (patching) agar arsitektur **MBG** dapat dijalankan dengan lancar tanpa *error*, khususnya pada GPU generasi terbaru (NVIDIA RTX 50 series).

## 1. Spesifikasi Environment (Hardware & Software)
Untuk mereproduksi keberhasilan eksekusi, berikut adalah spesifikasi sistem yang digunakan:
- **OS**: Linux
- **GPU**: NVIDIA GeForce RTX 5060 Ti (Arsitektur `sm_120`)
- **CUDA Version**: 12.8
- **Python Version**: 3.12
- **Virtual Environment**: Python `venv` (`~/awangga/venv`)

## 2. Pembuatan Virtual Environment
Pastikan Anda membuat dan menggunakan *virtual environment* yang bersih agar *dependencies* tidak bentrok.
```bash
# Membuat virtual environment
python3.12 -m venv ~/awangga/venv

# Mengaktifkan virtual environment
source ~/awangga/venv/bin/activate
```

## 3. Instalasi Dependencies
Karena GPU RTX 5060 Ti memerlukan CUDA yang sangat baru, instalasi komponen Mamba-SSM sangat rawan terhadap isu inkompatibilitas. Berikut adalah versi library spesifik yang **WAJIB** digunakan:

1. **Instalasi PyTorch**
   Gunakan versi `2.7.0+cu128` (atau versi Nightly/spesifik CUDA 12.8) yang kompatibel dengan hasil kompilasi roda (wheel) lokal Anda.
   ```bash
   pip install torch==2.7.0 --extra-index-url https://download.pytorch.org/whl/cu128
   ```

2. **Instalasi Transformers**
   Hindari versi `transformers` terbaru (seperti 4.51.x) karena dapat memicu `ImportError` akibat penghapusan fungsi lama. Gunakan versi `4.40.0`.
   ```bash
   pip install transformers==4.40.0
   ```

3. **Instalasi Causal-Conv1d dan Mamba-SSM**
   Anda harus mengompilasi library ini secara manual dari source agar sesuai dengan arsitektur GPU terbaru. Proses ini memakan waktu sekitar 5-15 menit.
   ```bash
   pip install "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.0"
   pip install "git+https://github.com/state-spaces/mamba.git@v2.2.4"
   ```

## 4. Code Patching (SANGAT PENTING!)
Versi `mamba-ssm (2.2.4)` memiliki ketidakcocokan (mismatch) argumen dengan fungsi backend C++ pada `causal-conv1d (1.5.0)`. Jika tidak dimodifikasi, Anda akan mendapatkan `TypeError` saat menjalankan *forward pass*. 

Anda harus mengedit file *source code* mamba-ssm di dalam virtual environment Anda:
**File:** `~/awangga/venv/lib/python3.12/site-packages/mamba_ssm/ops/selective_scan_interface.py`

**Ubah blok kode `forward` (sekitar baris 208):**
*Sebelum:*
```python
conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
    x, conv1d_weight, conv1d_bias, None, None, None, True
)
```
*Sesudah:*
```python
conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
import causal_conv1d.cpp_functions as cpp_functions
conv1d_out = cpp_functions.causal_conv1d_fwd_function(
    x, conv1d_weight, conv1d_bias, None, None, None, True
)
```

**Ubah blok kode `backward` (sekitar baris 356):**
*Sebelum:*
```python
dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
    x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
)
```
*Sesudah:*
```python
import causal_conv1d.cpp_functions as cpp_functions
dx, dconv1d_weight, dconv1d_bias, *_ = cpp_functions.causal_conv1d_bwd_function(
    x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
)
```

## 5. Penyesuaian pada MBG_Comprehensive_Guide.py
Bila Anda menjalankan script demonstrasi bawaan (`MBG_Comprehensive_Guide.py`), lakukan perbaikan berikut agar tidak *crash*:
1. Pastikan model demonstrasi dan dataset dummy menggunakan memori GPU yang sama:
   ```python
   # Tambahkan .to(device)
   x_tuab = torch.randn(4, 22, 4, 200).to(device)
   tuab_model = MBGForClassification(num_channels=22, num_segments=4).to(device)
   ```
2. Saat me-logging tensor dari GPU, pastikan diubah ke memori CPU terlebih dahulu dengan memanggil `.cpu().numpy()`.
3. Skrip secara bawaan melakukan import `google.colab`. Anda harus mengomentari (*comment out*) baris Google Drive tersebut dan mengubah path `DRIVE_ROOT` ke folder lokal seperti `./MBG_Project`.
4. Jangan jalankan fungsi benchmark inference pada mode CPU (`device='cpu'`), karena algoritma *Selective Scan Mamba* membutuhkan fungsionalitas CUDA secara native.

## Kesimpulan
Dengan mengikuti pedoman *dependencies* dan menyematkan patch python pada *interface* Mamba di atas, model Mamba-Based Graph akan berjalan dengan mulus secara _native_ pada arsitektur perangkat keras *cutting-edge* seperti NVIDIA seri RTX 50 (Blackwell).
