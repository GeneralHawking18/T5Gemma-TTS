import os
import torch
import soundfile as sf
import librosa
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from einops import pack, unpack, rearrange
from vector_quantize_pytorch.residual_fsq import ResidualFSQ


from torch import nn
from modeling_xcodec2 import XCodec2Model


# =============================================================================
# ONNX-Compatible Monkey Patch cho ResidualFSQ
# Thay thế einx.get_at() bằng torch operations để tương thích với ONNX tracing
# =============================================================================
def onnx_compatible_get_codes_from_indices(self, indices):
    """
    ONNX-compatible version of get_codes_from_indices.
    Thay thế einx.get_at() bằng F.embedding() để tương thích ONNX.
    """
    batch, quantize_dim = indices.shape[0], indices.shape[-1]

    # Pack indices (giữ nguyên logic gốc)
    indices, ps = pack([indices], 'b * q')

    # Padding nếu quantize_dim < num_quantizers (cho quantize dropout)
    if quantize_dim < self.num_quantizers:
        indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

    # Mask và fill dummy value
    mask = indices == -1
    indices = indices.masked_fill(mask, 0)

    # THAY THẾ einx.get_at() bằng F.embedding (ONNX-compatible)
    # self.codebooks: [num_quantizers, codebook_size, codebook_dim]
    # indices: [batch, seq, num_quantizers]
    batch_size, seq_len, num_q = indices.shape
    all_codes = []

    for q_idx in range(self.num_quantizers):
        # Lấy indices cho quantizer này: [batch, seq]
        idx_q = indices[:, :, q_idx]

        # Dùng F.embedding (ONNX-compatible) thay vì einx.get_at
        # codebooks[q_idx]: [codebook_size, codebook_dim]
        codes_q = F.embedding(idx_q, self.codebooks[q_idx])
        # codes_q: [batch, seq, codebook_dim]

        all_codes.append(codes_q)

    # Stack: [num_quantizers, batch, seq, codebook_dim]
    all_codes = torch.stack(all_codes, dim=0)

    # Mask out dropout codes
    all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)

    # Scale theo từng quantizer
    scales = rearrange(self.scales, 'q d -> q 1 1 d')
    all_codes = all_codes * scales

    # Unpack về shape gốc
    all_codes, = unpack(all_codes, ps, 'q b * d')

    return all_codes


# Apply monkey-patch TRƯỚC khi tạo model
ResidualFSQ.get_codes_from_indices = onnx_compatible_get_codes_from_indices


# =============================================================================
# ONNX-Compatible ISTFT Implementation
# Thay thế complex numbers bằng real operations sử dụng DFT matrix
# =============================================================================
def irfft_real_onnx(spec_real, spec_imag, n_fft):
    """
    ONNX-compatible IRFFT implementation using real arithmetic only.

    Thay thế torch.fft.irfft(complex_spec) bằng matmul với DFT matrix.

    Args:
        spec_real: [B, N//2+1, T] - Real part of spectrum (mag * cos(phase))
        spec_imag: [B, N//2+1, T] - Imaginary part of spectrum (mag * sin(phase))
        n_fft: FFT size

    Returns:
        signal: [B, N, T] - Time domain signal
    """
    B, freq_bins, T = spec_real.shape
    device = spec_real.device
    dtype = spec_real.dtype

    # Tạo IDFT basis: cos và sin components
    # Cho n = 0..N-1 và k = 0..N//2
    n = torch.arange(n_fft, device=device, dtype=dtype).unsqueeze(1)  # [N, 1]
    k = torch.arange(freq_bins, device=device, dtype=dtype).unsqueeze(0)  # [1, freq_bins]

    # omega = 2*pi*k*n/N
    omega = 2 * np.pi * k * n / n_fft  # [N, freq_bins]

    # IDFT basis: cos và -sin (vì irfft uses negative sign for imag)
    cos_basis = torch.cos(omega)  # [N, freq_bins]
    sin_basis = torch.sin(omega)  # [N, freq_bins]

    # Scale factors: DC và Nyquist có scale 1, còn lại có scale 2 (để bù cho conjugate symmetry)
    scale = torch.ones(freq_bins, device=device, dtype=dtype) * 2.0
    scale[0] = 1.0  # DC
    if freq_bins == n_fft // 2 + 1 and n_fft % 2 == 0:
        scale[-1] = 1.0  # Nyquist (chỉ khi n_fft chẵn)

    # Apply scale
    cos_basis = cos_basis * scale.unsqueeze(0)  # [N, freq_bins]
    sin_basis = sin_basis * scale.unsqueeze(0)  # [N, freq_bins]

    # IRFFT: x[n] = (1/N) * sum_k (real[k] * cos(2*pi*k*n/N) - imag[k] * sin(2*pi*k*n/N)) * scale[k]
    # Reshape spec: [B, freq_bins, T] -> [B, T, freq_bins]
    spec_real_t = spec_real.permute(0, 2, 1)  # [B, T, freq_bins]
    spec_imag_t = spec_imag.permute(0, 2, 1)  # [B, T, freq_bins]

    # Matmul: [B, T, freq_bins] @ [freq_bins, N] -> [B, T, N]
    signal = (torch.matmul(spec_real_t, cos_basis.T) -
              torch.matmul(spec_imag_t, sin_basis.T)) / n_fft

    # Permute back: [B, T, N] -> [B, N, T]
    signal = signal.permute(0, 2, 1)

    return signal


# =============================================================================
# Pre-computed kernels cho Overlap-Add (static shapes cho ONNX export)
# =============================================================================
# ISTFT config từ model: n_fft=392, hop_length=98, win_length=392
ISTFT_WIN_LENGTH = 392
ISTFT_HOP_LENGTH = 98

# Pre-compute upsample kernel: [392, 1, 1]
_upsample_kernel = None

# Pre-compute shift kernel: [1, 392, 392]
_shift_kernel = None


def get_overlap_add_kernels(device, dtype):
    """Lazy init và cache kernels cho overlap-add."""
    global _upsample_kernel, _shift_kernel

    if _upsample_kernel is None or _upsample_kernel.device != device:
        W = ISTFT_WIN_LENGTH

        # Upsample kernel: [W, 1, 1]
        _upsample_kernel = torch.ones(W, 1, 1, device=device, dtype=dtype)

        # Shift kernel: [1, W, W]
        _shift_kernel = torch.zeros(1, W, W, device=device, dtype=dtype)
        for w in range(W):
            _shift_kernel[0, w, W - 1 - w] = 1.0

    return _upsample_kernel, _shift_kernel


def overlap_add_onnx(frames, hop_length, win_length):
    """
    TensorRT-compatible overlap-add using ConvTranspose1d + Conv1d.
    Thay thế scatter_add vì TensorRT không hỗ trợ ScatterElements với reduction.

    Args:
        frames: [B, win_length, T] - Windowed frames
        hop_length: Hop size
        win_length: Window length

    Returns:
        signal: [B, output_length] - Reconstructed signal
    """
    B, W, T = frames.shape
    device = frames.device
    dtype = frames.dtype

    # Get pre-computed kernels
    upsample_kernel, shift_kernel = get_overlap_add_kernels(device, dtype)

    # Step 1: Upsample mỗi channel với stride=hop_length
    upsampled = F.conv_transpose1d(frames, upsample_kernel, stride=hop_length, groups=W)
    # upsampled: [B, W, (T-1)*hop+1]

    # Step 2: Pad để chuẩn bị cho shift
    upsampled_padded = F.pad(upsampled, (0, W - 1))  # [B, W, L+W-1]

    # Step 3: Apply shift kernel để shift và sum
    output = F.conv1d(upsampled_padded, shift_kernel)  # [B, 1, output_length]
    output = output.squeeze(1)  # [B, output_length]

    return output


def onnx_compatible_istft_forward(self, spec_real, spec_imag):
    """
    ONNX-compatible ISTFT forward.

    Args:
        spec_real: [B, N//2+1, T] - Real part (mag * cos(phase))
        spec_imag: [B, N//2+1, T] - Imag part (mag * sin(phase))

    Returns:
        y: [B, audio_len] - Reconstructed audio
    """
    if self.padding == "same":
        pad_left = self._pad_left
        pad_right = self._pad_right
    else:
        raise ValueError("Chỉ hỗ trợ padding='same' cho ONNX export")

    B, N, T = spec_real.shape

    # Inverse FFT sử dụng real arithmetic
    ifft = irfft_real_onnx(spec_real, spec_imag, self.n_fft)  # [B, n_fft, T]

    # Apply window
    ifft = ifft * self.window.unsqueeze(0).unsqueeze(-1)  # [B, n_fft, T]

    # Overlap and Add sử dụng scatter_add (ONNX-compatible)
    output_length = (T - 1) * self.hop_length + self.win_length
    y = overlap_add_onnx(ifft, self.hop_length, self.win_length)  # [B, output_length]

    # Slice để bỏ padding
    y = y[:, pad_left: (-pad_right if pad_right > 0 else None)]

    # Window envelope - tính bằng cách tương tự
    window_sq = self.window.square()  # [win_length]

    # Tạo frames cho window envelope
    window_frames = window_sq.unsqueeze(0).unsqueeze(-1).expand(1, -1, T)  # [1, win_length, T]
    window_envelope = overlap_add_onnx(window_frames, self.hop_length, self.win_length).squeeze(0)  # [output_length]
    window_envelope = window_envelope[pad_left: (-pad_right if pad_right > 0 else None)]

    # Normalize - tránh division by zero
    y = y / (window_envelope.unsqueeze(0) + 1e-8)

    return y


def onnx_compatible_istft_head_forward(self, x):
    """
    ONNX-compatible ISTFTHead forward - không dùng complex numbers.
    """
    x_pred = self.out(x)
    x_pred = x_pred.transpose(1, 2)
    mag, p = x_pred.chunk(2, dim=1)
    mag = torch.exp(mag)
    mag = torch.clip(mag, max=1e2)

    # Tạo real và imag parts thay vì complex number
    cos_p = torch.cos(p)
    sin_p = torch.sin(p)
    spec_real = mag * cos_p  # Real part: mag * cos(phase)
    spec_imag = mag * sin_p  # Imag part: mag * sin(phase)

    # Gọi ISTFT với real inputs
    audio = onnx_compatible_istft_forward(self.istft, spec_real, spec_imag)

    return audio.unsqueeze(1), x_pred


# Import và monkey-patch ISTFTHead
from vq.codec_decoder_vocos import ISTFTHead
ISTFTHead.forward = onnx_compatible_istft_head_forward


# =============================================================================
# Model Wrapper cho ONNX Export
# =============================================================================
# Define device globally để dùng trong _init_model
device = "cuda" if torch.cuda.is_available() else "cpu"


class T5GemmaAudioTokenizer(nn.Module):
    def __init__(self):
        super().__init__()

        temp_model = self._init_model()
        self.fc_post_a = temp_model.fc_post_a            # Lớp Linear trung gian
        self.generator = temp_model.generator            # Vocoder (Tạo ra sóng âm)

        del temp_model
        torch.cuda.empty_cache() # Dọn rác VRAM nếu dùng GPU

    def _init_model(self):
        model_path = "NandemoGHS/Anime-XCodec2-44.1kHz-v2"
        model = XCodec2Model.from_pretrained(model_path)
        model.eval().to(device)
        return model

    def forward(self, vq_code: torch.Tensor):
        vq_post_emb = self.generator.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
        vq_post_emb = vq_post_emb.transpose(1, 2)  # [batch, 1024, frames]
        vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)
        recon_audio = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0] 
        
        return recon_audio

# ==========================================
# 3. Logic: Có Cache hay Không?
# ==========================================

if __name__ == "__main__":

    cache_file = "vq_code_cache.npy"  # Tên file để lưu/đọc cache
    vq_code = None

    if os.path.exists(cache_file):
        # === TRƯỜNG HỢP 1: ĐÃ CÓ FILE NPY (Nhanh) ===
        print(f"✅ Tìm thấy file cache '{cache_file}'. Đang load trực tiếp...")
        
        # Load từ file npy
        vq_numpy = np.load(cache_file)
        
        # Chuyển thành Tensor & đẩy lên GPU
        vq_code = torch.from_numpy(vq_numpy).to(device)
        

    print("Đang export ONNX với ONNX-compatible implementations...")
    model = T5GemmaAudioTokenizer().eval()

    # Pre-init kernels với đúng device và dtype trước khi export
    # Điều này đảm bảo kernels là static tensors
    get_overlap_add_kernels(device, torch.float32)
    print(f"Kernels initialized on {device}")

    # Export với standard ONNX exporter (opset 17)
    torch.onnx.export(
        model=model,
        args=(vq_code,),
        f="./vocoder.onnx",
        input_names=['vq_code'],
        output_names=['recon_audio'],
        opset_version=17,
        dynamic_axes={
            "vq_code": {0: "batch_size", 2: "seq_length"},
            "recon_audio": {0: "batch_size", 2: "audio_length"},
        }
    )
    print("✅ ONNX export thành công!")


