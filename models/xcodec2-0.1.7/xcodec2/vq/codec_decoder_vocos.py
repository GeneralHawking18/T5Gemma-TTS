import sys

import numpy as np
import torch
import torch.nn as nn
from .residual_vq import ResidualVQ
from .module import WNConv1d, DecoderBlock, ResLSTM
from .alias_free_torch import *
from .  import activations
from typing import Optional, List
from .module   import ConvNeXtBlock,   AdaLayerNorm
from .bs_roformer5 import TransformerBlock
# from rotary_embedding_torch import RotaryEmbedding
from torchtune.modules import RotaryPositionalEmbeddings
from vector_quantize_pytorch import ResidualFSQ
from torch.nn import Module, ModuleList
from torch.nn.utils.parametrizations import weight_norm
import typing as tp
class ISTFT(nn.Module):
    """
    Triển khai tùy chỉnh của ISTFT vì torch.istft không cho phép padding tùy chỉnh (khác `center=True`) với windowing.
    Lý do là kiểm tra NOLA (Nonzero Overlap Add) thất bại ở các cạnh.
    Xem vấn đề: https://github.com/pytorch/pytorch/issues/62323
    Cụ thể, trong bối cảnh vocoding nơ-ron, chúng ta quan tâm đến padding "same" tương tự như CNN.
    Ràng buộc NOLA được đáp ứng vì dù sao chúng ta cũng cắt bỏ các mẫu được đệm.

    Tham số:
        n_fft (int): Kích thước của biến đổi Fourier.
        hop_length (int): Khoảng cách giữa các khung cửa sổ trượt liền kề.
        win_length (int): Kích thước của khung cửa sổ và bộ lọc STFT.
        padding (str, optional): Loại padding. Các tùy chọn là "center" hoặc "same". Mặc định là "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding phải là 'center' hoặc 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)
        if self.padding == "same":
            d = int(self.win_length - self.hop_length)
            self._pad_left  = max(0, d // 2)
            self._pad_right = max(0, d - self._pad_left)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Tính toán Biến đổi Fourier Thời gian Ngắn Ngược (ISTFT) của một spectrogram phức hợp.

        Tham số:
            spec (Tensor): Spectrogram phức hợp đầu vào có dạng (B, N, T), trong đó B là kích thước batch,
                            N là số lượng bin tần số, và T là số lượng khung thời gian.

        Trả về:
            Tensor: Tín hiệu miền thời gian được tái tạo có dạng (B, L), trong đó L là độ dài của tín hiệu đầu ra.
        """
        if self.padding == "center":
            # Quay về triển khai gốc của pytorch
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad_left  = self._pad_left
            pad_right = self._pad_right
        else:
            raise ValueError("Padding phải là 'center' hoặc 'same'.")

        assert spec.dim() == 3, "Mong đợi đầu vào là tensor 3D"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )[:, 0, 0, pad_left: ( -pad_right if pad_right > 0 else None)]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        ).squeeze()[pad_left: ( -pad_right if pad_right > 0 else None)]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class UpSamplerBlock(nn.Module):
    """Transpose Conv cộng với các khối Resnet để upsample embedding đặc trưng."""
    def __init__(self, in_channels: int, upsample_factors: List[int], kernel_sizes: Optional[List[int]] = None):
        super().__init__()
        self.in_channels = in_channels
        self.upsample_factors = list(upsample_factors or [])
        self.kernel_sizes = list(kernel_sizes or [8] * len(self.upsample_factors))

        assert len(self.kernel_sizes) == len(self.upsample_factors), "kernel_sizes và upsample_factors phải có cùng độ dài"

        self.upsample_layers = nn.ModuleList()
        self.resnet_blocks  = nn.ModuleList()
        self.out_proj = nn.Linear(self.in_channels // (2 ** len(self.upsample_factors)), self.in_channels, bias=True)

        for i, (k, u) in enumerate(zip(self.kernel_sizes, self.upsample_factors)):
            c_in  = self.in_channels // (2 ** i)
            c_out = self.in_channels // (2 ** (i + 1))
            self.upsample_layers.append(
                weight_norm(nn.ConvTranspose1d(c_in, c_out, kernel_size=k, stride=u, padding=(k - u) // 2))
            )
            self.resnet_blocks.append(
                ResnetBlock(in_channels=c_out, out_channels=c_out, dropout=0.0, temb_channels=0)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> ... -> [B, C', L']
        for up, rsblk in zip(self.upsample_layers, self.resnet_blocks):
            x = rsblk(up(x))
        # [B, C', L'] -> [B, L', C] (Trở về hidden_dim ban đầu)
        return nonlinearity(self.out_proj(x.transpose(1, 2)))


class FourierHead(nn.Module):
    """Lớp cơ sở cho các module fourier ngược."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tham số:
            x (Tensor): Tensor đầu vào có dạng (B, L, H), trong đó B là kích thước batch,
                        L là độ dài chuỗi, và H biểu thị kích thước mô hình.

        Trả về:
            Tensor: Tín hiệu âm thanh miền thời gian được tái tạo có dạng (B, T), trong đó T là độ dài của tín hiệu đầu ra.
        """
        raise NotImplementedError("Các lớp con phải triển khai phương thức forward.")


class ISTFTHead(FourierHead):
    """
    Module ISTFT Head để dự đoán các hệ số phức STFT.

    Tham số:
        dim (int): Kích thước ẩn của mô hình.
        n_fft (int): Kích thước của biến đổi Fourier.
        hop_length (int): Khoảng cách giữa các khung cửa sổ trượt liền kề, nên căn chỉnh với
                          độ phân giải của các đặc trưng đầu vào.
        padding (str, optional): Loại padding. Các tùy chọn là "center" hoặc "same". Mặc định là "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lượt truyền xuôi của module ISTFTHead.

        Tham số:
            x (Tensor): Tensor đầu vào có dạng (B, L, H), trong đó B là kích thước batch,
                        L là độ dài chuỗi, và H biểu thị kích thước mô hình.

        Trả về:
            Tensor: Tín hiệu âm thanh miền thời gian được tái tạo có dạng (B, T), trong đó T là độ dài của tín hiệu đầu ra.
        """
        x_pred = self.out(x )
        # x_pred = x
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # bảo vệ để ngăn chặn độ lớn quá mức
        # việc wrapping xảy ra ở đây. Hai dòng này tạo ra giá trị thực và ảo
        x = torch.cos(p)
        y = torch.sin(p)
        # tính toán lại pha ở đây không tạo ra cái gì mới
        # chỉ tốn thời gian
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # tốt hơn là tạo ra giá trị phức trực tiếp
        with torch.autocast(device_type="cuda", enabled=False):
            S = mag.float() * (x.float() + 1j * y.float())
        # S = mag * (x + 1j * y)
        # S = S.to(mag.dtype)
        audio = self.istft(S)
        return audio.unsqueeze(1),x_pred


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        h_ = self.proj_out(h_)

        return x + h_

def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"tạo attention loại '{attn_type}' với {in_channels} kênh đầu vào")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)


class Backbone(nn.Module):
    """Lớp cơ sở cho backbone của bộ sinh (generator). Nó bảo toàn độ phân giải thời gian qua tất cả các lớp."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Tham số:
            x (Tensor): Tensor đầu vào có dạng (B, C, L), trong đó B là kích thước batch,
                        C biểu thị các đặc trưng đầu ra, và L là độ dài chuỗi.

        Trả về:
            Tensor: Đầu ra có dạng (B, L, H), trong đó B là kích thước batch, L là độ dài chuỗi,
                    và H biểu thị kích thước mô hình.
        """
        raise NotImplementedError("Các lớp con phải triển khai phương thức forward.")


class VocosBackbone(Backbone):
    """
    Vocos backbone được xây dựng với các khối ConvNeXt. Hỗ trợ điều kiện bổ sung với Chuẩn hóa Lớp Thích ứng (Adaptive Layer Normalization)

    Tham số:
        input_channels (int): Số lượng kênh đặc trưng đầu vào.
        dim (int): Kích thước ẩn của mô hình.
        intermediate_dim (int): Kích thước trung gian được sử dụng trong ConvNeXtBlock.
        num_layers (int): Số lượng lớp ConvNeXtBlock.
        layer_scale_init_value (float, optional): Giá trị khởi tạo cho việc scaling lớp. Mặc định là `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Số lượng embedding cho AdaLayerNorm.
                                                None nghĩa là mô hình không có điều kiện. Mặc định là None.
    """

    def __init__(
        self,  hidden_dim=1024,depth=12,heads=16,pos_meb_dim=64):
        super().__init__()

        self.embed = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)



        self.temb_ch = 0
        block_in = hidden_dim
        dropout = 0.1

        prior_net : tp.List[nn.Module] = [
            ResnetBlock(in_channels=block_in,out_channels=block_in,
                        temb_channels=self.temb_ch,dropout=dropout),
            ResnetBlock(in_channels=block_in,out_channels=block_in,
                        temb_channels=self.temb_ch,dropout=dropout),
        ]
        self.prior_net = nn.Sequential(*prior_net)

        depth = depth
        time_rotary_embed = RotaryPositionalEmbeddings(dim=pos_meb_dim)


        transformer_blocks = [
            TransformerBlock(dim=hidden_dim, n_heads=heads, rotary_embed=time_rotary_embed)
            for _ in range(depth)
        ]


        self.transformers = nn.Sequential(*transformer_blocks)
        self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        post_net : tp.List[nn.Module] = [
            ResnetBlock(in_channels=block_in,out_channels=block_in,
                        temb_channels=self.temb_ch,dropout=dropout),
            ResnetBlock(in_channels=block_in,out_channels=block_in,
                        temb_channels=self.temb_ch,dropout=dropout),
        ]
        self.post_net = nn.Sequential(*post_net)

    def forward(self, x: torch.Tensor ) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.embed(x)
        x = self.prior_net(x)
        x = x.transpose(1, 2)
        x= self.transformers(x)
        x = x.transpose(1, 2)
        x = self.post_net(x)
        x = x.transpose(1, 2)
        x = self.final_layer_norm(x)
        return x




def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class CodecDecoderVocos(nn.Module):
    def __init__(self,
                 hidden_dim=1024,
                 depth=12,
                 heads=16,
                 pos_meb_dim=64,
                 hop_length=320,
                 vq_num_quantizers=1,
                 vq_dim=2048, #1024 2048
                 vq_commit_weight=0.25,
                 vq_weight_init=False,
                 vq_full_commit_loss=False,
                 codebook_size=16384,
                 codebook_dim=16,
                 sample_rate: int = 16000,
                 upsample_factors: Optional[List[int]] = None,
                 upsample_kernel_sizes: Optional[List[int]] = None,
                ):
        super().__init__()
        self.hop_length = hop_length
        self.sample_rate = int(sample_rate)

        self.quantizer = ResidualFSQ(
            dim = vq_dim,
            levels = [4, 4, 4, 4, 4,4,4,4],
            num_quantizers = 1
        )

        # self.quantizer = ResidualVQ(
        #     num_quantizers=vq_num_quantizers,
        #     dim=vq_dim,
        #     codebook_size=codebook_size,
        #     codebook_dim=codebook_dim,
        #     threshold_ema_dead_code=2,
        #     commitment=vq_commit_weight,
        #     weight_init=vq_weight_init,
        #     full_commit_loss=vq_full_commit_loss,
        # )


        self.backbone = VocosBackbone( hidden_dim=hidden_dim,depth=depth,heads=heads,pos_meb_dim=pos_meb_dim)

        self.upsampler = None
        self._ups_total = 1
        if upsample_factors and len(upsample_factors) > 0:
            self.upsampler = UpSamplerBlock(in_channels=hidden_dim,
                                            upsample_factors=upsample_factors,
                                            kernel_sizes=upsample_kernel_sizes or [8]*len(upsample_factors))
            self._ups_total = int(np.prod(upsample_factors))

        # sanity check
        if (self.sample_rate % 50) != 0 or \
           ((self.sample_rate // 50) != (self.hop_length * self._ups_total)):
           raise ValueError(f"sample_rate {self.sample_rate}, hop_length {self.hop_length}, upsample_factors {upsample_factors} không khớp!")


        self.head = ISTFTHead(dim=hidden_dim, n_fft=self.hop_length*4, hop_length=self.hop_length, padding="same")

        self.reset_parameters()

    def forward(self, x, vq=True):
        if vq is True:
            # x, q, commit_loss = self.quantizer(x)
            x = x.permute(0, 2, 1)
            x, q = self.quantizer(x)
            x = x.permute(0, 2, 1)
            q = q.permute(0, 2, 1)
            return x, q, None
        x = self.backbone(x)
        if self.upsampler is not None:
            x = self.upsampler(x.transpose(1,2))
        x, xpred  = self.head(x)

        return x, xpred

    def vq2emb(self, vq):
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self):
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq):
        x = vq[None,:,:]
        x = self.model(x)
        return x

    def inference_0(self, x):
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None

    def inference(self, x):
        x = self.model(x)
        return x, None


    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)



class CodecDecoderVocos_transpose(nn.Module):
    def __init__(self,
                 hidden_dim=1024,
                 depth=12,
                 heads=16,
                 pos_meb_dim=64,
                 hop_length=320,
                 vq_num_quantizers=1,
                 vq_dim=1024, #1024 2048
                 vq_commit_weight=0.25,
                 vq_weight_init=False,
                 vq_full_commit_loss=False,
                 codebook_size=16384,
                 codebook_dim=16,
                ):
        super().__init__()
        self.hop_length = hop_length


        self.quantizer = ResidualVQ(
            num_quantizers=vq_num_quantizers,
            dim=vq_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=2,
            commitment=vq_commit_weight,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
        )


        self.backbone = VocosBackbone( hidden_dim=hidden_dim,depth=depth,heads=heads,pos_meb_dim=pos_meb_dim)

        self.inverse_mel_conv = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1  # Đảm bảo độ dài đầu ra khớp trước khi mã hóa
            ),
            nn.GELU(),
            nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1
            )
        )

        self.head = ISTFTHead(dim=hidden_dim, n_fft=self.hop_length*4, hop_length=self.hop_length, padding="same")

        self.reset_parameters()

    def forward(self, x, vq=True):
        if vq is True:
            x, q, commit_loss = self.quantizer(x)
            return x, q, commit_loss
        x = self.backbone(x)
        x,_  = self.head(x)

        return x ,_

    def vq2emb(self, vq):
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self):
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq):
        x = vq[None,:,:]
        x = self.model(x)
        return x

    def inference_0(self, x):
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None

    def inference(self, x):
        x = self.model(x)
        return x, None


    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)




def main():
    # Cài đặt thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # Khởi tạo mô hình
    model = CodecDecoderVocos_transpose().to(device)
    print("Mô hình đã khởi tạo.")

    # Tạo đầu vào kiểm tra: batch_size x in_channels x sequence_length
    batch_size = 2
    in_channels = 1024
    sequence_length = 50  #Độ dài mẫu, có thể điều chỉnh khi cần
    dummy_input = torch.randn(batch_size, in_channels, sequence_length).to(device)
    print(f"Kích thước đầu vào giả: {dummy_input.shape}")

    # Đặt mô hình ở chế độ đánh giá
    model.eval()

    # Truyền xuôi (Sử dụng VQ)
    # with torch.no_grad():
    #     try:
    #         output, q, commit_loss = model(dummy_input, vq=True)
    #         print("Forward pass with VQ:")
    #         print(f"Output shape: {output.shape}")
    #         print(f"Quantized codes shape: {q.shape}")
    #         print(f"Commitment loss: {commit_loss}")
    #     except Exception as e:
    #         print(f"Error during forward pass with VQ: {e}")

    # Truyền xuôi (Không sử dụng VQ)
    with torch.no_grad():
        # try:
        output_no_vq = model(dummy_input, vq=False)
        print("\nTruyền xuôi không có VQ:")
        print(f"Kích thước đầu ra: {output_no_vq.shape}")
        c=1
        # except Exception as e:
        #     print(f"Error during forward pass without VQ: {e}")


    # model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # model_size_mb = model_size_bytes / (1024 ** 2)
    # print(f"Model size: {model_size_bytes} bytes ({model_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()