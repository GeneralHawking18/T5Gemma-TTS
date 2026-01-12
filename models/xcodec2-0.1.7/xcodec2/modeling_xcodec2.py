import torch
import torch.nn as nn
from transformers import PreTrainedModel
if __package__ or "." in __name__:
    from .configuration_bigcodec import BigCodecConfig
    from .vq.codec_encoder import CodecEncoder_Transformer
    from .vq.codec_decoder_vocos import CodecDecoderVocos
    from .vq.module import SemanticEncoder
else:
    from configuration_bigcodec import BigCodecConfig
    from vq.codec_encoder import CodecEncoder_Transformer
    from vq.codec_decoder_vocos import CodecDecoderVocos
    from vq.module import SemanticEncoder
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import torch.nn.functional as F

class XCodec2Model(PreTrainedModel):
	config_class = BigCodecConfig

	def __init__(self, config: BigCodecConfig):
		super().__init__(config)

		# 1) Mô hình ngữ nghĩa (Semantic Model)
		# Hiểu nôm na là phần "bộ não" hiểu ý nghĩa của âm thanh
		self.semantic_model = Wav2Vec2BertModel.from_pretrained(
			"facebook/w2v-bert-2.0",
			output_hidden_states=True
		)
		self.semantic_model.eval()

		self.SemanticEncoder_module = SemanticEncoder(
			 config.semantic_hidden_size,
			 config.semantic_hidden_size,
			 config.semantic_hidden_size
		)

		# 2) Bộ mã hóa Codec (Codec Encoder)
		# Giống như việc nén file nhạc lại cho gọn
		self.CodecEnc = CodecEncoder_Transformer()

		# 3) Bộ giải mã Codec (Codec Decoder)
		# Giống như máy phát nhạc, bung file nén ra thành âm thanh
		self.generator = CodecDecoderVocos(
			hidden_dim=config.codec_decoder_hidden_size,
			hop_length=config.hop_length,
			sample_rate=config.sample_rate,
			upsample_factors=config.upsample_factors,
			upsample_kernel_sizes=config.upsample_kernel_sizes,
		)

		# 4) Hai lớp kết nối đầy đủ (Fully Connected Layers - cầu nối dữ liệu)
		self.fc_prior = nn.Linear(2048, 2048)
		self.fc_post_a = nn.Linear(2048, 1024)
		feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
		self.feature_extractor = feature_extractor

	def forward(self, input_waveform, sample_rate=16000):
		"""
		Hàm forward ở đây không nhất thiết phải gọi là forward, có thể tách ra thành phương thức khác;
		nhưng nếu muốn tương thích với pipeline của HuggingFace, bạn cần đặt logic cốt lõi vào trong forward.

		Tham số:
		  input_waveform: [batch_size, waveform_length] (Dữ liệu sóng âm đầu vào)
		  sample_rate: mặc định 16000
		Trả về:
		  Âm thanh giọng nói được tái tạo (Tensor)
		"""
		# 1) Trích xuất đặc trưng (Feature Extraction)
		# Nếu cần đệm thêm (padding) để đủ kích thước thì làm ở đây
		wav = input_waveform
		pad_for_wav = (320 - (wav.shape[1] % 320))

		wav = torch.nn.functional.pad(wav, (0, pad_for_wav))

		input_features = self.feature_extractor(
			F.pad(wav[0,:].cpu(), (160, 160)),
			sampling_rate=sample_rate,
			return_tensors="pt"
		).input_features.to(self.device)  # [batch, frames, feat_dim]

		# 2) Lớp ngữ nghĩa (Semantic Layer)
		semantic_output = self.semantic_model(input_features)
		semantic_hidden_16 = semantic_output.hidden_states[16]  # Lấy lớp thứ 16
		semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)  # [batch, hidden_dim, frames]
		semantic_encoded = self.SemanticEncoder_module(semantic_hidden_16)

		# 3) Bộ mã hóa Codec (Codec Encoder)
		wav = wav.to(self.device)  # shape: [batch, 1, time]
		vq_emb = self.CodecEnc(wav.unsqueeze(1))  # [batch, time//down, 1024] đây chỉ là ví dụ
		vq_emb = vq_emb.transpose(1, 2)  # -> [batch, 1024, frames]

		# 4) Ghép nối (Concatenation)
		# Trộn thông tin ngữ nghĩa và thông tin âm học lại với nhau
		concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)  # [batch, 1024 + 1024, frames]

		# 5) fc_prior (Lớp tuyến tính trước khi vào decoder)
		concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

		# 6) Phần lượng tử hóa của bộ giải mã (Decoder Quantization)
		_, vq_code, _ = self.generator(concat_emb, vq=True)
		vq_post_emb = self.generator.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
		vq_post_emb = vq_post_emb.transpose(1, 2)

		# 7) fc_post_a (Lớp tuyến tính sau khi lượng tử hóa)
		vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)

		# 8) Cuối cùng giải mã thành dạng sóng (Waveform Decoding)
		recon_audio = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0]
		# recon_audio: [batch, time]
		return recon_audio

	def encode_code(self, input_waveform, sample_rate=16000):
		"""
		Mã hóa âm thanh đầu vào thành biểu diễn mã (code representation).

		Tham số:
		  input_waveform: [batch_size, waveform_length]
		  sample_rate: mặc định 16000
		Trả về:
		  Mã đã được mã hóa (Tensor - vq_code)
		"""
		with torch.no_grad():
			wav = input_waveform
			pad_for_wav = (320 - (wav.shape[1] % 320))

			wav = torch.nn.functional.pad(wav, (0, pad_for_wav))

			input_features = self.feature_extractor(
				F.pad(wav[0,:].cpu(), (160, 160)),
				sampling_rate=sample_rate,
				return_tensors="pt"
			).input_features.to(self.device)  # [batch, frames, feat_dim]

			# 2) Lớp ngữ nghĩa
			semantic_output = self.semantic_model(input_features)
			semantic_hidden_16 = semantic_output.hidden_states[16]  # Lấy lớp thứ 16
			semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)  # [batch, hidden_dim, frames]
			semantic_encoded = self.SemanticEncoder_module(semantic_hidden_16)

			# 3) Bộ mã hóa Codec
			wav = wav.to(self.device)  # shape: [batch, 1, time]
			vq_emb = self.CodecEnc(wav.unsqueeze(1))  # [batch, time//down, 1024]
			vq_emb = vq_emb.transpose(1, 2)  # -> [batch, 1024, frames]

			# 4) Ghép nối
			concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)  # [batch, 2048, frames]

			# 5) fc_prior
			concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

			# 6) Phần lượng tử hóa của decoder, lấy ra code
			_, vq_code, _ = self.generator(concat_emb, vq=True)
			# vq_code: [batch, frames]
			return vq_code

	def decode_code(self, vq_code):
		"""
		Giải mã các mã (code) đã được mã hóa trở lại thành âm thanh.

		Tham số:
		  vq_code: Mã đã được mã hóa (Tensor) [batch, frames]
		Trả về:
		  Âm thanh sau giải mã (Tensor) [batch, waveform_length]
		"""
		with torch.no_grad():
			# Lấy các embedding đã được lượng tử hóa từ indices
			vq_post_emb = self.generator.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
			vq_post_emb = vq_post_emb.transpose(1, 2)  # [batch, 1024, frames]

			# 7) fc_post_a
			vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)  # [batch, 1024, frames]

			# 8) Cuối cùng giải mã thành dạng sóng
			recon_audio = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0]  # [batch, time]
			return recon_audio

	def encode_batch_feats(self, input_waveform, input_features):
		"""
		Mã hóa âm thanh đầu vào thành biểu diễn mã (phiên bản xử lý theo batch features).

		Tham số:
		  input_waveform: [batch_size, 1, waveform_length]
		  input_features: Các đặc trưng đầu vào
		Trả về:
		  Mã đã được mã hóa (Tensor)
		"""
		with torch.no_grad():
			# 2) Lớp ngữ nghĩa
			semantic_output = self.semantic_model(input_features[:,0,:,:])
			semantic_hidden_16 = semantic_output.hidden_states[16]  # Lấy lớp thứ 16
			semantic_hidden_16 = semantic_hidden_16.transpose(1, 2)  # [batch, hidden_dim, frames]
			semantic_encoded = self.SemanticEncoder_module(semantic_hidden_16)

			# 3) Bộ mã hóa Codec
			wav = input_waveform  # .unsqueeze(1).to(self.device)  # shape: [batch, 1, time]
			vq_emb = self.CodecEnc(wav)  # [batch, time//down, 1024]
			vq_emb = vq_emb.transpose(1, 2)  # -> [batch, 1024, frames]

			# 4) Ghép nối
			concat_emb = torch.cat([semantic_encoded, vq_emb], dim=1)  # [batch, 2048, frames]

			# 5) fc_prior
			concat_emb = self.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

			# 6) Phần lượng tử hóa của decoder, lấy ra code
			_, vq_code, _ = self.generator(concat_emb, vq=True)
			# vq_code: [batch, frames]
			return vq_code
