# Encoder Comparison Report: ONNX vs PyTorch

## Summary

**CRITICAL FINDING**: The ONNX encoder produces significantly different outputs compared to the PyTorch encoder due to an SDPA monkeypatch applied during export.

## Test Results

### Comparison 1: ONNX vs T5GemmaVoice PyTorch Encoder (Same Model)

| Text | Tokens | PyTorch mean | PyTorch std | ONNX mean | ONNX std | Cosine Sim | Status |
|------|--------|--------------|-------------|-----------|----------|------------|--------|
| こんにちは | 2 | -0.002 | 0.255 | -0.014 | 0.365 | **0.59** | ❌ DIFFER |
| こんにちは、今日は | 4 | 0.001 | 0.281 | -0.014 | 0.412 | **0.54** | ❌ DIFFER |
| Hello world | 3 | -0.002 | 0.279 | -0.016 | 0.482 | **0.53** | ❌ DIFFER |

### Key Observations

1. **Cosine Similarity ~0.53-0.59** - The outputs are only about 55% similar
2. **Standard Deviation Mismatch**:
   - PyTorch: std ≈ 0.25-0.29
   - ONNX: std ≈ 0.36-0.48 (43-65% higher)
3. **Max Difference**: Up to 8.5 in individual values

## Root Cause Analysis

### The SDPA Monkeypatch Problem

The ONNX export script (`export_onnx_encoder.py`) applies a monkeypatch to `transformers.masking_utils.sdpa_mask`:

```python
def _custom_no_vmap_sdpa_mask(...):
    # Custom implementation that changes attention behavior
```

**This monkeypatch fundamentally changes how the encoder computes attention!**

#### Proof: Before vs After Monkeypatch (Same PyTorch Model)

| Metric | WITHOUT Monkeypatch | WITH Monkeypatch |
|--------|---------------------|------------------|
| Mean | -0.002 | -0.014 |
| Std | 0.255 | 0.365 |
| First 5 values | [-0.17, -0.13, 0.14, -0.19, 0.31] | [-0.81, -0.23, 0.66, -0.04, 0.53] |
| Cosine Sim | 1.0 (baseline) | **0.59** |

The ONNX encoder output **matches the monkeypatched PyTorch output** (cosine sim ~0.99 between ONNX and monkeypatched PyTorch), confirming the monkeypatch is baked into the ONNX model.

## Impact on TTS Quality

Since the encoder produces different hidden states:
1. The decoder receives incorrect conditioning information
2. Audio quality is degraded
3. Pronunciation and prosody may be affected
4. The hybrid model cannot match 4bit PyTorch model quality

## Recommendations

### Option 1: Use PyTorch Encoder (Recommended)

Modify `inference_hybrid_complete.py` to use PyTorch encoder instead of ONNX:

```python
# Instead of ONNX encoder
# self.encoder_session = ort.InferenceSession(...)

# Use PyTorch encoder from the model
self.encoder = model.backbone.model.encoder  # or model.model.encoder
```

**Pros**: Correct output, matches 4bit model
**Cons**: Requires GPU memory for encoder

### Option 2: Re-export ONNX Without Monkeypatch

The T5Gemma model uses `vmap`-based attention masking which is incompatible with standard ONNX tracing. Re-exporting would require:
- Rewriting the attention mechanism
- Using a different export method (e.g., torch.export with dynamo)

**Pros**: ONNX for deployment
**Cons**: Complex, may not be feasible

### Option 3: Accept Quality Degradation

If ONNX is strictly required and quality loss is acceptable.

**Pros**: Simple
**Cons**: ~45% degradation in encoder output quality

## Conclusion

The ONNX encoder file (`onnx_models_fp16/encoder.onnx`) is fundamentally broken due to the SDPA monkeypatch applied during export. **Do not use it for production TTS**.

For correct audio quality matching the 4bit model, use the PyTorch encoder.
