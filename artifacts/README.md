# Validated binary

`comfy_kitchen_gfx1100.so` is the exact comfy-kitchen `0.2.35` HIP extension used by the persisted W7900D result.

- SHA-256: `0328285d26d04786c0e72a463db5952bf8c9f9aab4544d88d71ce595fc45560d`
- Target: x86-64, Python abi3, ROCm/HIP 7.2.4, `gfx1100`
- Contains: accepted INT8 attention and fused RMSNorm/RoPE kernels

Do not install this binary into a different ROCm, comfy-kitchen, Python ABI, or GPU architecture. Use `scripts/rebuild-extension.sh` and rerun the frozen micro oracles instead.
