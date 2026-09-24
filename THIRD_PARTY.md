# Third-party components

This project redistributes pinned source archives and build inputs solely to make the validated image reproducible.

| Component | Revision/version | Source | License |
|---|---|---|---|
| ComfyUI | `d158420950922dde49dad1fa87fa549735fd5137` | https://github.com/comfyanonymous/ComfyUI | GPL-3.0 |
| comfy-kitchen | `0.2.35`, `b2a2972ac68c395bbda8ad9030e8ae1089287815` | https://github.com/Comfy-Org/comfy-kitchen | Apache-2.0 |
| nanobind | `3.1.0` wheel | https://github.com/wjakob/nanobind | BSD-3-Clause |
| CMake | `4.4.2` wheel | https://cmake.org | BSD-3-Clause |
| ROCm PyTorch image | ROCm 7.2.4 / PyTorch 2.10.0 | https://hub.docker.com/r/rocm/pytorch | Component-specific upstream licenses |
| Qwen-Image-2.1 weights | ModelScope revision `5dc8a835...` | https://modelscope.cn/models/Comfy-Org/Qwen-Image-2.1 | See model repository terms |

The vendored ComfyUI and comfy-kitchen archives include their upstream license and notice files. The two files in `kernels/` retain their upstream SPDX copyright and Apache-2.0 identifiers.
