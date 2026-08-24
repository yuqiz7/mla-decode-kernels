"""JIT build of the gqa_decode_v0 extension via torch.utils.cpp_extension.load."""

import os

from torch.utils.cpp_extension import load

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")
os.makedirs(_BUILD_DIR, exist_ok=True)

_ext = load(
    name="gqa_decode_v0_ext",
    sources=[
        os.path.join(_REPO_ROOT, "kernels", "gqa_decode", "v0_naive.cu"),
        os.path.join(_REPO_ROOT, "python", "gqa_decode_ext.cpp"),
    ],
    build_directory=_BUILD_DIR,
    extra_include_paths=[os.path.join(_REPO_ROOT, "include")],
    extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a"],
    verbose=False,
)

gqa_decode_v0 = _ext.gqa_decode_v0
