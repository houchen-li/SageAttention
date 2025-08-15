import subprocess
from typing import Dict, List
from setuptools import setup, find_packages, Extension

import torch, torch_musa
from torch_musa.utils.simple_porting import SimplePorting
from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension, MUSA_HOME

CXX_FLAGS: List[str] = [
    "-O3",
    "-DNDEBUG",
    "-fopenmp",
    "-lgomp",
    "-std=c++17",
    "-DENABLE_BF16",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-Wno-switch-bool",
]

MCC_FLAGS: List[str] = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-std=c++17",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-fno-strict-aliasing",
    "-fmusa-flush-denormals-to-zero",
    "-mllvm",
    "-mtgpu-enable-max-ilp-scheduling-strategy=0",
    "-mllvm",
    "-mtgpu-enchanced-minreg-schedule=1",
    "-mllvm",
    "-mtgpu-enable-cse=0",
]

MAPPING_RULE: Dict[str, str] = {
    "#include <ATen/cuda/CUDAContext.h>": '#include "torch_musa/csrc/aten/musa/MUSAContext.h"',
    "cuda/": "musa/",
    ".cuh": ".muh",
    "<cuda_fp8.h>": "<musa_fp8.h>",
    "nv_bfloat16": "mt_bfloat16",
    ".load_128b_async": ".template load_128b_async",
    ".advance_offset": ".template advance_offset",
}


def main() -> int:
    SimplePorting(cuda_dir_path="csrc", mapping_rule=MAPPING_RULE).run()
    # subprocess.run(["git", "apply", "musa.patch"], check=True, text=True)

    qattn_extention: Extension = MUSAExtension(
        name="sageattention._qattn_mp31",
        sources=[
            "csrc_musa/qattn/pybind_sm80.cpp",
            "csrc_musa/qattn/qk_int_sv_f16_cuda_sm80.mu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "mcc": MCC_FLAGS,
        },
    )
    fused_extension: Extension = MUSAExtension(
        name="sageattention._fused",
        sources=["csrc_musa/fused/pybind.cpp", "csrc_musa/fused/fused.mu"],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "mcc": MCC_FLAGS,
        },
    )

    ext_modules: List[Extension] = [qattn_extention, fused_extension]

    setup(
        name="sageattention",
        version="2.2.0",
        author="SageAttention team",
        license="Apache 2.0 License",
        description="Accurate and efficient plug-and-play low-bit attention.",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/thu-ml/SageAttention",
        packages=find_packages(),
        python_requires=">=3.9",
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
    )

    return 0


if __name__ == "__main__":
    main()
