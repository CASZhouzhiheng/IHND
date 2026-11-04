from __future__ import annotations

import ctypes as ct
from ctypes.util import find_library
import os
from pathlib import Path
import sys

import torch


class NativeStrength:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)
        self.context_tensor = torch.empty(1, device=self.device)
        torch.cuda.synchronize(self.device)
        self.directory_handles = []
        library_dir = Path(torch.__file__).parent / "lib"
        if os.name == "nt":
            self.directory_handles.append(os.add_dll_directory(str(library_dir)))
            candidates = sorted(library_dir.glob("nvrtc64_*.dll"))
            if os.environ.get("CUDA_PATH"):
                cuda_bin = Path(os.environ["CUDA_PATH"]) / "bin"
                self.directory_handles.append(os.add_dll_directory(str(cuda_bin)))
                candidates.extend(sorted(cuda_bin.glob("nvrtc64_*.dll")))
            if not candidates:
                raise RuntimeError("NVRTC library is unavailable; select the torch backend")
            nvrtc = ct.WinDLL(str(candidates[0]))
            self.driver = ct.WinDLL("nvcuda.dll")
        else:
            candidates = [str(p) for p in library_dir.glob("libnvrtc.so*")]
            for directory in sys.path:
                root = Path(directory) / "nvidia/cuda_nvrtc/lib"
                if root.is_dir():
                    candidates.extend(str(p) for p in sorted(root.glob("libnvrtc.so*")))
            found = find_library("nvrtc")
            if found:
                candidates.append(found)
            if not candidates:
                raise RuntimeError("NVRTC library is unavailable; select the torch backend")
            nvrtc = ct.CDLL(candidates[0])
            self.driver = ct.CDLL(find_library("cuda") or "libcuda.so.1")
        signatures = {
            "nvrtcCreateProgram": [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p],
            "nvrtcCompileProgram": [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)],
            "nvrtcGetProgramLogSize": [ct.c_void_p, ct.POINTER(ct.c_size_t)],
            "nvrtcGetProgramLog": [ct.c_void_p, ct.c_void_p],
            "nvrtcGetCUBINSize": [ct.c_void_p, ct.POINTER(ct.c_size_t)],
            "nvrtcGetCUBIN": [ct.c_void_p, ct.c_void_p],
            "nvrtcDestroyProgram": [ct.POINTER(ct.c_void_p)],
        }
        for name, arguments in signatures.items():
            getattr(nvrtc, name).argtypes = arguments
            getattr(nvrtc, name).restype = ct.c_int
        source = Path(__file__).with_name("strength.cu").read_bytes()
        program = ct.c_void_p()
        self.check(nvrtc.nvrtcCreateProgram(ct.byref(program), source, b"strength.cu", 0, None, None), "create program")
        try:
            major, minor = torch.cuda.get_device_capability(self.device)
            options = (ct.c_char_p*3)(f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++14", b"--fmad=false")
            status = nvrtc.nvrtcCompileProgram(program, 3, options)
            size = ct.c_size_t()
            self.check(nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size)), "log size")
            log = ct.create_string_buffer(size.value)
            self.check(nvrtc.nvrtcGetProgramLog(program, log), "compiler log")
            if status:
                raise RuntimeError(log.value.decode())
            self.check(nvrtc.nvrtcGetCUBINSize(program, ct.byref(size)), "binary size")
            binary = ct.create_string_buffer(size.value)
            self.check(nvrtc.nvrtcGetCUBIN(program, binary), "binary")
        finally:
            nvrtc.nvrtcDestroyProgram(ct.byref(program))
        self.driver.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
        self.driver.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
        self.driver.cuModuleUnload.argtypes = [ct.c_void_p]
        self.driver.cuLaunchKernel.argtypes = [ct.c_void_p]+[ct.c_uint]*7+[ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p]
        self.module, self.function = ct.c_void_p(), ct.c_void_p()
        self.check(self.driver.cuModuleLoadData(ct.byref(self.module), binary), "load kernel")
        self.check(self.driver.cuModuleGetFunction(ct.byref(self.function), self.module, b"strengths"), "get function")

    @staticmethod
    def check(code, operation):
        if code:
            raise RuntimeError(f"CUDA {operation} failed with status {code}")

    def __call__(self, q, edges, observed, output, counts, mode=0):
        batch, _, timepoints = q.shape
        arguments = [ct.c_void_p(v.data_ptr()) for v in (q, edges, observed, output, counts)]
        arguments += [ct.c_int(v) for v in (len(edges), timepoints, mode, batch)]
        pointers = (ct.c_void_p*len(arguments))(*[ct.cast(ct.byref(v), ct.c_void_p) for v in arguments])
        stream = ct.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
        self.check(self.driver.cuLaunchKernel(self.function, (len(edges)+7)//8, batch, 1, 256, 1, 1, 0, stream, pointers, None), "launch")

    def close(self):
        if getattr(self, "module", None):
            torch.cuda.synchronize(self.device)
            self.driver.cuModuleUnload(self.module)
            self.module = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
