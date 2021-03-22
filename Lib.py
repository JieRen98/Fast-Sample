import ctypes
import numpy as np


class GPU_Sample_Lib:
    def __init__(self, so_path="./kernel.so", seed=0):
        dll = ctypes.CDLL(so_path)
        init = dll.init
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_void_p

        init_substorage = dll.init_substorage
        init_substorage.argtypes = [ctypes.c_size_t]
        init_substorage.restype = ctypes.c_void_p
        self.init_substorage = init_substorage

        sample_exponential = dll.sample_exponential
        sample_exponential.argtypes = [ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self._sample_exponential = sample_exponential

        sample_normal = dll.sample_normal
        sample_normal.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._sample_normal = sample_normal

        sample_uniform = dll.sample_uniform
        sample_uniform.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._sample_uniform = sample_uniform

        sample_gamma = dll.sample_gamma
        sample_gamma.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self._sample_gamma = sample_gamma

        sample_multinomial = dll.sample_multinomial
        sample_multinomial.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        self._sample_multinomial = sample_multinomial

        to_cpu = dll.to_cpu
        to_cpu.argtypes = [ctypes.c_void_p]
        to_cpu.restype = ctypes.POINTER(ctypes.c_float)
        self._to_cpu = to_cpu

        to_cpu_num = dll.to_cpu_num
        to_cpu_num.argtypes = [ctypes.c_void_p]
        to_cpu_num.restype = ctypes.c_size_t
        self._to_cpu_num = to_cpu_num

        sync = dll.sync
        self.sync = sync

        self.status = init(seed)

    def get_substorage(self, nElm):
        return self.init_substorage(nElm)

    def sample_exponential(self, para_lambda, sst):
        self._sample_exponential(para_lambda, sst, self.status, 0)

    def sample_uniform(self, sst):
        self._sample_uniform(sst, self.status)

    def sample_gamma(self, shape, scale, sst):
        self._sample_gamma(shape, scale, sst, self.status, 0)

    def sample_normal(self, sst):
        self._sample_normal(sst, self.status)

    def sample_multinomial(self, param, rep, sst):
        param = np.ascontiguousarray(param, dtype=np.float32)
        num = len(param)
        param_ptr = ctypes.cast(param.ctypes.data, ctypes.POINTER(ctypes.c_float))
        self._sample_multinomial(param_ptr, num, rep, sst, self.status)

    def to_cpu(self, sst):
        nElm = self._to_cpu_num(sst)
        return np.ctypeslib.as_array(self._to_cpu(sst), (nElm,))