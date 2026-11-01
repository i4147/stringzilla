import os
import sys
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from typing import List, Tuple, Final
import subprocess
import concurrent.futures
import threading
import time
CPP_MEMORY_PER_WORKER_GB: Final[float] = 2.0
CUDA_MEMORY_PER_WORKER_GB: Final[float] = 4.0
def _memory_available_and_total_gb():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            fields = {line.split(":", 1)[0]: line.split()[1] for line in handle if ":" in line}
        return int(fields["MemAvailable"]) / 1024**2, int(fields["MemTotal"]) / 1024**2
    except (OSError, KeyError, ValueError, IndexError):
        return None
def _max_compile_workers(memory_per_worker_gb: float = CPP_MEMORY_PER_WORKER_GB) -> int:
    requested = os.environ.get("SZ_MAX_COMPILE_WORKERS", "")
    if requested.isdigit() and int(requested) > 0:
        return int(requested)
    workers = min(os.cpu_count() or 1, 8)
    memory = _memory_available_and_total_gb()
    if memory is not None:
        workers = min(workers, int(memory[0] // memory_per_worker_gb))
    return max(1, workers)
def _log_build_event(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
def _memory_status() -> str:
    memory = _memory_available_and_total_gb()
    return "unknown" if memory is None else f"{memory[0]:.1f} of {memory[1]:.1f} GB"
def _start_memory_sampler(interval_seconds: float = 10.0):
    stop_event = threading.Event()
    def _sample() -> None:
        while not stop_event.wait(interval_seconds):
            _log_build_event(f"memory {_memory_status()}")
    threading.Thread(target=_sample, daemon=True).start()
    return stop_event.set
def _depfile_prerequisites(dep_path: str):
    try:
        with open(dep_path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    text = text.replace("\\\n", " ")
    if ":" in text:
        text = text.split(":", 1)[1]
    return [token for token in text.split() if token]
def _object_is_fresh(obj_path: str, dep_path: str) -> bool:
    if not os.path.exists(obj_path):
        return False
    prerequisites = _depfile_prerequisites(dep_path)
    if not prerequisites:
        return False
    object_mtime = os.path.getmtime(obj_path)
    for prerequisite in prerequisites:
        try:
            if os.path.getmtime(prerequisite) > object_mtime:
                return False
        except OSError:
            return False
    return True
def _run_compilations_in_parallel(jobs, max_workers: int) -> None:
    if not jobs:
        return
    workers = max(1, min(max_workers, len(jobs)))
    _log_build_event(f"compiling {len(jobs)} source(s) with {workers} worker(s), memory {_memory_status()}")
    def _run_and_log(job, label):
        _log_build_event(f"start {label}")
        started_at = time.monotonic()
        job()
        _log_build_event(f"done  {label} in {time.monotonic() - started_at:.1f}s, memory {_memory_status()}")
    stop_sampler = _start_memory_sampler()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_and_log, job, label): label for job, label in jobs}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    raise RuntimeError(f"Compilation failed: {futures[future]}") from error
    finally:
        stop_sampler()
_CUDA_ARCHES: Final = ["80-real", "90-real", "90-virtual"]
_KEPLER_ARCHES: Final = ["80-real", "80-virtual"]
_HOPPER_ARCHES: Final = ["90-real", "90-virtual"]
def _cuda_gencode_flags(cuda_source: str) -> List[str]:
    stem = os.path.splitext(os.path.basename(cuda_source))[0]
    if stem.endswith("_hopper"):
        arches = _HOPPER_ARCHES
    elif stem.endswith("_kepler"):
        arches = _KEPLER_ARCHES
    else:
        arches = _CUDA_ARCHES
    flags = []
    for arch in arches:
        number, kind = arch.split("-")
        code = f"sm_{number}" if kind == "real" else f"compute_{number}"
        flags += ["-gencode", f"arch=compute_{number},code={code}"]
    return flags
def _parallel_compiler_compile(
    self,
    sources,
    output_dir=None,
    macros=None,
    include_dirs=None,
    debug=0,
    extra_preargs=None,
    extra_postargs=None,
    depends=None,
):
    is_msvc = getattr(self, "compiler_type", "") == "msvc"
    if is_msvc and not self.initialized:
        self.initialize()
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs
    )
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    use_depfiles = not is_msvc
    force = getattr(self, "_sz_force", False)
    if is_msvc:
        msvc_compile_opts = list(extra_preargs or [])
        msvc_compile_opts.append("/c")
        if debug:
            msvc_compile_opts.extend(self.compile_options_debug)
        else:
            msvc_compile_opts.extend(self.compile_options)
    def _compile_one(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        if is_msvc:
            if ext in self._c_extensions:
                input_opt = "/Tc" + src
            elif ext in self._cpp_extensions:
                input_opt = "/Tp" + src
            else:
                return
            args = [self.cc] + msvc_compile_opts + pp_opts
            if ext in self._cpp_extensions:
                args.append("/EHsc")
            args.extend([input_opt, "/Fo" + obj])
            args.extend(extra_postargs)
            self.spawn(args)
        elif use_depfiles:
            dep_path = obj + ".d"
            if not force and _object_is_fresh(obj, dep_path):
                return
            self._compile(obj, src, ext, cc_args, extra_postargs + ["-MMD", "-MF", dep_path], pp_opts)
        else:
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    def _compile_one_logged(obj):
        label = build[obj][0] if obj in build else obj
        if use_depfiles and not force and _object_is_fresh(obj, obj + ".d"):
            _log_build_event(f"skip  {label} (object up to date)")
            return
        _log_build_event(f"start {label}")
        started_at = time.monotonic()
        _compile_one(obj)
        _log_build_event(f"done  {label} in {time.monotonic() - started_at:.1f}s, memory {_memory_status()}")
    workers = max(1, min(_max_compile_workers(), len(objects)))
    _log_build_event(f"compiling {len(objects)} source(s) with {workers} worker(s), memory {_memory_status()}")
    stop_sampler = _start_memory_sampler()
    try:
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for _ in pool.map(_compile_one_logged, objects):
                    pass
        else:
            for obj in objects:
                _compile_one_logged(obj)
    finally:
        stop_sampler()
    return objects
class ParallelBuildExt(build_ext):
    def build_extension(self, ext):
        import types
        if self.compiler is not None and not getattr(self.compiler, "_sz_parallelized", False):
            self.compiler._sz_force = bool(self.force)
            self.compiler.compile = types.MethodType(_parallel_compiler_compile, self.compiler)
            self.compiler._sz_parallelized = True
        if sys.platform == "linux" or sys.platform.startswith("freebsd"):
            c_compile_args, _, _ = linux_settings(use_cpp=False)
            cpp_compile_args, _, _ = linux_settings(use_cpp=True)
        elif sys.platform == "darwin":
            c_compile_args, _, _ = darwin_settings(use_cpp=False)
            cpp_compile_args, _, _ = darwin_settings(use_cpp=True)
        elif sys.platform == "win32":
            c_compile_args, _, _ = windows_settings(use_cpp=False)
            cpp_compile_args, _, _ = windows_settings(use_cpp=True)
        else:
            c_compile_args, cpp_compile_args = [], []
        sources = list(ext.sources or [])
        c_sources = [s for s in sources if s.endswith(".c")]
        cpp_sources = [s for s in sources if s.endswith((".cc", ".cpp", ".cxx"))]
        objects: List[str] = []
        if c_sources:
            objects += self.compiler.compile(
                c_sources,
                output_dir=self.build_temp,
                macros=ext.define_macros,
                include_dirs=ext.include_dirs,
                debug=self.debug,
                extra_postargs=c_compile_args,
                depends=ext.depends,
            )
        if cpp_sources:
            objects += self.compiler.compile(
                cpp_sources,
                output_dir=self.build_temp,
                macros=ext.define_macros,
                include_dirs=ext.include_dirs,
                debug=self.debug,
                extra_postargs=cpp_compile_args,
                depends=ext.depends,
            )
        if getattr(ext, "extra_objects", None):
            objects += list(ext.extra_objects)
        self.compiler.link_shared_object(
            objects,
            self.get_ext_fullpath(ext.name),
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=getattr(ext, "runtime_library_dirs", None),
            extra_postargs=ext.extra_link_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
            target_lang=self.compiler.detect_language(ext.sources),
        )
class NumpyBuildExt(ParallelBuildExt):
    def build_extension(self, ext):
        import numpy as np
        numpy_include = np.get_include()
        if numpy_include not in ext.include_dirs:
            ext.include_dirs.append(numpy_include)
        super().build_extension(ext)
class CudaBuildExtension(NumpyBuildExt):
    def build_extension(self, ext):
        if any(source.endswith(".cu") for source in ext.sources or []):
            self._build_cuda_extension(ext)
        super().build_extension(ext)
    def _build_cuda_extension(self, ext):
        cuda_sources = [s for s in ext.sources if s.endswith(".cu")]
        c_sources = [s for s in ext.sources if not s.endswith(".cu")]
        os.makedirs(self.build_temp, exist_ok=True)
        host_cxx = os.environ.get("CUDAHOSTCXX")
        is_msvc = getattr(self.compiler, "compiler_type", "") == "msvc"
        if is_msvc:
            runtime_library_flag = "/MDd" if self.debug else "/MD"
            host_compiler_flags = ["/Zc:preprocessor", "/Zc:__cplusplus", "/utf-8", "/Oi-", runtime_library_flag]
        else:
            host_compiler_flags = [
                "-fPIC",
                "-fno-builtin-memcmp",
                "-fno-builtin-memchr",
                "-fno-builtin-memcpy",
                "-fno-builtin-memset",
            ]
        use_depfiles = not is_msvc
        objects = []
        nvcc_jobs = []
        for cuda_source in cuda_sources:
            obj_name = os.path.splitext(os.path.basename(cuda_source))[0] + ".o"
            obj_path = os.path.join(self.build_temp, obj_name)
            dep_path = obj_path + ".d"
            objects.append(obj_path)
            nvcc_cmd = [
                "nvcc",
                "-c",
                cuda_source,
                "-o",
                obj_path,
                "-std=c++20",
                "-O2",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                *_cuda_gencode_flags(cuda_source),
                "-Xfatbin=--compress-all",
                "-DSZ_DYNAMIC_DISPATCH=1",
                "-DSZ_USE_CUDA=1",
            ]
            if use_depfiles:
                nvcc_cmd.extend(["-MMD", "-MF", dep_path])
            if host_cxx:
                nvcc_cmd.extend(["-ccbin", host_cxx])
            for inc_dir in ext.include_dirs:
                nvcc_cmd.extend(["-I", inc_dir])
            for define in ext.define_macros:
                if len(define) == 2:
                    nvcc_cmd.append(f"-D{define[0]}={define[1]}")
                else:
                    nvcc_cmd.append(f"-D{define[0]}")
            for flag in host_compiler_flags:
                nvcc_cmd.extend(["-Xcompiler", flag])
            if use_depfiles and not self.force and _object_is_fresh(obj_path, dep_path):
                _log_build_event(f"skip  {cuda_source} (object up to date)")
                continue
            nvcc_jobs.append((lambda command=nvcc_cmd: subprocess.check_call(command), cuda_source))
        _run_compilations_in_parallel(nvcc_jobs, _max_compile_workers(CUDA_MEMORY_PER_WORKER_GB))
        ext.sources = c_sources
        ext.extra_objects = getattr(ext, "extra_objects", []) + objects
def sz_target_name() -> str:
    val = os.environ.get("SZ_TARGET")
    if val:
        return val
    try:
        with open("SZ_TARGET.env", "r", encoding="utf-8") as f:
            v = f.read().strip()
            if v:
                return v
    except FileNotFoundError:
        pass
    return "stringzilla"
sz_target: Final[str] = sz_target_name()
def get_compiler() -> str:
    if platform.python_implementation() == "CPython":
        compiler = platform.python_compiler().lower()
        return "gcc" if "gcc" in compiler else "llvm" if "clang" in compiler else ""
    return ""
def is_64bit_x86() -> bool:
    override = os.environ.get("SZ_IS_64BIT_X86_") if "SZ_IS_64BIT_X86_" in os.environ else None
    if override is not None:
        if override == "0":
            return False
        elif override == "1":
            return True
        else:
            raise ValueError("Invalid value for SZ_IS_64BIT_X86_: must be '0' or '1'")
    arch = platform.machine().lower()
    return (arch in ("x86_64", "x64", "amd64")) and (sys.maxsize > 2**32)
def is_64bit_arm() -> bool:
    override = os.environ.get("SZ_IS_64BIT_ARM_") if "SZ_IS_64BIT_ARM_" in os.environ else None
    if override is not None:
        if override == "0":
            return False
        elif override == "1":
            return True
        else:
            raise ValueError("Invalid value for SZ_IS_64BIT_ARM_: must be '0' or '1'")
    arch = platform.machine().lower()
    return (arch in ("arm64", "aarch64")) and (sys.maxsize > 2**32)
def is_64bit_riscv() -> bool:
    arch = platform.machine().lower()
    return arch.startswith("riscv64") and (sys.maxsize > 2**32)
def is_big_endian() -> bool:
    return sys.byteorder == "big"
def linux_settings(use_cpp: bool = False) -> Tuple[List[str], List[str], List[Tuple[str]]]:
    compile_args = [
        "-std=c++17" if use_cpp else "-std=c99",
        "-D_GNU_SOURCE",
        "-O2",
        "-fdiagnostics-color=always",
        "-Wno-unknown-pragmas",
        "-Wno-unused-function",
        "-fPIC",
        "-g",
    ]
    if not use_cpp:
        compile_args += [
            "-Wno-incompatible-pointer-types",
            "-Wno-discarded-qualifiers",
        ]
    link_args = [
        "-fPIC",
    ]
    macros_args = [
        ("SZ_IS_BIG_ENDIAN_", "1" if is_big_endian() else "0"),
        ("SZ_IS_64BIT_X86_", "1" if is_64bit_x86() else "0"),
        ("SZ_IS_64BIT_ARM_", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_WESTMERE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_GOLDMONT", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_HASWELL", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_ICELAKE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_NEON", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEONAES", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEONSHA", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE2", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE2AES", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_RVV", os.environ.get("SZ_USE_RVV") or ("1" if is_64bit_riscv() else "0")),
    ]
    return compile_args, link_args, macros_args
def darwin_settings(use_cpp: bool = False) -> Tuple[List[str], List[str], List[Tuple[str]]]:
    min_macos = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0")
    if is_64bit_arm():
        current_arch_flags = ["-arch", "arm64"]
    elif is_64bit_x86():
        current_arch_flags = ["-arch", "x86_64"]
    else:
        current_arch_flags = []
    compile_args = [
        "-std=c++17" if use_cpp else "-std=c99",
        "-O2",
        "-fcolor-diagnostics",
        "-Wno-unknown-pragmas",
        "-fPIC",
        f"-mmacosx-version-min={min_macos}",
        *current_arch_flags,
    ]
    if not use_cpp:
        compile_args += [
            "-Wno-incompatible-function-pointer-types",
            "-Wno-incompatible-pointer-types",
            "-Wno-ignored-qualifiers",
        ]
    link_args = [
        "-fPIC",
        *current_arch_flags,
    ]
    macros_args = [
        ("SZ_IS_64BIT_X86_", "1" if is_64bit_x86() else "0"),
        ("SZ_IS_64BIT_ARM_", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_WESTMERE", "1" if not is_64bit_arm() and is_64bit_x86() else "0"),
        ("SZ_USE_GOLDMONT", "1" if not is_64bit_arm() and is_64bit_x86() else "0"),
        ("SZ_USE_HASWELL", "1" if not is_64bit_arm() and is_64bit_x86() else "0"),
        ("SZ_USE_SKYLAKE", "0"),
        ("SZ_USE_ICELAKE", "0"),
        ("SZ_USE_NEON", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEONAES", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEONSHA", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE", "0"),
        ("SZ_USE_SVE2", "0"),
    ]
    return compile_args, link_args, macros_args
def windows_settings(use_cpp: bool = False) -> Tuple[List[str], List[str], List[Tuple[str]]]:
    compile_args = [
        "/std:c++17" if use_cpp else "/std:c11",
        "/W3",
        "/O2",
        "/wd4365",
        "/wd4820",
        "/wd5027",
        "/wd4626",
        "/wd4127",
    ]
    macros_args = [
        ("SZ_IS_BIG_ENDIAN_", "1" if is_big_endian() else "0"),
        ("SZ_IS_64BIT_X86_", "1" if is_64bit_x86() else "0"),
        ("SZ_IS_64BIT_ARM_", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_WESTMERE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_GOLDMONT", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_HASWELL", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_ICELAKE", "1" if is_64bit_x86() else "0"),
        ("SZ_USE_NEON", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEONAES", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_NEONSHA", "1" if is_64bit_arm() else "0"),
        ("SZ_USE_SVE", "0"),
        ("SZ_USE_SVE2", "0"),
    ]
    if is_64bit_arm():
        macros_args.append(("_ARM64_", "1"))
    elif is_64bit_x86():
        macros_args.append(("_AMD64_", "1"))
    link_args = []
    return compile_args, link_args, macros_args
use_cpp: Final[bool] = sz_target != "stringzilla"
if sys.platform == "linux" or sys.platform.startswith("freebsd"):
    compile_args, link_args, macros_args = linux_settings(use_cpp=use_cpp)
elif sys.platform == "darwin":
    compile_args, link_args, macros_args = darwin_settings(use_cpp=use_cpp)
elif sys.platform == "win32":
    compile_args, link_args, macros_args = windows_settings(use_cpp=use_cpp)
else:
    compile_args, link_args, macros_args = [], [], []
STRINGZILLA_CORE_SOURCES = [
    "c/stringzilla/runtime.c",
    "c/stringzilla/compare.c",
    "c/stringzilla/memory.c",
    "c/stringzilla/hash.c",
    "c/stringzilla/cipher.c",
    "c/stringzilla/find.c",
    "c/stringzilla/sort.c",
    "c/stringzilla/intersect.c",
    "c/stringzilla/utf8_norm.c",
    "c/stringzilla/utf8_runes.c",
    "c/stringzilla/utf8_tokens.c",
    "c/stringzilla/utf8_wordbreaks.c",
    "c/stringzilla/utf8_graphemes.c",
    "c/stringzilla/utf8_sentences.c",
    "c/stringzilla/utf8_linebreaks.c",
    "c/stringzilla/utf8_uncased_fold.c",
    "c/stringzilla/utf8_uncased.c",
]
STRINGZILLAS_API_CPP_SOURCES = [
    "c/stringzillas/runtime.cpp",
    "c/stringzillas/levenshtein.cpp",
    "c/stringzillas/needleman_wunsch.cpp",
    "c/stringzillas/smith_waterman.cpp",
    "c/stringzillas/fingerprints.cpp",
]
STRINGZILLAS_API_CU_SOURCES = [
    "c/stringzillas/runtime.cu",
    "c/stringzillas/levenshtein.cu",
    "c/stringzillas/needleman_wunsch.cu",
    "c/stringzillas/smith_waterman.cu",
    "c/stringzillas/fingerprints.cu",
]
STRINGZILLAS_CPUS_SOURCES = [
    "c/stringzillas/levenshtein_serial.cpp",
    "c/stringzillas/levenshtein_icelake.cpp",
    "c/stringzillas/levenshtein_haswell.cpp",
    "c/stringzillas/levenshtein_neon.cpp",
    "c/stringzillas/levenshtein_rvv.cpp",
    "c/stringzillas/needleman_wunsch_serial.cpp",
    "c/stringzillas/needleman_wunsch_icelake.cpp",
    "c/stringzillas/needleman_wunsch_haswell.cpp",
    "c/stringzillas/needleman_wunsch_neon.cpp",
    "c/stringzillas/needleman_wunsch_rvv.cpp",
    "c/stringzillas/smith_waterman_serial.cpp",
    "c/stringzillas/smith_waterman_icelake.cpp",
    "c/stringzillas/smith_waterman_haswell.cpp",
    "c/stringzillas/smith_waterman_neon.cpp",
    "c/stringzillas/smith_waterman_rvv.cpp",
]
STRINGZILLAS_CUDA_SOURCES = [
    "c/stringzillas/levenshtein_cuda.cu",
    "c/stringzillas/needleman_wunsch_cuda.cu",
    "c/stringzillas/smith_waterman_cuda.cu",
]
STRINGZILLAS_KEPLER_SOURCES = [
    "c/stringzillas/levenshtein_kepler.cu",
]
STRINGZILLAS_HOPPER_SOURCES = [
    "c/stringzillas/levenshtein_hopper.cu",
    "c/stringzillas/needleman_wunsch_hopper.cu",
    "c/stringzillas/smith_waterman_hopper.cu",
]
STRINGZILLAS_RUNTIME_SOURCES = ["forkunion/c/forkunion.cpp"]
ext_modules = []
entry_points = {}
command_class = {}
if sz_target == "stringzilla":
    __lib_name__ = "stringzilla"
    ext_modules = [
        Extension(
            "stringzilla",
            [
                "python/stringzilla/stringzilla.c",
                "python/stringzilla/shared.c",
                "python/stringzilla/file.c",
                "python/stringzilla/str.c",
                "python/stringzilla/strs.c",
                "python/stringzilla/memory.c",
                "python/stringzilla/hash.c",
                "python/stringzilla/cipher.c",
                "python/stringzilla/compare.c",
                "python/stringzilla/find.c",
                "python/stringzilla/sort.c",
                "python/stringzilla/intersect.c",
                "python/stringzilla/utf8_runes.c",
                "python/stringzilla/utf8_tokens.c",
                "python/stringzilla/utf8_boundaries.c",
                "python/stringzilla/utf8_wordbreaks.c",
                "python/stringzilla/utf8_graphemes.c",
                "python/stringzilla/utf8_sentences.c",
                "python/stringzilla/utf8_linebreaks.c",
                "python/stringzilla/utf8_uncased_fold.c",
                "python/stringzilla/utf8_uncased.c",
                "python/stringzilla/utf8_norm.c",
            ]
            + STRINGZILLA_CORE_SOURCES,
            include_dirs=["include", "c/stringzilla"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=[("SZ_DYNAMIC_DISPATCH", "1")] + macros_args,
        ),
    ]
    entry_points = {"console_scripts": []}
    command_class = {"build_ext": ParallelBuildExt}
elif sz_target == "stringzillas-cpus":
    __lib_name__ = "stringzillas-cpus"
    ext_modules = [
        Extension(
            "stringzillas",
            [
                "python/stringzillas/stringzillas.c",
                "python/stringzillas/device_scope.c",
                "python/stringzillas/similarities.c",
                "python/stringzillas/fingerprints.c",
            ]
            + STRINGZILLAS_API_CPP_SOURCES
            + STRINGZILLAS_CPUS_SOURCES
            + STRINGZILLAS_RUNTIME_SOURCES,
            include_dirs=["include", "c/stringzillas", "forkunion/include"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=[("SZ_DYNAMIC_DISPATCH", "1"), ("SZ_USE_CUDA", "0"), ("FU_WITH_TOPOLOGY", "0")] + macros_args,
        ),
    ]
    command_class = {"build_ext": NumpyBuildExt}
elif sz_target == "stringzillas-cuda":
    __lib_name__ = "stringzillas-cuda"
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    if os.name == "nt":
        cuda_link_args = link_args + [f"/LIBPATH:{os.path.join(cuda_home, 'lib', 'x64')}", "cudart.lib", "cuda.lib"]
    else:
        cuda_link_args = link_args + [
            f"-L{cuda_home}/lib64",
            f"-L{cuda_home}/lib64/stubs",
            "-lcudart",
            "-lcuda",
            "-lstdc++",
        ]
    ext_modules = [
        Extension(
            "stringzillas",
            [
                "python/stringzillas/stringzillas.c",
                "python/stringzillas/device_scope.c",
                "python/stringzillas/similarities.c",
                "python/stringzillas/fingerprints.c",
            ]
            + STRINGZILLAS_API_CU_SOURCES
            + STRINGZILLAS_CPUS_SOURCES
            + STRINGZILLAS_CUDA_SOURCES
            + STRINGZILLAS_KEPLER_SOURCES
            + STRINGZILLAS_HOPPER_SOURCES
            + STRINGZILLAS_RUNTIME_SOURCES,
            include_dirs=["include", "c/stringzillas", "forkunion/include", f"{cuda_home}/include"],
            extra_compile_args=compile_args,
            extra_link_args=cuda_link_args,
            define_macros=[("SZ_DYNAMIC_DISPATCH", "1"), ("SZ_USE_CUDA", "1"), ("FU_WITH_TOPOLOGY", "0")] + macros_args,
            language="c++",
        ),
    ]
    command_class = {"build_ext": CudaBuildExtension}
else:
    raise ValueError("Unknown target specified with SZ_TARGET environment variable.")
__version__ = open("VERSION", "r").read().strip()
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()
if sz_target == "stringzilla":
    __description__ = "Search, hash, sort, and process strings faster via SWAR and SIMD"
elif sz_target == "stringzillas-cpus":
    __description__ = (
        "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, on multi-core CPUs"
    )
elif sz_target == "stringzillas-cuda":
    __description__ = (
        "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, and CUDA on Nvidia GPUs"
    )
elif sz_target == "stringzillas-rocm":
    __description__ = (
        "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, and ROCm on AMD GPUs"
    )
else:
    __description__ = "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, and GPGPU"
install_requires = []
if sz_target != "stringzilla":
    install_requires = [f"stringzilla=={__version__}"]
setup(
    name=__lib_name__,
    version=__version__,
    description=__description__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/ashvardanian/StringZilla",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: OS Independent",
        "Topic :: File Formats",
        "Topic :: Internet :: Log Analysis",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: System :: Logging",
        "Topic :: Text Processing :: General",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.10",
    include_dirs=[],
    setup_requires=[],
    ext_modules=ext_modules,
    packages=[],
    entry_points=entry_points,
    cmdclass=command_class,
    install_requires=install_requires,
)
