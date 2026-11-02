"""OS selection, installed-wheel preference and safe wheel installation."""

import importlib.util
import json
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import bridge
from r2t2_core import native

spec = importlib.util.spec_from_file_location("setup_r2t2_worker", Path(__file__).parents[1] / "scripts/setup_worker.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class PlatformRuntimeTests(unittest.TestCase):
    def test_linux_selects_bin_python_and_never_windows_executable(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bridge, "ROOT", Path(tmp)):
            base = Path(tmp) / ".runtime/worker"
            (base / "Scripts").mkdir(parents=True)
            (base / "Scripts/python.exe").touch()
            (base / "python.exe").touch()
            expected = base / "bin/python"
            self.assertEqual(bridge._worker_python("linux"), expected)
            (base / "bin").mkdir()
            (base / "bin/python3").touch()
            self.assertEqual(bridge._worker_python("linux"), base / "bin/python3")
            expected.touch()
            self.assertEqual(bridge._worker_python("linux"), expected)

    def test_installed_wheel_is_loaded_without_legacy_build_directory(self):
        expected = object()
        package = types.SimpleNamespace(API_VERSION=1, LLAMA_COMMIT=native.LLAMA_COMMIT,
                                        load_native=lambda: expected)
        with patch.object(native.importlib, "import_module", return_value=package):
            self.assertIs(native._load_extension(Path("nonexistent-build")), expected)

    def test_broken_wheel_dependency_is_not_hidden_by_legacy_fallback(self):
        with patch.object(native.importlib, "import_module",
                          side_effect=ModuleNotFoundError("missing numpy", name="numpy")):
            with self.assertRaisesRegex(ModuleNotFoundError, "numpy"):
                native._load_extension(Path("nonexistent-build"))

    def test_missing_wheel_and_build_give_platform_neutral_setup_command(self):
        with patch.object(native, "_installed_native", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "python scripts/setup_worker.py"):
                native._load_extension(Path("nonexistent-build"))

    def test_native_api_mismatch_is_not_silently_accepted(self):
        package = types.SimpleNamespace(API_VERSION=9, LLAMA_COMMIT=native.LLAMA_COMMIT)
        with patch.object(native.importlib, "import_module", return_value=package):
            with self.assertRaisesRegex(RuntimeError, "Incompatible"):
                native._installed_native()

    def test_legacy_linux_extension_uses_abi_suffix_without_dll_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            extension = build / "python" / ("qwen3asr_native" + native.importlib.machinery.EXTENSION_SUFFIXES[0])
            extension.parent.mkdir()
            extension.touch()
            fake_os = types.SimpleNamespace(name="posix", environ={})
            with patch.object(native, "os", fake_os), \
                    patch.object(native, "_installed_native", return_value=None), \
                    patch.object(native.importlib, "import_module", return_value="linux extension"), \
                    patch.object(native.sys, "path", list(native.sys.path)):
                self.assertEqual(native._load_extension(build), "linux extension")

    def test_setup_interpreter_paths_match_each_host(self):
        base = Path("worker")
        self.assertEqual(setup.interpreter(base, "linux"), base / "bin/python")
        self.assertEqual(setup.interpreter(base, "win32"), base / "Scripts/python.exe")

    def test_wheel_selection_uses_backend_and_actual_compatible_tags(self):
        manifest = {"version": setup.NATIVE_VERSION, "llama_commit": setup.LLAMA_COMMIT,
                    "wheels": [{"filename": "cpu.whl", "backend": "cpu", "sha256": "a"*64,
                                "tags": ["cp312-cp312-manylinux_2_28_x86_64"]},
                               {"filename": "gpu.whl", "backend": "cuda", "sha256": "b"*64,
                                "tags": ["cp312-cp312-win_amd64"]}]}
        tags = {"cp312-cp312-manylinux_2_28_x86_64"}
        self.assertEqual(setup.select_wheel(manifest, tags, "cpu")["filename"], "cpu.whl")
        with self.assertRaisesRegex(RuntimeError, "No compatible"):
            setup.select_wheel(manifest, tags, "cuda")
        manifest["wheels"][0]["filename"] = "../escape.whl"
        with self.assertRaisesRegex(ValueError, "asset name"):
            setup.select_wheel(manifest, tags, "cpu")

    def test_local_wheel_checks_model_backend_provenance_and_no_executables(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "native.whl"
            with zipfile.ZipFile(wheel, "w") as z:
                z.writestr("r2t2_native/build-info.json", json.dumps({
                    "llama_commit": setup.LLAMA_COMMIT, "api_version": 1, "cuda": False}))
            setup.check_wheel(wheel, "cpu")
            with self.assertRaisesRegex(ValueError, "does not match"):
                setup.check_wheel(wheel, "cuda")
            with zipfile.ZipFile(wheel, "a") as z:
                z.writestr("llama.exe", b"binary")
            with self.assertRaisesRegex(ValueError, "executable"):
                setup.check_wheel(wheel, "cpu")

    def test_auto_backend_does_not_assume_cuda_without_usable_nvidia_gpu(self):
        with patch.object(setup.shutil, "which", return_value=None):
            self.assertEqual(setup.detect_backend("auto"), "cpu")
        self.assertEqual(setup.detect_backend("cuda"), "cuda")


if __name__ == "__main__":
    unittest.main()
