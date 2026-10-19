import ctypes
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import bridge
from r2t2_core import native


class WorkerPythonTests(unittest.TestCase):
    def test_venv_precedes_portable_and_missing_keeps_setup_path(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(bridge, "ROOT", Path(directory)):
            base = Path(directory) / ".runtime/worker"
            venv = base / "Scripts/python.exe"
            portable = base / "python.exe"
            self.assertEqual(bridge._worker_python(), venv)
            base.mkdir(parents=True)
            portable.touch()
            self.assertEqual(bridge._worker_python(), portable)
            venv.parent.mkdir()
            venv.touch()
            self.assertEqual(bridge._worker_python(), venv)

    def test_override_is_used_by_worker_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            override = root / "custom/python.exe"
            override.parent.mkdir()
            override.touch()
            manager = bridge.WorkerManager()
            process = Mock()
            process.poll.return_value = None
            with patch.object(bridge, "ROOT", root), patch.dict(bridge.os.environ, {
                    "R2T2_WORKER_PYTHON": str(override)}), \
                    patch.object(bridge.subprocess, "Popen", return_value=process) as spawn, \
                    patch.object(manager, "_request", return_value={"generation": "test-generation"}):
                try:
                    manager._start()
                    self.assertEqual(spawn.call_args.args[0][0], str(override))
                    self.assertEqual(manager.generation, "test-generation")
                    self.assertEqual(spawn.call_args.kwargs["cwd"], str(root))
                finally:
                    manager.close()

    def test_invalid_override_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portable = root / ".runtime/worker/python.exe"
            portable.parent.mkdir(parents=True)
            portable.touch()
            with patch.object(bridge, "ROOT", root), patch.dict(bridge.os.environ, {
                    "R2T2_WORKER_PYTHON": str(root / "missing.exe")}):
                with self.assertRaisesRegex(bridge.WorkerError, "Isolated worker Python not found"):
                    bridge.WorkerManager()._start()


class BackendPluginTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.build = Path(self.temporary.name) / "中文组件"
        self.build.mkdir()
        self.registry = Mock()
        self.registry.ggml_backend_load.return_value = 1
        for patcher in (
                patch.object(native.ctypes, "WinDLL", return_value=self.registry, create=True),
                patch.object(native, "_BACKEND_REGISTRIES", {}),
                patch.object(native, "_LOADED_PLUGINS", set())):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.dll = native.ctypes.WinDLL

    def make_plugins(self, *names, registry=True):
        (self.build / "backends").mkdir(exist_ok=True)
        for name in names:
            (self.build / "backends" / name).touch()
        if registry:
            (self.build / "lib").mkdir(exist_ok=True)
            (self.build / "lib/ggml.dll").touch()

    def test_source_build_without_plugins_is_unchanged(self):
        native._load_backends(self.build)
        self.dll.assert_not_called()

    def test_plugins_without_registry_report_the_missing_library(self):
        self.make_plugins("ggml-cuda.dll", registry=False)
        with self.assertRaisesRegex(FileNotFoundError, "matching registry DLL"):
            native._load_backends(self.build)
        self.dll.assert_not_called()

    def test_plugins_register_once_and_keep_the_registry_alive(self):
        self.make_plugins("ggml-cuda.dll", "ggml-cpu-test.dll")
        native._load_backends(self.build)
        native._load_backends(self.build)
        self.dll.assert_called_once_with(str(self.build / "lib/ggml.dll"))
        paths = [call.args[0] for call in self.registry.ggml_backend_load.call_args_list]
        self.assertEqual(paths, [str(path).encode("utf-8") for path in sorted((self.build / "backends").glob("*.dll"))])
        self.assertIs(native._BACKEND_REGISTRIES[self.build / "lib/ggml.dll"], self.registry)
        self.assertEqual(self.registry.ggml_backend_load.argtypes, [ctypes.c_char_p])
        self.assertIs(self.registry.ggml_backend_load.restype, ctypes.c_void_p)

    def test_unsupported_cpu_variant_does_not_block_other_plugins(self):
        self.make_plugins("ggml-cpu-unsupported.dll", "ggml-cuda.dll")
        self.registry.ggml_backend_load.side_effect = [None, 1]
        native._load_backends(self.build)
        self.assertEqual(native._LOADED_PLUGINS, {self.build / "backends/ggml-cuda.dll"})

    def test_failed_plugin_can_be_retried_without_registering_successes_again(self):
        self.make_plugins("ggml-cpu-test.dll", "ggml-cuda.dll")
        self.registry.ggml_backend_load.side_effect = [1, None, 1]
        native._load_backends(self.build)
        native._load_backends(self.build)
        self.assertEqual(self.registry.ggml_backend_load.call_count, 3)
        self.assertEqual(len(native._LOADED_PLUGINS), 2)

    def test_extension_import_without_a_toolkit_uses_component_dlls(self):
        self.make_plugins("ggml-cuda.dll")
        extension = self.build / "python/Release/qwen3asr_native.pyd"
        extension.parent.mkdir(parents=True)
        extension.touch()
        add_directory = Mock()
        fake_os = types.SimpleNamespace(name="nt", environ={"CUDA_PATH": str(self.build / "missing-toolkit")},
                                        add_dll_directory=add_directory)
        with patch.object(native, "os", fake_os), patch.object(native, "_DLL_HANDLES", []), \
                patch.object(native.sys, "path", list(native.sys.path)), \
                patch.object(native.importlib, "import_module", return_value="extension") as load:
            self.assertEqual(native._load_extension(self.build), "extension")
            registered_dirs = {call.args[0] for call in add_directory.call_args_list}
            self.assertEqual(registered_dirs, {str(self.build / "lib"), str(self.build / "backends")})
            load.assert_called_once_with("qwen3asr_native")
            self.assertEqual(self.registry.ggml_backend_load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
