from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "my_on_policy_runner.py"
)


class DummyOnPolicyRunner:
    pass


class DummyVecEnv:
    pass


class DummyEmpiricalNormalization:
    pass


def load_runner_module():
    torch_module = types.ModuleType("torch")
    rsl_rl_module = types.ModuleType("rsl_rl")
    rsl_rl_env_module = types.ModuleType("rsl_rl.env")
    rsl_rl_env_module.VecEnv = DummyVecEnv
    rsl_rl_modules_module = types.ModuleType("rsl_rl.modules")
    rsl_rl_modules_module.EmpiricalNormalization = DummyEmpiricalNormalization
    rsl_rl_runners_module = types.ModuleType("rsl_rl.runners")
    rsl_rl_runner_module = types.ModuleType("rsl_rl.runners.on_policy_runner")
    rsl_rl_runner_module.OnPolicyRunner = DummyOnPolicyRunner

    isaaclab_rl_module = types.ModuleType("isaaclab_rl")
    isaaclab_rl_rsl_module = types.ModuleType("isaaclab_rl.rsl_rl")
    isaaclab_rl_rsl_module.export_policy_as_onnx = mock.Mock()

    whole_body_tracking_module = types.ModuleType("whole_body_tracking")
    whole_body_tracking_utils_module = types.ModuleType("whole_body_tracking.utils")
    exporter_module = types.ModuleType("whole_body_tracking.utils.exporter")
    exporter_module.attach_onnx_metadata = mock.Mock()
    exporter_module.export_motion_policy_as_onnx = mock.Mock()

    stub_modules = {
        "torch": torch_module,
        "rsl_rl": rsl_rl_module,
        "rsl_rl.env": rsl_rl_env_module,
        "rsl_rl.modules": rsl_rl_modules_module,
        "rsl_rl.runners": rsl_rl_runners_module,
        "rsl_rl.runners.on_policy_runner": rsl_rl_runner_module,
        "isaaclab_rl": isaaclab_rl_module,
        "isaaclab_rl.rsl_rl": isaaclab_rl_rsl_module,
        "whole_body_tracking": whole_body_tracking_module,
        "whole_body_tracking.utils": whole_body_tracking_utils_module,
        "whole_body_tracking.utils.exporter": exporter_module,
    }
    module_spec = importlib.util.spec_from_file_location("my_on_policy_runner_test", RUNNER_PATH)
    assert module_spec and module_spec.loader
    runner_module = importlib.util.module_from_spec(module_spec)
    with mock.patch.dict(sys.modules, stub_modules):
        module_spec.loader.exec_module(runner_module)
    return runner_module


class OnnxCheckpointExportTest(unittest.TestCase):
    def test_resume_continues_after_saved_iteration(self) -> None:
        runner_module = load_runner_module()
        runner = types.SimpleNamespace(current_learning_iteration=0)

        runner_module._set_next_learning_iteration(runner, 58500)

        self.assertEqual(runner.loaded_checkpoint_iteration, 58500)
        self.assertEqual(runner.current_learning_iteration, 58501)

    def test_training_preserves_loaded_checkpoint_before_learning(self) -> None:
        train_source = (RUNNER_PATH.parents[4] / "scripts" / "rsl_rl" / "train.py").read_text(encoding="utf-8")

        load_position = train_source.index("runner.load(resume_path)")
        save_position = train_source.index("runner.save(preserved_checkpoint_path)")
        learn_position = train_source.index("runner.learn(")

        self.assertLess(load_position, save_position)
        self.assertLess(save_position, learn_position)
        self.assertIn('f"model_{loaded_iteration}.pt"', train_source)

    def test_uses_checkpoint_iteration_for_versioned_onnx_name(self) -> None:
        runner_module = load_runner_module()

        export_directory, versioned_filename, latest_filename = runner_module.get_onnx_export_names(
            "/tmp/g1_flat/example_run/model_1500.pt"
        )

        self.assertEqual(export_directory, "/tmp/g1_flat/example_run")
        self.assertEqual(versioned_filename, "model_1500.onnx")
        self.assertEqual(latest_filename, "example_run.onnx")

    def test_updates_latest_onnx_without_removing_versioned_export(self) -> None:
        runner_module = load_runner_module()
        with tempfile.TemporaryDirectory() as directory:
            export_directory = Path(directory)
            versioned_path = export_directory / "model_500.onnx"
            latest_path = export_directory / "example_run.onnx"
            versioned_path.write_bytes(b"checkpoint-500")

            runner_module.update_latest_onnx(
                str(export_directory),
                versioned_path.name,
                latest_path.name,
            )

            self.assertEqual(versioned_path.read_bytes(), b"checkpoint-500")
            self.assertEqual(latest_path.read_bytes(), b"checkpoint-500")


if __name__ == "__main__":
    unittest.main()
