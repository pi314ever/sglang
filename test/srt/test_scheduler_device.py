import os
import unittest
from unittest import mock

from sglang.srt.utils import get_scheduler_device


class TestSchedulerDeviceEnv(unittest.TestCase):

    def test_env_override_cuda(self):
        """Should return env value when set (e.g., CUDA)."""
        with mock.patch.dict(os.environ, {"SGLANG_SCHEDULER_DEVICE": "cuda:1"}):
            result = get_scheduler_device("cuda:0")
            self.assertEqual(result, "cuda:1")

    def test_env_override_hpu(self):
        """Should return env value even if HPU is present."""
        with mock.patch.dict(os.environ, {"SGLANG_SCHEDULER_DEVICE": "hpu"}):
            with mock.patch("sglang.srt.utils.is_hpu", return_value=True):
                result = get_scheduler_device("hpu")
                self.assertEqual(result, "hpu")

    def test_default_cuda_when_not_hpu(self):
        """Should return worker device if not HPU and no env var."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("sglang.srt.utils.is_hpu", return_value=False):
                result = get_scheduler_device("cuda:0")
                self.assertEqual(result, "cuda:0")

    def test_default_cpu_when_hpu(self):
        """Should return cpu if HPU is used and no env var is set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("sglang.srt.utils.is_hpu", return_value=True):
                result = get_scheduler_device("hpu")
                self.assertEqual(result, "cpu")


if __name__ == "__main__":
    unittest.main()
