import unittest
from unittest.mock import patch, MagicMock
from dev_parallel_check import run_cmd, run_parallel_checks

class TestDevParallelCheck(unittest.TestCase):

    @patch('subprocess.run')
    def test_run_cmd(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        cmd, code, out, err = run_cmd("echo hi")
        self.assertEqual(code, 0)
        self.assertEqual(out, "OK")

    @patch('dev_parallel_check.run_cmd')
    def test_run_parallel_checks(self, mock_cmd):
        mock_cmd.side_effect = lambda cmd: (cmd, 0, "OK", "")
        res = run_parallel_checks()
        self.assertIn("pytest --quiet", res)
        self.assertEqual(res["pytest --quiet"]["code"], 0)

if __name__ == '__main__':
    unittest.main()
