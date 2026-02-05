import unittest
from pathlib import Path
from nanobot.agent.safety import OscillationDetector

class TestSafety(unittest.TestCase):
    def setUp(self):
        self.workspace = Path("/tmp/test_workspace")
        self.workspace.mkdir(exist_ok=True)
        self.detector = OscillationDetector(self.workspace, window_size=4, threshold=2)

    def test_oscillation_detection(self):
        # No oscillation
        self.assertIsNone(self.detector.check("ls", {"path": "."}))
        self.assertIsNone(self.detector.check("cat", {"file": "a.txt"}))

        # Oscillation
        self.detector.check("ls", {"path": "."})
        self.assertIsNotNone(self.detector.check("ls", {"path": "."}))

    def test_state_breaking_resets_history(self):
        self.detector.check("ls", {"path": "."})
        self.assertIsNone(self.detector.check("touch", {"file": "a.txt"}))
        self.assertIsNone(self.detector.check("ls", {"path": "."}))
        self.assertIsNone(self.detector.check("ls", {"path": "."})) # Should not trigger yet
        self.assertIsNotNone(self.detector.check("ls", {"path": "."})) # Should trigger now

    def test_normalization(self):
        # This test is a bit tricky as it depends on the workspace root.
        # We'll check that the normalized output is consistent.
        args1 = {"command": "cat ./file.txt"}
        args2 = {"command": f"cat {self.workspace / 'file.txt'}"}

        sig1 = self.detector.normalize_args("exec", args1)
        sig2 = self.detector.normalize_args("exec", args2)

        self.assertEqual(sig1, sig2)

    def test_no_oscillation_at_window_limit(self):
        self.assertIsNone(self.detector.check("ls", {"path": "dir1"}))
        self.assertIsNone(self.detector.check("ls", {"path": "dir2"}))
        self.assertIsNone(self.detector.check("ls", {"path": "dir3"}))
        self.assertIsNone(
            self.detector.check("ls", {"path": "dir4"})
        )  # Reaches window size, but no oscillation
        # Now, if we add one more unique one, the oldest ("dir1") is pushed out
        self.assertIsNone(self.detector.check("ls", {"path": "dir5"}))
        # And adding a repeat of a recent one should not trigger
        self.assertIsNone(self.detector.check("ls", {"path": "dir2"}))
        # But repeating it again should
        self.assertIsNotNone(self.detector.check("ls", {"path": "dir2"}))


if __name__ == "__main__":
    unittest.main()
