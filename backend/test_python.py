import unittest
import sys
import os
import subprocess

class TestPythonSyntax(unittest.TestCase):
    def test_syntax_compile(self):
        # Run py_compile to check syntax errors without executing the imports
        tryon_path = os.path.join(os.path.dirname(__file__), "tryon_local.py")
        print(f"Checking syntax of {tryon_path}")
        
        import py_compile
        compiled_file = py_compile.compile(tryon_path)
        self.assertIsNotNone(compiled_file, "Compilation failed")
        print("Syntax check passed!")

    def test_cli_arguments_if_dependencies_installed(self):
        # Skip executing the script if torch/transformers are not installed (e.g., in CI environments)
        try:
            import torch
            import transformers
            dependencies_available = True
        except ImportError:
            dependencies_available = False
            print("PyTorch/Transformers not installed. Skipping CLI execution test.")

        if dependencies_available:
            tryon_path = os.path.join(os.path.dirname(__file__), "tryon_local.py")
            result = subprocess.run(
                [sys.executable, tryon_path, "--help"],
                capture_output=True,
                text=True
            )
            self.assertEqual(result.returncode, 0, f"Script failed with: {result.stderr}")
            self.assertIn("Local AI Virtual Try-On", result.stdout)
            print("CLI arguments help test passed!")

if __name__ == "__main__":
    unittest.main()
