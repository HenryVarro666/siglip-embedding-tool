import subprocess
import sys


def test_installed_training_module_help_works_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "siglip_tools.finetune", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--pairs" in result.stdout
    assert "--max-steps" in result.stdout
