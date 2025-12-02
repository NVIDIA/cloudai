import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils.tqdm import disable_progress_bars

from cloudai.core import HFModel, InstallStatusResult


@dataclass
class HFModelManager:
    """Manager for HuggingFace models."""

    root_path: Path

    def model_path(self, model: HFModel) -> Path:
        return self.root_path / "hub" / model.model_name

    def download_model(self, model: HFModel) -> InstallStatusResult:
        logging.debug(f"Downloading HF model {model.model_name} into {self.root_path / 'hub'}")
        disable_progress_bars()
        try:
            local_path: str = snapshot_download(repo_id=model.model_name, cache_dir=self.root_path.absolute() / "hub")
            model.installed_path = Path(local_path)
        except Exception as e:
            return InstallStatusResult(False, f"Failed to download HF model {model.model_name}: {e}")

        return InstallStatusResult(True)

    def is_model_downloaded(self, model: HFModel) -> InstallStatusResult:
        logging.debug(f"Checking if HF model {model.model_name} is already downloaded in {self.root_path / 'hub'}")
        disable_progress_bars()
        try:
            local_path: str = snapshot_download(
                repo_id=model.model_name, cache_dir=self.root_path.absolute() / "hub", local_files_only=True
            )
        except Exception as e:
            return InstallStatusResult(False, f"Failed to download HF model {model.model_name}: {e}")

        return InstallStatusResult(True, local_path)

    def remove_model(self, model: HFModel) -> InstallStatusResult:
        logging.debug(f"Removing HF model {model.model_name} from {self.root_path / 'hub'}")
        res = self.is_model_downloaded(model)
        if not res.success:
            return InstallStatusResult(True, f"HF model {model.model_name} is not downloaded.")

        cmd = ["hf", "cache", "rm", "-y", f"model/{model.model_name}"]
        env = os.environ | {"HF_HOME": str(self.root_path.absolute())}
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        logging.debug(
            f"Run {cmd=} with HF_HOME={env['HF_HOME']} returned code {p.returncode}, stdout: {p.stdout}, stderr: {p.stderr}"
        )
        if p.returncode != 0:
            return InstallStatusResult(False, f"Failed to remove HF model {model.model_name}: {p.stderr} {p.stdout}")

        return InstallStatusResult(True)
