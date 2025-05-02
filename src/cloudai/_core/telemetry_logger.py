import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from one_logger.core import OneLogger


class TelemetryLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict) -> None: ...

    @abstractmethod
    def set_output_dir(self, output_dir: Path) -> None: ...

    @abstractmethod
    def set_name(self, name: str) -> None: ...


class OneLoggerTelemetry(TelemetryLogger):
    def __init__(self) -> None:
        self.config = {
            "enable_one_logger": True,
            "is_baseline_run": False,
            "wandb": {
                "host": "https://api.wandb.ai",
                "entity": "hwinf_dcm",
                "project": "CloudAIt",
                "name": "cloudai-default-name",
                "save_dir": str(Path.cwd() / ".wandb"),
            },
        }
        self._one_logger: Optional[OneLogger] = None

    @property
    def actual_logger(self) -> OneLogger:
        """Delay creation of OneLogger instance until it's needed."""
        logging.info("Creating OneLogger instance.")
        if self._one_logger is None:
            self._one_logger = OneLogger(config=self.config)
        return self._one_logger

    def set_name(self, name: str) -> None:
        self.config["wandb"]["name"] = name

    def set_output_dir(self, output_dir: Path) -> None:
        self.config["wandb"]["save_dir"] = str(output_dir)

    def log_metrics(self, metrics: dict) -> None:
        self.actual_logger.log_metrics(metrics)
