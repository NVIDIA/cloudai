import logging
from dataclasses import dataclass, field
from pathlib import Path

from .registry import Registry, Singleton
from .telemetry_logger import TelemetryLogger


@dataclass
class Telemetry(metaclass=Singleton):
    _loggers: list[TelemetryLogger] = field(default_factory=list)

    @property
    def loggers(self) -> list[TelemetryLogger]:
        if self._loggers:
            return self._loggers

        self._loggers = [logger() for logger in Registry().loggers]
        logging.debug(f"Telemetry loggers: {[logger.__class__.__name__ for logger in self._loggers]}")
        return self._loggers

    def log_metrics(self, metrics: dict) -> None:
        for logger in self.loggers:
            try:
                logger.log_metrics(metrics)
            except Exception as e:
                logging.error(f"Error logging metrics for {logger.__class__.__name__}: {e}")

    def set_output_dir(self, output_dir: Path) -> None:
        for logger in self.loggers:
            try:
                logger.set_output_dir(output_dir)
            except Exception as e:
                logging.error(f"Error setting output dir for {logger.__class__.__name__}: {e}")

    def set_name(self, name: str) -> None:
        for logger in self.loggers:
            try:
                logger.set_name(name)
            except Exception as e:
                logging.error(f"Error setting name for {logger.__class__.__name__}: {e}")
