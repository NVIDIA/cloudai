from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class ConstraintChecker(ABC):
    """
    Abstract base class for checking constraints on actions.

    This class should be subclassed to implement specific constraint checks.

    Attributes:
        test_run (Any): The test run instance associated with this checker.
    """

    def __init__(self, test_run: Any):
        self.test_run = test_run

    @abstractmethod
    def validate(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate the action against the constraints.

        Args:
            action (Dict[str, Any]): The action to validate.

        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: True if the action is valid, False otherwise.
                - str: Error message if the action is invalid.
        """
        pass
