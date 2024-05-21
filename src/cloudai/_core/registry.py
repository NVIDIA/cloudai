from typing import Dict, Type

from cloudai.parser.core.base_system_parser import BaseSystemParser


class Singleton(type):
    """Singleton metaclass."""

    _instance = None

    def __new__(cls, name, bases, dct):
        if not isinstance(cls._instance, cls):
            cls._instance = super().__new__(cls, name, bases, dct)
        return cls._instance


class Registry(metaclass=Singleton):
    """Registry for implementations mappings."""

    system_parsers_map: Dict[str, Type[BaseSystemParser]] = {}

    def add_system_parser(self, name: str, value: Type[BaseSystemParser]) -> None:
        """
        Add a new system parser implementation mapping.

        Args:
            name (str): The name of the system parser.
            value (Type[BaseSystemParser]): The system parser implementation.

        Raises:
            ValueError: If the system parser implementation already exists.
        """
        if name in self.system_parsers_map:
            raise ValueError(f"Duplicating implementation for '{name}', use 'update()' for replacement.")
        self.update_system_parser(name, value)

    def update_system_parser(self, name: str, value: Type[BaseSystemParser]) -> None:
        """
        Create or replace system parser implementation mapping.

        Args:
            name (str): The name of the system parser.
            value (Type[BaseSystemParser]): The system parser implementation.

        Raises:
            ValueError: If the system parser implementation does not exist.
        """
        if not issubclass(value, BaseSystemParser):
            raise ValueError(f"Invalid system implementation for '{name}', should be subclass of 'System'.")
        self.system_parsers_map[name] = value
