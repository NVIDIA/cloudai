from typing import Dict, Type

from cloudai.parser.core.base_system_parser import BaseSystemParser


class Singleton(type):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = type.__new__(cls, *args, **kwargs)
        return cls._instance


class Registry(metaclass=Singleton):
    system_parsers_map: Dict[str, Type[BaseSystemParser]] = {}

    def add_system_parser(self, name: str, value: Type[BaseSystemParser]) -> None:
        if name in self.system_parsers_map:
            raise ValueError(f"Duplicating implementation for '{name}', use 'update()' for replacement.")
        self.update_system_parser(name, value)

    def update_system_parser(self, name: str, value: Type[BaseSystemParser]) -> None:
        if not issubclass(value, BaseSystemParser):
            raise ValueError(f"Invalid system implementation for '{name}', should be subclass of 'System'.")
        self.system_parsers_map[name] = value
