Development
===========

This document targets developers who want to contribute to the project's core.

.. mermaid::

   graph TD
       subgraph _core
           base_modules
           core_implementations
           registry
       end

       subgraph runners
           SlurmRunner
           StandaloneRunner
       end

       subgraph installers
           SlurmInstaller
           StandaloneInstaller
       end

       subgraph systems
           SlurmSystem
           StandaloneSystem
       end

       installers --> _core
       runners --> _core
       systems --> _core

Core Modules
------------

We use `import-linter <https://github.com/seddonym/import-linter>`_ to ensure no core modules import higher level modules.

``Registry`` object is a singleton that holds implementation mappings. Users can register their own implementations to the registry or replace the default implementations.

Cache
-----

Some prerequisites can be installed. For example: Docker images, git repos with executable scripts, etc. All such "installables" are kept under system ``install_path``.

Installables are shared among all tests. So if any number of tests use the same installable, it is installed only once for a particular system TOML.

.. mermaid::

   classDiagram
       class Installable {
           <<abstract>>
           + __eq__(other: object)
           + __hash__()
       }

       class DockerImage {
           + url: str
           + install_path: str | Path
       }

       class GitRepo {
           + git_url: str
           + commit_hash: str
           + install_path: Path
       }

       class PythonExecutable {
           + git_repo: GitRepo
           + venv_path: Path
       }

       Installable <|-- DockerImage
       Installable <|-- GitRepo
       Installable <|-- PythonExecutable
       PythonExecutable --> GitRepo

       class BaseInstaller {
           <<abstract>>
           + install(items: Iterable[Installable])
           + uninstall(items: Iterable[Installable])
           + is_installed(items: Iterable[Installable]) -> bool

           * install_one(item: Installable)
           * uninstall_one(item: Installable)
           * is_installed_one(item: Installable) -> bool
       }

       BaseInstaller <|-- SlurmInstaller
       BaseInstaller <|-- StandaloneInstaller

