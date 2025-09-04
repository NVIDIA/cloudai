# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CloudAI"
copyright = "2025, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA CORPORATION & AFFILIATES"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

exclude_patterns = ["_build"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"

# MyST parser configuration
myst_enable_extensions = [
    "deflist",
    "colon_fence",
    "html_image",
]

# Configure MyST to handle mermaid code blocks properly
myst_fence_as_directive = ["mermaid"]

# Mermaid configuration
mermaid_version = "latest"
mermaid_init_js = "mermaid.initialize({startOnLoad:true});"

# Set the root document to index
root_doc = "index"

source_suffix = [".rst", ".md"]
