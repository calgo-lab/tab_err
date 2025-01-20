import sys
from pathlib import Path

sys.path.insert(0, str(Path("..", "tab_err").resolve()))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tab_err"
copyright = "2025, Philipp Jung and Sebastian Jäger"
author = "Philipp Jung and Sebastian Jäger"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["autoapi.extension", "sphinx.ext.napoleon", "myst_parser"]
templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for autoapi -------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../../tab_err/"]
autoapi_keep_files = True
autoapi_root = "api"
autoapi_member_order = "groupwise"
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/calgo-lab/tab_err",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "secondary_sidebar_items": {
        "**": ["page-toc", "edit-this-page"],
    },
}

html_context = {
    "github_user": "calgo-lab",
    "github_repo": "tab_err",
    "github_version": "main",
    "doc_path": "docs/source",
}
