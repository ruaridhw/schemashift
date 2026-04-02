import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "schemashift"
author = "Ruaridh Williamson"
release = "0.1.0"
copyright = "2026, Ruaridh Williamson"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_autodoc_typehints",
]

html_theme = "furo"
html_title = "schemashift"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/ruaridhw/schemashift",
    "source_branch": "main",
    "source_directory": "docs/",
}

# MyST
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
]
myst_heading_anchors = 3

# Napoleon — Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True

# autodoc
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_member_order = "groupwise"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# sphinx-autodoc-typehints
always_document_param_types = False
typehints_fully_qualified = False

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

# Source file types
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
