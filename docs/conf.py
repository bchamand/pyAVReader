# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import datetime
import inspect
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.abspath("../"))

import avreader


# -- Project information -----------------------------------------------------

project = "pyAVReader"
author = "Benjamin Chamand"
version = avreader.__version__
release = avreader.__version__
copyright = f"{datetime.datetime.now().year}, {author}."


# -- Github Project information ----------------------------------------------

github_user = "bchamand"
github_repo = "avreader"


def subprocess_cmd(cmd):
    res = subprocess.run(
        cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    return res.stdout.strip().decode("utf-8")


github_version = version
try:
    list_tags = subprocess_cmd("git tag --list --sort=-v:refname").split("\n")
    github_version = f"v{version}"
    if list_tags and (list_tags[0] == f"v{version}"):
        github_version = "stable"
except:
    pass

git_revision = ""
try:
    git_revision = subprocess_cmd("git rev-parse --short HEAD")
except:
    pass


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.linkcode",
    # "sphinx.ext.intersphinx",
    "recommonmark",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
add_module_names = True
source_suffix = {".rst": "restructuredtext"}
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_js_files = ["js/sphinx_rtd_versions.js"]

html_context = {
    "display_github": True,
    "github_user": github_user,
    "github_repo": github_repo,
    "github_version": "master" if "dev" in version else version,
    "commit": git_revision,
    "conf_py_path": "/docs/",
}

html_theme_options = {
    "display_version": True,
    "collapse_navigation": False,
}


# -- Extension configuration -------------------------------------------------

# autodoc settings
autodoc_typehints = "description"

autosummary_generate = True
autodoc_member_order = "alphabetical"

# napoleon settings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

# linkcode settings
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(avreader.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "{}#L{:d}-L{:d}".format(*find_source())
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"

    branch = "master" if "dev" in version else f"v{version}"
    github_url = (
        f"https://github.com/{github_user}/{github_repo}/blob/{branch}/{filename}"
    )
    return github_url
