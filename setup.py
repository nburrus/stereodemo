import site
import sys
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# Project metadata lives in pyproject.toml. This file only keeps compatibility
# with older editable-install workflows.
import setuptools
setuptools.setup()
