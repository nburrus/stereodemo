import site
import sys
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# Everything is defined in setup.cfg, added this file only
# to support editable mode.
import setuptools
setuptools.setup()
