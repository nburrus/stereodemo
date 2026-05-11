set shell := ["bash", "-uc"]

default:
    just --list

# Build source and wheel release artifacts into dist/.
build-release:
    ./build_release.sh

# Upload dist/* to PyPI.
#
# Set a PyPI API token before running this:
#   export UV_PUBLISH_TOKEN='pypi-...'
#   just publish
#
# Or pass it for one command:
#   UV_PUBLISH_TOKEN='pypi-...' just publish
publish:
    @if [[ -z "${UV_PUBLISH_TOKEN:-}" ]]; then \
        echo "UV_PUBLISH_TOKEN is not set."; \
        echo "Create a PyPI API token, then run:"; \
        echo "  export UV_PUBLISH_TOKEN='pypi-...'"; \
        echo "  just publish"; \
        exit 1; \
    fi
    uv publish dist/*
