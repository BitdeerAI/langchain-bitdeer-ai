#!/bin/bash
# Checks, builds and publishes a release to PyPI. Stops at the first failure, so nothing is uploaded
# unless every check passes. Run it from a clean checkout of the release tag:
#
#     UV_PUBLISH_TOKEN="$(<read the token from your password manager>)" make release
#
# Set UV_PUBLISH_URL=https://test.pypi.org/legacy/ to publish to TestPyPI instead.

set -euo pipefail

export UV_FROZEN=true

if [ -z "${UV_PUBLISH_TOKEN:-}" ]; then
    echo "UV_PUBLISH_TOKEN is not set." >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "The working tree has uncommitted changes." >&2
    exit 1
fi

version="$(uv version --short)"
if [ "$(git tag --points-at HEAD)" != "v$version" ]; then
    echo "HEAD must be tagged v$version (the version in pyproject.toml), and nothing else." >&2
    exit 1
fi

git fetch --quiet origin main
if ! git merge-base --is-ancestor HEAD origin/main; then
    echo "HEAD is not on origin/main." >&2
    exit 1
fi

make lint
make check_imports

# Every Python version in the classifiers, each in a throwaway environment so .venv is left alone.
for python in 3.10 3.11 3.12 3.13 3.14; do
    uv run --isolated --python "$python" --group test \
        pytest --disable-socket --allow-unix-socket tests/unit_tests/
done

# Collects the integration tests without running them, so a broken import fails here even though the
# live tests are skipped without an API key.
uv run --group test --group test_integration pytest -m compile tests/integration_tests

if [ -n "$(git status --porcelain)" ]; then
    echo "The checks changed files in the working tree." >&2
    exit 1
fi

rm -rf dist
uv build --no-sources
uv run --isolated --no-project --with dist/*.whl python -I -c "import langchain_bitdeer_ai"
uv run --isolated --no-project --with dist/*.tar.gz python -I -c "import langchain_bitdeer_ai"

uv publish
