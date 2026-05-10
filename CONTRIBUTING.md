# Making a new release

- Bump the version numbers in `pyproject.toml` and `stereodemo/__init__.py`

```
./build_release.sh
uv run twine upload dist/*
```

Username is always `__token__`
