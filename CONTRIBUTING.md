# Making a new release

- Bump the version numbers in `setup.cfg` and `stereodemo/__init__.py`

```
./build_release.sh
twine upload dist/*
```

Username is always `__token__`
