# Making a new release

- Bump the version numbers in `setup.cfg` and `stereodemo/__init__.py`

```
rm dist/*
python3 -m build
twine upload dist/*
```

Username is always `__token__`
