# Contributing
This guide documents the release and maintenance workflow for `sb-salt`. Use it when setting up local development, updating versions, building release artifacts, publishing to PyPI, or configuring a new maintainer for package releases.

## Prerequisites
1. Use Python 3.9 or newer.

2. Install `uv` from https://github.com/astral-sh/uv.

3. Have a PyPI account with access to the `sb-salt` project.

4. Confirm the GitHub repository has a `PYPI_TOKEN` secret configured in Settings > Secrets > Actions.

## Local Development Setup
1. Clone the repository:

```bash
git clone https://github.com/SunbirdAI/salt.git
cd salt
```

2. Switch to the release branch:

```bash
git checkout sb-salt
```

3. Install the package in editable mode with development dependencies:

```bash
uv pip install -e ".[dev]"
```

4. Run the test suite:

```bash
pytest tests/
```

## Versioning Policy
1. Use Semantic Versioning: `MAJOR.MINOR.PATCH`.

2. Use a PATCH version for bug fixes:

```text
0.1.1
```

3. Use a MINOR version for new features:

```text
0.2.0
```

4. Use a MAJOR version for breaking changes:

```text
1.0.0
```

5. Define the version in one place only:

```bash
salt/__init__.py
```

6. Do not edit the version in `pyproject.toml`; it reads the version dynamically from `salt/__init__.py`.

## Release Process
1. Make all release changes on the `sb-salt` branch, not `main`:

```bash
git checkout sb-salt
```

2. Update `__version__` in `salt/__init__.py`:

```python
__version__ = "X.Y.Z"
```

3. Run the test suite:

```bash
pytest tests/
```

4. Commit the release change:

```bash
git add salt/__init__.py pyproject.toml
git commit -m "Release vX.Y.Z — short description of changes"
```

5. Tag the release:

```bash
git tag vX.Y.Z
```

6. Push the release branch:

```bash
git push origin sb-salt
```

7. Push the release tag:

```bash
git push origin vX.Y.Z
```

8. Let GitHub Actions run `.github/workflows/publish.yml`. It will build the `.whl` and `.tar.gz` artifacts with:

```bash
python -m build
```

9. Let GitHub Actions upload the artifacts to PyPI with `twine` using the `PYPI_TOKEN` repository secret.

10. Verify the release at https://pypi.org/project/sb-salt.

## Manual Build And Upload
1. Use this only if GitHub Actions is unavailable.

2. Install the build and upload tools:

```bash
uv pip install build twine
```

3. Build the package artifacts:

```bash
python -m build
```

4. Check the artifacts before upload:

```bash
twine check dist/*
```

5. Upload the artifacts:

```bash
twine upload dist/*
```

6. Use these PyPI credentials when prompted:

```text
Username: __token__
Password: your pypi-... token
```

## First-Time PyPI Setup
1. Open PyPI account settings at https://pypi.org/manage/account/.

2. Go to API tokens.

3. Create a token scoped to the `sb-salt` project.

4. Add the token to GitHub repository secrets:

```text
Settings > Secrets > Actions > New repository secret
```

5. Name the secret:

```text
PYPI_TOKEN
```

## Branch Strategy
1. Use `main` for development and experimentation.

2. Use `sb-salt` as the release branch.

3. Tag all versioned releases from `sb-salt`.

4. Never tag a release from `main`.

## What Not To Do
1. Never upload the same version twice; PyPI will reject it.

2. Never manually edit the version in `pyproject.toml`; it reads from `salt/__init__.py` dynamically.

3. Never run `twine upload` on unverified `dist/` artifacts.

4. Always run this before any manual upload:

```bash
twine check dist/*
```
