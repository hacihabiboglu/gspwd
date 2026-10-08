# Contributing

The software is licensed under the PolyForm Strict License 1.0.0, which does not permit
modification or redistribution without permission. That includes pull requests and forks you
publish.

- **Bug reports and questions** are welcome as issues.
- **Code contributions:** please open an issue first and wait for written permission (see the
  *Research permission* paragraph in the README). Contributors will be asked to assign or license
  their contribution to the licensor so that it can be offered under the project's licenses.

For maintainers:

```bash
pip install -e ".[dev]"      # runtime + pytest, ruff, matplotlib, pandas
ruff check src tests && ruff format src tests
pytest                       # fast; no healpy needed
pip install -e ".[oracle]" && pytest -m oracle   # optional: compare against healpy
```

- `tests/golden/golden_v1.npz` was produced by the original reference implementation
  (`scripts/make_golden.py`). Changes to estimator numerics must either keep these tests green or
  come with an explicit, documented decision (see `CHANGELOG.md`).
