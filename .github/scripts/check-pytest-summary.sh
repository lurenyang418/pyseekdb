#!/usr/bin/env bash

# Validate pytest's final summary as a defense-in-depth check: pipe-based test
# runners can otherwise obscure a non-success outcome. Fail closed if the
# summary reports failures/errors or does not confirm that at least one test passed.
set -euo pipefail

pytest_log="${1:-pytest.log}"

if [[ ! -s "${pytest_log}" ]]; then
  echo "pytest log is missing or empty: ${pytest_log}" >&2
  exit 1
fi

summary="$(awk 'NF { last = $0 } END { print last }' "${pytest_log}")"
failure_pattern='(^|[[:space:]])[1-9][0-9]*[[:space:]]+(failed|errors?)([,=[:space:]]|$)'
success_pattern='(^|[[:space:]])[1-9][0-9]*[[:space:]]+passed([,=[:space:]]|$)'

if printf '%s\n' "${summary}" | grep -Eiq "${failure_pattern}"; then
  echo "pytest reported failures: ${summary}" >&2
  exit 1
fi

if ! printf '%s\n' "${summary}" | grep -Eiq "${success_pattern}"; then
  echo "could not verify a successful pytest summary: ${summary}" >&2
  exit 1
fi
