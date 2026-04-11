#!/usr/bin/env bash
#
# scripts/bump-version.sh vX.Y[.Z]
#
# Bumps the four version strings to match a git tag, then re-reads each file
# and fails loudly if any string didn't take. Run before tagging — see §3a in
# doc/process.md.
#
# Files bumped:
#   1. Cargo.toml                            ([workspace.package] version)
#   2. crates/llm-wasm/Cargo.toml            (excluded crate, standalone)
#   3. crates/llm-python/Cargo.toml          (excluded crate, standalone)
#   4. crates/llm-python/pyproject.toml      (Python wheel — uv cache key)
#
# The 5 in-workspace crates (llm-core, llm-openai, llm-anthropic, llm-store,
# llm-cli) inherit version from [workspace.package] and need no edits.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 vX.Y[.Z]" >&2
    exit 2
fi

TAG="$1"
VERSION="${TAG#v}"
case "$VERSION" in
    *.*.*) ;;
    *.*)   VERSION="${VERSION}.0" ;;
    *)
        echo "error: tag must be vX.Y or vX.Y.Z, got: $TAG" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

FILES=(
    "Cargo.toml"
    "crates/llm-wasm/Cargo.toml"
    "crates/llm-python/Cargo.toml"
    "crates/llm-python/pyproject.toml"
)

# Bump: rewrite the first `^version = "..."` line in each file. The package
# version is always at column 0; dependency versions live inside `{ ... }`
# blocks and won't match.
for rel in "${FILES[@]}"; do
    f="$REPO_ROOT/$rel"
    if [ ! -f "$f" ]; then
        echo "error: missing file: $rel" >&2
        exit 1
    fi
    perl -i -pe '
        BEGIN { $done = 0 }
        if (!$done && /^version = "[^"]*"/) {
            s/^version = "[^"]*"/version = "'"$VERSION"'"/;
            $done = 1;
        }
    ' "$f"
done

# Verify: re-read each file and assert the first ^version line matches.
fail=0
for rel in "${FILES[@]}"; do
    f="$REPO_ROOT/$rel"
    actual=$(perl -ne 'if (/^version = "([^"]*)"/) { print $1; exit }' "$f")
    if [ "$actual" != "$VERSION" ]; then
        printf '  FAIL  %s  (got "%s", want "%s")\n' "$rel" "$actual" "$VERSION" >&2
        fail=1
    else
        printf '  ok    %s  -> %s\n' "$rel" "$VERSION"
    fi
done

if [ "$fail" -ne 0 ]; then
    echo >&2
    echo "bump-version: verification failed — leaving files in mid-bump state for inspection." >&2
    exit 1
fi

cat <<EOF

Bumped to $VERSION. Next:

  git add ${FILES[*]}
  git commit -m "Bump version to $VERSION"
  git tag $TAG
EOF
