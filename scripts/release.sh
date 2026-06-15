#!/usr/bin/env bash
#
# Cut a new x3fuse-core release.
#
# Releases are published by CI (.github/workflows/ci.yml) when a tag
# matching `v*` is pushed: the `release` job builds every target and runs
# `gh release create --generate-notes`. This script automates the two
# manual halves of that flow:
#
#   1. prepare  — bump the workspace version across every Cargo.toml that
#                 carries it, refresh + verify the build, commit the bump
#                 on a `release/vX.Y.Z` branch, and (optionally) open a PR.
#   2. --tag    — once the bump has merged to `main`, create the annotated
#                 `vX.Y.Z` tag and push it, which is what triggers CI.
#
# Usage:
#   scripts/release.sh <version>                 # prepare the bump (branch + commit)
#   scripts/release.sh <version> --push          # ...also push branch + open a PR
#   scripts/release.sh <version> --tag           # tag main + push (run after merge)
#
# Options:
#   --tag           Tag phase: create + push vX.Y.Z (run on `main` after the bump merges).
#   --push          Prepare phase: push the release branch and open a PR (needs `gh`).
#   --no-branch     Prepare phase: bump on the current branch instead of release/vX.Y.Z.
#   --skip-verify   Skip the fmt/clippy/build/test gates (faster; not recommended).
#   -y, --yes       Don't prompt before pushing the tag.
#   -h, --help      Show this help.
#
# <version> is a bare semver like `0.1.1` — the `v` prefix is added for the tag.
#
# Requires: bash, git, cargo, perl on PATH (plus `gh` for --push).

set -euo pipefail

# ---------------------------------------------------------------------------
# Args + helpers
# ---------------------------------------------------------------------------

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; $d'; }
die()   { echo "error: $*" >&2; exit 1; }
note()  { echo "==> $*"; }

VERSION=""
DO_TAG=0
DO_PUSH=0
NO_BRANCH=0
SKIP_VERIFY=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)          DO_TAG=1 ;;
        --push)         DO_PUSH=1 ;;
        --no-branch)    NO_BRANCH=1 ;;
        --skip-verify)  SKIP_VERIFY=1 ;;
        -y|--yes)       ASSUME_YES=1 ;;
        -h|--help)      usage; exit 0 ;;
        -*)             die "unknown option: $1 (try --help)" ;;
        *)
            [[ -z "$VERSION" ]] || die "unexpected extra argument: $1"
            VERSION="$1"
            ;;
    esac
    shift
done

[[ -n "$VERSION" ]] || { usage; exit 64; }
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.+][0-9A-Za-z.-]+)?$ ]] \
    || die "version '$VERSION' is not a bare semver like 0.1.1 (no leading 'v')"

TAG="v$VERSION"

for tool in git cargo perl; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool not found on PATH: $tool"
done

# Resolve workspace root from this script's location (scripts/release.sh).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_MANIFEST="$ROOT/Cargo.toml"
SYS_MANIFEST="$ROOT/crates/x3f-sys/Cargo.toml"
cd "$ROOT"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git work tree"

# Current workspace version lives on the only line that starts with `version =`
# (the [workspace.dependencies] pins are indented inside `{ ... }` tables).
CURRENT="$(perl -ne 'print $1 and exit if /^version = "([^"]+)"/' "$ROOT_MANIFEST")"
[[ -n "$CURRENT" ]] || die "could not read current version from $ROOT_MANIFEST"

# ---------------------------------------------------------------------------
# Shared preflight
# ---------------------------------------------------------------------------

require_clean_tree() {
    if ! git diff --quiet || ! git diff --cached --quiet; then
        die "working tree is dirty — commit or stash first"
    fi
}

run_verify() {
    if [[ "$SKIP_VERIFY" -eq 1 ]]; then
        note "skipping verification gates (--skip-verify)"
        return
    fi
    note "cargo fmt --all --check";                         cargo fmt --all --check
    note "cargo clippy --workspace --all-targets";          cargo clippy --workspace --all-targets -- -D warnings
    note "cargo build --workspace --all-targets (refreshes Cargo.lock)"
    cargo build --workspace --all-targets
    note "cargo test --workspace";                          cargo test --workspace
}

confirm() {
    # confirm "<prompt>" — returns 0 to proceed.
    [[ "$ASSUME_YES" -eq 1 ]] && return 0
    [[ -t 0 ]] || die "refusing to proceed non-interactively without --yes: $1"
    local ans
    read -r -p "$1 [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]]
}

# ---------------------------------------------------------------------------
# Tag phase
# ---------------------------------------------------------------------------

if [[ "$DO_TAG" -eq 1 ]]; then
    require_clean_tree

    [[ "$CURRENT" == "$VERSION" ]] \
        || die "Cargo.toml is at $CURRENT, not $VERSION — has the version-bump PR merged into this branch?"

    git rev-parse -q --verify "refs/tags/$TAG" >/dev/null \
        && die "tag $TAG already exists locally"
    if git ls-remote --exit-code --tags origin "$TAG" >/dev/null 2>&1; then
        die "tag $TAG already exists on origin"
    fi

    BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    [[ "$BRANCH" == "main" ]] \
        || die "refusing to tag from '$BRANCH' — checkout an up-to-date main first"

    note "creating annotated tag $TAG"
    git tag -a "$TAG" -m "Release $TAG"

    if confirm "Push $TAG to origin? This triggers the CI release."; then
        note "git push origin $TAG"
        git push origin "$TAG"
        echo
        echo "✔ pushed $TAG — watch the 'publish release' job in GitHub Actions."
    else
        echo
        echo "Tag created locally but not pushed. Push it when ready with:"
        echo "    git push origin $TAG"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Prepare phase (version bump)
# ---------------------------------------------------------------------------

[[ "$CURRENT" != "$VERSION" ]] \
    || die "version is already $VERSION in $ROOT_MANIFEST (nothing to bump)"

require_clean_tree

git rev-parse -q --verify "refs/tags/$TAG" >/dev/null \
    && die "tag $TAG already exists — pick a new version"
if git ls-remote --exit-code --tags origin "$TAG" >/dev/null 2>&1; then
    die "tag $TAG already exists on origin — pick a new version"
fi

note "bumping $CURRENT -> $VERSION"

if [[ "$NO_BRANCH" -eq 0 ]]; then
    BRANCH="release/$TAG"
    [[ "$(git rev-parse --abbrev-ref HEAD)" == "main" ]] \
        || die "refusing to create $BRANCH from a non-main branch — checkout main first, or use --no-branch intentionally"
    if [[ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]]; then
        note "git checkout -b $BRANCH"
        git checkout -b "$BRANCH"
    fi
fi

# Bump every place the version is written by hand:
#   - [workspace.package] version           (root Cargo.toml)
#   - [workspace.dependencies] x3f-* pins    (root Cargo.toml)
#   - crates/x3f-sys/Cargo.toml version      (the only crate not on version.workspace)
# The other crates inherit via `version.workspace = true`. perl -i is used for
# identical behavior on macOS (BSD) and Linux (GNU) hosts. \Q…\E quotes the dots.
CUR_V="$CURRENT" NEW_V="$VERSION" perl -i -pe \
    's/^version = "\Q$ENV{CUR_V}\E"$/version = "$ENV{NEW_V}"/' "$ROOT_MANIFEST"
CUR_V="$CURRENT" NEW_V="$VERSION" perl -i -pe \
    's/, version = "\Q$ENV{CUR_V}\E" \}/, version = "$ENV{NEW_V}" }/g' "$ROOT_MANIFEST"
CUR_V="$CURRENT" NEW_V="$VERSION" perl -i -pe \
    's/^version = "\Q$ENV{CUR_V}\E"$/version = "$ENV{NEW_V}"/' "$SYS_MANIFEST"

# Fail loudly if any manifest still carries the old version where we expected a bump.
grep -q "^version = \"$VERSION\"" "$ROOT_MANIFEST" || die "failed to bump $ROOT_MANIFEST"
grep -q "^version = \"$VERSION\"" "$SYS_MANIFEST"  || die "failed to bump $SYS_MANIFEST"

echo
note "version changes:"
git --no-pager diff -- "$ROOT_MANIFEST" "$SYS_MANIFEST"
echo

run_verify

note "committing version bump"
# Cargo.lock is gitignored in this repo, so only the manifests are committed.
git add "$ROOT_MANIFEST" "$SYS_MANIFEST"
git commit -m "release: $TAG"

if [[ "$DO_PUSH" -eq 1 ]]; then
    CUR_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    note "git push -u origin $CUR_BRANCH"
    git push -u origin "$CUR_BRANCH"
    if command -v gh >/dev/null 2>&1; then
        note "opening PR"
        gh pr create --base main --head "$CUR_BRANCH" \
            --title "release: $TAG" \
            --body "Bump workspace version to \`$VERSION\`. After merge, tag the release with \`scripts/release.sh $VERSION --tag\`."
    else
        echo "note: gh not found — open the PR manually."
    fi
fi

cat <<EOF

✔ prepared release $TAG.

Next steps:
  1. Merge the version-bump $( [[ "$NO_BRANCH" -eq 0 ]] && echo "branch (PR)" || echo "commit" ) into main.
  2. On an up-to-date main, run:
         scripts/release.sh $VERSION --tag
     to create and push $TAG, which triggers the CI 'publish release' job.
EOF
