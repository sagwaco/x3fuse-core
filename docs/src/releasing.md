# Releasing

Releases are **published by CI**. The [`release` job in
`.github/workflows/ci.yml`](https://github.com/sagwaco/x3fuse-core/blob/master/.github/workflows/ci.yml)
runs only on pushes of a tag matching `v*`; it waits for the `lint`,
`build`, `ios-xcframework`, and `wasm32` jobs, then calls `gh release
create --generate-notes` and attaches every built artifact:

- per-platform `x3f_extract` CLI binaries (Linux x86_64, macOS arm64, macOS x86_64),
- the iOS `X3F.xcframework.zip`,
- the `wasm32-wasip1` and `wasm32-unknown-unknown` bundles (`libx3f.a` + `x3f.wasm` + `x3f.h`).

**Pushing the tag is the only trigger.** Running the workflow manually
(`workflow_dispatch`) builds the artifacts but does *not* publish a
release — the `release` job is gated on `refs/tags/v*`. Release notes are
generated from the commits since the previous tag, so there is no
`CHANGELOG` to maintain.

## Where the version lives

The version is set once in the workspace and inherited everywhere else:

| Location | How it's set |
| --- | --- |
| Root `Cargo.toml` → `[workspace.package] version` | source of truth |
| Root `Cargo.toml` → `[workspace.dependencies]` `x3f-sys` / `x3f-core` pins | bumped to match |
| `crates/x3f-sys/Cargo.toml` → `version` | **hardcoded** — the one crate not on `version.workspace` |
| `crates/x3f-core`, `crates/x3f-cli`, `crates/x3f-ffi-c` | `version.workspace = true` (inherited) |

`Cargo.lock` is `.gitignore`d in this repo, so a version-bump commit
touches only the two `Cargo.toml` files above.

## Cutting a release with the script

[`scripts/release.sh`](https://github.com/sagwaco/x3fuse-core/blob/master/scripts/release.sh)
automates both halves of the flow. The version argument is a bare semver
(`0.1.1`); the script adds the `v` prefix for the tag.

**1. Prepare the version bump.** On a clean, up-to-date `main`:

```sh
scripts/release.sh 0.1.1
```

This creates a `release/v0.1.1` branch, bumps every manifest above,
refreshes `Cargo.lock`, runs the CI gates locally (`cargo fmt --check`,
`clippy`, `build`, `test`), and commits `release: v0.1.1`. Add `--push`
to also push the branch and open a PR (needs the `gh` CLI):

```sh
scripts/release.sh 0.1.1 --push
```

**2. Merge** the bump PR into `main` (the repo merges through PRs).

**3. Tag the release.** On an up-to-date `main`:

```sh
scripts/release.sh 0.1.1 --tag
```

This verifies `main` is actually at `0.1.1`, creates the annotated tag
`v0.1.1`, and — after a confirmation prompt — pushes it, which starts the
CI release. Watch the **publish release** job in GitHub Actions; when it
finishes, the release is live on the
[Releases page](https://github.com/sagwaco/x3fuse-core/releases) with all
artifacts attached.

### Flags

| Flag | Effect |
| --- | --- |
| `--push` | Prepare phase: push the branch and open a PR. |
| `--tag` | Tag phase: create + push `vX.Y.Z` (run after the bump merges). |
| `--no-branch` | Bump on the current branch instead of creating `release/vX.Y.Z` from `main`. |
| `--skip-verify` | Skip the fmt/clippy/build/test gates (faster; not recommended). |
| `-y`, `--yes` | Don't prompt before pushing the tag. |

The script refuses to run on a dirty tree, rejects a version with a
leading `v`, aborts if the tag already exists locally or on `origin`,
creates release branches only from `main`, and tags only from `main`.

## Cutting a release manually

The script is just the steps below; do them by hand if you prefer.

```sh
# 1. Bump the version in the three hand-written spots:
#      - root Cargo.toml [workspace.package] version
#      - root Cargo.toml [workspace.dependencies] x3f-sys / x3f-core pins
#      - crates/x3f-sys/Cargo.toml version
$EDITOR Cargo.toml crates/x3f-sys/Cargo.toml

# 2. Refresh the lockfile and verify the workspace builds + passes CI gates.
cargo build --workspace --all-targets
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# 3. Commit the bump on a branch and open a PR.
git checkout -b release/v0.1.1
git commit -am "release: v0.1.1"
git push -u origin release/v0.1.1
# ...open + merge the PR into main...

# 4. On an up-to-date main, tag and push. The `v` prefix is required —
#    both the workflow trigger and the release job key on refs/tags/v*.
git checkout main && git pull
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

## Notes

- **Tag the merged commit.** The tag should point at the version-bump
  commit on `main`, not at a feature branch. `git describe --tags` shows
  how far `main` is past the last release; those commits become the
  auto-generated notes.
- **Don't reuse a tag.** Once pushed, a `vX.Y.Z` tag is the published
  release. To fix a bad release, cut the next patch version rather than
  retagging.
