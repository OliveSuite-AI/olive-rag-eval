# Releasing olive-rag-eval

Repository: https://github.com/OliveSuite-AI/olive-rag-eval

Package: https://pypi.org/project/olive-rag-eval/

Intended PyPI owner: `sekharsambit25`. The account profile URL is not an upload
endpoint. The release workflow uploads to the standard production PyPI registry.

## One-time account configuration

1. Authenticate GitHub CLI using an account with repository write access:
   `gh auth login -h github.com`. Classic tokens used for HTTPS Git pushes need
   the `workflow` scope to add workflow files, in addition to repository access.
2. In the repository settings, create a GitHub Actions environment named `pypi`.
   Restrict deployment to `main` and configure required reviewers as appropriate.
3. Sign in to PyPI as `sekharsambit25` and open
   https://pypi.org/manage/account/publishing/ . For a new project, add a pending
   GitHub Trusted Publisher with these exact values:

   | Field | Value |
   |---|---|
   | PyPI project name | `olive-rag-eval` |
   | GitHub owner | `OliveSuite-AI` |
   | GitHub repository | `olive-rag-eval` |
   | Workflow filename | `python-wheels.yml` |
   | Environment | `pypi` |

   If the project already exists, configure its Trusted Publisher in project
   settings using the same repository, workflow and environment.

No long-lived PyPI token is needed. GitHub's publish job requests an OIDC token
only after the required build and test jobs pass. Do not commit registry tokens,
GitHub credentials or local credential configuration.

## Release procedure

1. Set the version in `[workspace.package]` in `Cargo.toml`, regenerate
   `Cargo.lock`, and update the changelog. Never reuse a published version for
   different artifacts.
2. Run the Rust checks, runtime tests, asset sync check and installed-wheel tests.
   Push to `main`; inspect every job in the `Python wheels` workflow. Its matrix
   builds macOS ARM64/x86_64, Linux ARM64/x86_64 and Windows x86_64 wheels and tests
   CPython 3.10/3.14 installations with pip and uv. The source archive is also
   rebuilt and installed outside the checkout.
3. Update README release/support statements to match observed CI results.
4. Trigger a release from `main`:

   ```sh
   gh workflow run python-wheels.yml --repo OliveSuite-AI/olive-rag-eval --ref main -f publish=true
   gh run list --repo OliveSuite-AI/olive-rag-eval --workflow python-wheels.yml
   ```

5. Approve the `pypi` environment deployment if reviewers are configured. Confirm
   that the publish job succeeds and PyPI lists the expected version, source
   archive and all five wheels. A GitHub push or green build alone does not prove
   publication.
6. Install the exact release from PyPI in a clean environment, run the packaged
   example, and then create the matching Git tag/release.

The workflow publishes only when manually dispatched with `publish=true` on
`main`. Pull requests and ordinary pushes run checks without uploading. If an
upload partially succeeds, inspect PyPI before retrying; published files are
immutable and cannot be replaced.
