# Concourse CI/CD Pipeline

Automated testing, building, and deployment for Alpha Gen.

## Quick Start

```bash
# 1. Install fly CLI and cosign
brew install fly cosign  # macOS
# or download from your Concourse web UI

# 2. Login to Concourse
fly -t prod login -c https://your-concourse.com

# 3. Generate SSH deploy key
ssh-keygen -t ed25519 -f ~/.ssh/alpha-gen-deploy -C "concourse-ci"
# Add public key to GitHub: Settings > Deploy keys

# 4. Generate Cosign keys for image signing
just ci-generate-keys
# Follow prompts and save the password

# 5. Setup credentials
just ci-setup
vim ci/credentials.yml  # Fill in your values (including Cosign keys)

# 6. Set pipeline
just ci-set prod

# 7. Unpause and expose
just ci-unpause prod
just ci-expose prod
```

## Pipeline Jobs

| Job | Description | Trigger |
|-----|-------------|---------|
| `test` | Lint, type-check, and pytest | Every commit |
| `build-and-push` | Docker build, push, and sign | After tests pass |
| `verify-signature` | Verify image signatures | After build |
| `release` | Tagged releases with signing | Manual |

## Common Commands

```bash
# Setup
just ci-generate-keys      # Generate Cosign signing keys
just ci-setup               # Create credentials file

# Trigger jobs
just ci-test prod
just ci-build prod
just ci-release prod

# Watch jobs
just ci-watch-test prod
just ci-watch-build prod

# View status
just ci-status prod
just ci-builds prod

# Manage
just ci-pause prod
just ci-unpause prod
just ci-destroy prod

# Validate locally
just ci-validate
```

## Credentials

Required in `ci/credentials.yml`:

```yaml
# Git (SSH)
git_uri: git@github.com:your-org/alpha-gen.git
git_private_key: |
  -----BEGIN OPENSSH PRIVATE KEY-----
  ...
  -----END OPENSSH PRIVATE KEY-----

# Docker Registry
docker_repository: registry.example.com/org/alpha-gen
registry_username: user
registry_password: pass

# Cosign Image Signing
cosign_private_key: |
  -----BEGIN ENCRYPTED COSIGN PRIVATE KEY-----
  ...
  -----END ENCRYPTED COSIGN PRIVATE KEY-----
cosign_public_key: |
  -----BEGIN PUBLIC KEY-----
  ...
  -----END PUBLIC KEY-----
cosign_password: your-password
```

Generate Cosign keys with: `just ci-generate-keys`

See `credentials.yml.example` for details.

## Pipeline Flow

```
Push to main
    ↓
┌───────────────────┐
│ test              │
│ - ruff check      │
│ - ruff format     │
│ - basedpyright    │
│ - pytest --cov    │
└───────────────────┘
    ↓
┌───────────────────┐
│ build-and-push    │
│ - docker build    │
│ - push :latest    │
│ - push :commit    │
│ - sign images     │
└───────────────────┘
    ↓
┌───────────────────┐
│ verify-signature  │
│ - verify :latest  │
└───────────────────┘
```

## Quality Gates

- ✅ Zero linting violations (ruff)
- ✅ Proper formatting (ruff format)
- ⚠️ Type check warnings (basedpyright, non-blocking)
- ✅ Tests pass (pytest)
- ✅ Images signed with Cosign
- ✅ Signatures verified

## Verifying Signed Images

After the pipeline signs images, you can verify them locally:

```bash
# Export the public key
cat ci/keys/cosign.pub

# Verify an image
cosign verify --key ci/keys/cosign.pub registry.example.com/org/alpha-gen:latest

# Or verify with the public key inline
cosign verify --key <(echo "-----BEGIN PUBLIC KEY-----
...
-----END PUBLIC KEY-----") registry.example.com/org/alpha-gen:latest
```

## Resources

- [Concourse Docs](https://concourse-ci.org/docs.html)
- [Cosign Docs](https://docs.sigstore.dev/cosign/overview/)
- [Pipeline Config](pipeline.yml)
- [Credentials Example](credentials.yml.example)
