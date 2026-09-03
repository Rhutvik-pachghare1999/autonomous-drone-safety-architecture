# Security Notes

## Secret scan (2026-09-03)

A scan of tracked files and git history was run for common secret categories:
API keys, passwords, tokens, private keys/certificates, and Wi-Fi credentials.

**Result: no credentials found in tracked files or git history.**

### `.env`

- The repository root contains a local `.env` file.
- It is listed in `.gitignore` and has **never** been committed
  (`git log --all -- .env` returns nothing).
- Its contents are **filesystem paths and tool versions only**
  (for example `ROS_ROOT`, `ISAAC_SIM_ROOT`, `CC`, `PYTHONPATH`).
- It contains **no** API keys, tokens, passwords, or certificates.

### Rotation actions required

**None.** No credential was exposed, so nothing needs to be rotated.

If a real secret is ever added to `.env` in the future, remember:
removing a file from git does **not** revoke an already-pushed credential.
Rotate the credential at its source (provider console) as well.

## Template

`.env.example` documents the expected variables with placeholder values.
Copy it to `.env` and edit for your machine:

```bash
cp .env.example .env
```

## `.gitignore` coverage

The following secret-bearing patterns are ignored: `.env`, `*.key`, `*.pem`,
`*.secret`. Verify before committing anything new:

```bash
git check-ignore .env    # should print ".env"
git status               # confirm no .env / *.key / *.pem staged
```
