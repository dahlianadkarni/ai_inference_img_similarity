# Security Audit

Run these commands to check for vulnerabilities:

```bash
# Install security tools
pip install pip-audit safety

# Audit dependencies for known CVEs
pip-audit

# Check against Safety DB
safety check

# Update outdated packages
pip list --outdated
```

## Version Constraints

Using `>=` allows patch updates automatically, which usually include security fixes. However:

- **Pillow 10.2.0+**: Includes fixes for image parsing vulnerabilities
- **imagehash**: Stable, minimal attack surface
- **tqdm**: Display-only, low risk

## Best Practices

1. **Run audits regularly**: `pip-audit` before each phase
2. **Pin in production**: Use `pip freeze > requirements.lock` for deployments
3. **Minimal dependencies**: Only install what you need (we removed 20+ unnecessary packages)
4. **Virtual environment**: Always use `venv` to isolate dependencies

## Current Status (Phase 1a)

✅ Only 3 runtime dependencies
✅ All are actively maintained
✅ Pillow >=10.2.0 has recent security patches
✅ No database, API, or network dependencies yet (minimal attack surface)
