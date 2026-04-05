"""Patch transformers dependency_versions_check.py to skip version checks."""
import os
import site

sp = site.getsitepackages()[0]
dvp = os.path.join(sp, "transformers", "dependency_versions_check.py")

if not os.path.exists(dvp):
    print(f"File not found: {dvp}")
    exit(0)

with open(dvp) as f:
    content = f.read()

if "auto-patched" in content:
    print("Already patched")
    exit(0)

if "require_version_core" in content:
    content = content.replace(
        "require_version_core(deps[pkg])",
        "pass  # auto-patched"
    )
    with open(dvp, "w") as f:
        f.write(content)
    print(f"PATCHED: {dvp}")
else:
    print("No patch needed")
