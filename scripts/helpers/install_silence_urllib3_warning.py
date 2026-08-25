#!/usr/bin/env python3
"""
Install a sitecustomize.py in the active virtualenv to silence
urllib3's NotOpenSSLWarning (LibreSSL on macOS).

This affects ONLY the current Python environment.
"""

import os
import site
import textwrap

WARNING_FILTER = r"""
import warnings

# Silence urllib3 LibreSSL warning on macOS
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1.1.1\+.*"
)
"""


def main():
    site_packages = site.getsitepackages()

    if not site_packages:
        raise RuntimeError("Could not locate site-packages directory")

    target_dir = site_packages[0]
    target_file = os.path.join(target_dir, "sitecustomize.py")

    content = textwrap.dedent(WARNING_FILTER).lstrip()

    if os.path.exists(target_file):
        with open(target_file, encoding="utf-8") as f:
            existing = f.read()

        if "urllib3 v2 only supports OpenSSL" in existing:
            print(f"✔ urllib3 warning already silenced in:\n  {target_file}")
            return

        print(f"⚠ Updating existing sitecustomize.py:\n  {target_file}")
        content = existing.rstrip() + "\n\n" + content

    else:
        print(f"➕ Creating sitecustomize.py:\n  {target_file}")

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Done. Restart Python to apply the change.")


if __name__ == "__main__":
    main()
