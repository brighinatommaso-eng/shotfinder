"""Audit the DLL imports of a built ShotFinder folder.

PyInstaller only bundles what it can see at build time. A DLL that is imported
by the bundle but lives outside ``_internal`` and outside a clean Windows
``System32`` (typically the Visual C++ redistributable) makes the release fail
on a machine without that redistributable installed.

Usage::

    python tools/check_runtime_deps.py dist\\ShotFinder
    python tools/check_runtime_deps.py dist\\ShotFinder --json runtime-deps.json

Exit code 0 means every import is bundled, an operating-system DLL, or a VC++
runtime that is bundled. Exit code 1 lists what is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pefile

# Visual C++ redistributable DLLs. Windows does not guarantee these; the bundle
# must carry every one it imports, even when the build machine has the
# redistributable installed in System32.
VC_RUNTIME_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^msvcp140(_\d+)?(_atomic_wait|_codecvt_ids)?\.dll$",
        r"^vcruntime140(_\d+)?\.dll$",
        r"^vcruntime140_threads\.dll$",
        r"^concrt140\.dll$",
        r"^vccorlib140\.dll$",
        r"^vcomp140\.dll$",
    )
)

# API set contracts are resolved by the loader on every supported Windows.
OS_NAME_PREFIXES = ("api-ms-win-", "ext-ms-")

PE_SUFFIXES = {".exe", ".dll", ".pyd"}


def is_vc_runtime(name: str) -> bool:
    return any(pattern.match(name) for pattern in VC_RUNTIME_PATTERNS)


def iter_pe_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in PE_SUFFIXES
    )


def imported_dlls(path: Path) -> set[str]:
    """Return every imported DLL basename, including delay-loaded imports."""
    pe = pefile.PE(str(path), fast_load=True)
    try:
        pe.parse_data_directories(
            directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
            ]
        )
        names: set[str] = set()
        for attribute in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
            for entry in getattr(pe, attribute, []) or []:
                raw = entry.dll
                if not raw:
                    continue
                name = raw.decode("ascii", "replace") if isinstance(raw, bytes) else str(raw)
                names.add(name)
        return names
    finally:
        pe.close()


def audit(root: Path, system32: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise SystemExit(f"bundle folder not found: {root}")
    if not (root / "ShotFinder.exe").is_file():
        raise SystemExit(f"ShotFinder.exe not found in {root}")
    if not system32.is_dir():
        raise SystemExit(f"System32 not found: {system32}")

    files = iter_pe_files(root)
    bundled = {path.name.lower() for path in files}
    # The local System32 is not a clean Windows: the build machine may have the
    # VC++ redistributable installed. Remove those names from the OS baseline so
    # they are judged by the bundling rule instead.
    system = {
        path.name.lower()
        for path in system32.iterdir()
        if path.is_file() and not is_vc_runtime(path.name)
    }

    imports: set[str] = set()
    unparsed: list[dict[str, str]] = []
    for path in files:
        try:
            imports.update(imported_dlls(path))
        except Exception as exc:  # noqa: BLE001 - reported as a build failure
            unparsed.append({"file": str(path.relative_to(root)), "error": str(exc)})

    by_lower = {name.lower(): name for name in imports}
    missing: list[str] = []
    vc_required: list[str] = []
    vc_missing: list[str] = []
    os_imports: list[str] = []
    for lower, name in sorted(by_lower.items()):
        if lower in bundled:
            if is_vc_runtime(name):
                vc_required.append(name)
            continue
        if is_vc_runtime(name):
            vc_required.append(name)
            vc_missing.append(name)
            continue
        if lower in system or lower.startswith(OS_NAME_PREFIXES):
            os_imports.append(name)
            continue
        missing.append(name)

    ok = not missing and not vc_missing and not unparsed
    return {
        "ok": ok,
        "bundle": str(root.resolve()),
        "system32": str(system32.resolve()),
        "scanned_files": len(files),
        "unique_imports": len(imports),
        "bundled_dlls": len(bundled),
        "vc_runtime_required": sorted(vc_required, key=str.lower),
        "vc_runtime_missing": sorted(vc_missing, key=str.lower),
        "os_imports": sorted(os_imports, key=str.lower),
        "missing": sorted(missing, key=str.lower),
        "unparsed": unparsed,
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"Bundle: {report['bundle']}")
    print(f"System32 baseline: {report['system32']}")
    print(
        f"Scanned {report['scanned_files']} PE file(s); "
        f"{report['unique_imports']} unique import(s); "
        f"{report['bundled_dlls']} bundled DLL(s)."
    )
    print()
    print("Visual C++ runtime imports:")
    if report["vc_runtime_required"]:
        for name in report["vc_runtime_required"]:
            state = "MISSING" if name in report["vc_runtime_missing"] else "bundled"
            print(f"  {name}: {state}")
    else:
        print("  (none)")
    print()
    print("Imports satisfied by System32:")
    for name in report["os_imports"]:
        print(f"  {name}")
    if not report["os_imports"]:
        print("  (none)")
    print()
    if report["missing"]:
        print("Imports NOT bundled and NOT in System32:")
        for name in report["missing"]:
            print(f"  {name}")
    else:
        print("Imports NOT bundled and NOT in System32: (none)")
    if report["unparsed"]:
        print()
        print("Unparsable PE files:")
        for item in report["unparsed"]:
            print(f"  {item['file']}: {item['error']}")
    print()
    print("OK" if report["ok"] else "FAILED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="folder containing ShotFinder.exe")
    parser.add_argument(
        "--system32",
        type=Path,
        default=Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
        help="System32 of the reference Windows installation",
    )
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)

    report = audit(args.bundle, args.system32)
    print_summary(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
