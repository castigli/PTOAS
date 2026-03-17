#!/usr/bin/env python3

import argparse
import pathlib
import re
import sys


PROJECT_VERSION_RE = re.compile(
    r"(project\s*\(\s*ptoas\s+VERSION\s+)([0-9]+\.[0-9]+)(\s*\))"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the base PTOAS version in the top-level CMakeLists.txt."
    )
    parser.add_argument(
        "--cmake-file",
        default="CMakeLists.txt",
        help="Path to the top-level CMakeLists.txt file.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Released version to write back, e.g. v0.8 or 0.8.",
    )
    return parser.parse_args()


def normalize_version(version: str) -> str:
    normalized = version.strip()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    if not re.fullmatch(r"[0-9]+\.[0-9]+", normalized):
        raise ValueError(f"invalid PTOAS version '{version}'")
    return normalized


def update_base_version(cmake_file: pathlib.Path, version: str) -> bool:
    content = cmake_file.read_text(encoding="utf-8")

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{version}{match.group(3)}"

    updated, count = PROJECT_VERSION_RE.subn(repl, content, count=1)
    if count != 1:
        raise ValueError(
            f"could not find 'project(ptoas VERSION x.y)' in {cmake_file}"
        )
    if updated == content:
        return False
    cmake_file.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    version = normalize_version(args.version)
    cmake_file = pathlib.Path(args.cmake_file)
    update_base_version(cmake_file, version)
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
