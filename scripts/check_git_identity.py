"""Reject commits that would be attributed to the wrong GitHub account."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


EXPECTED_NAME = "AnasNihal"
EXPECTED_EMAIL = "108085694+AnasNihal@users.noreply.github.com"
IDENTITY_PATTERN = re.compile(r"^(?P<name>.*) <(?P<email>[^>]+)> .*$")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _validate_identity(label: str, value: str) -> list[str]:
    match = IDENTITY_PATTERN.match(value)
    if not match:
        return [f"{label} identity has an invalid format: {value}"]
    errors: list[str] = []
    if match.group("name") != EXPECTED_NAME:
        errors.append(f"{label} name must be {EXPECTED_NAME}, got {match.group('name')}")
    if match.group("email") != EXPECTED_EMAIL:
        errors.append(f"{label} email must be {EXPECTED_EMAIL}, got {match.group('email')}")
    return errors


def check_current_identity() -> list[str]:
    errors = _validate_identity("author", _git("var", "GIT_AUTHOR_IDENT"))
    errors.extend(_validate_identity("committer", _git("var", "GIT_COMMITTER_IDENT")))
    return errors


def check_range(commit_range: str) -> list[str]:
    errors: list[str] = []
    commits = _git("log", "--format=%H%x00%an%x00%ae%x00%cn%x00%ce", commit_range).splitlines()
    for line in commits:
        commit, author_name, author_email, committer_name, committer_email = line.split("\x00")
        if author_name != EXPECTED_NAME or author_email != EXPECTED_EMAIL:
            errors.append(f"{commit[:8]} has invalid author {author_name} <{author_email}>")
        if committer_name != EXPECTED_NAME or committer_email != EXPECTED_EMAIL:
            errors.append(f"{commit[:8]} has invalid committer {committer_name} <{committer_email}>")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--range", dest="commit_range", help="Validate commits in a Git revision range")
    args = parser.parse_args()
    try:
        errors = check_range(args.commit_range) if args.commit_range else check_current_identity()
    except subprocess.CalledProcessError as exc:
        print(f"Git identity check could not run: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("Commit blocked: Git identity does not match the repository owner.", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print("Configure the repository with:", file=sys.stderr)
        print(f'git config --local user.name "{EXPECTED_NAME}"', file=sys.stderr)
        print(f'git config --local user.email "{EXPECTED_EMAIL}"', file=sys.stderr)
        return 1
    print("Git identity verified: AnasNihal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
