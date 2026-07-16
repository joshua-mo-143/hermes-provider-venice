"""Compatibility CLI for installing the Venice provider into Hermes.

Current Hermes releases can discover packaged plugins through entry points.
For older releases, or when this package is installed outside Hermes's Python
environment, this command copies the plugin into the user plugin directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PLUGIN_NAME = "venice"


def _hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".hermes"


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def install(*, force: bool = False) -> Path:
    src = _package_dir()
    dest = _hermes_home() / "plugins" / "model-providers" / PLUGIN_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if not force:
            raise FileExistsError(
                f"Already installed at {dest}\nRe-run with --force to replace it."
            )
        shutil.rmtree(dest)

    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "__main__.py",
        ),
    )
    return dest


def status() -> None:
    dest = _hermes_home() / "plugins" / "model-providers" / PLUGIN_NAME
    init = dest / "__init__.py"
    if init.is_file():
        print(f"installed: {dest}")
    else:
        print(f"not installed (expected {dest})")
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hermes-provider-venice",
        description="Install the Venice AI provider into Hermes Agent",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    install_p = sub.add_parser(
        "install",
        help="Copy the provider into $HERMES_HOME/plugins/model-providers/venice/",
    )
    install_p.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing install",
    )

    sub.add_parser("status", help="Check whether the provider is installed")

    args = parser.parse_args(argv)

    if args.cmd == "install":
        try:
            dest = install(force=args.force)
        except FileExistsError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
        print(f"Installed Venice provider → {dest}")
        print()
        print("Next steps:")
        print("  1. Add VENICE_API_KEY to ~/.hermes/.env")
        print("     (create a key at https://venice.ai/settings/api)")
        print("  2. hermes -z 'hello' --provider venice -m zai-org-glm-5-2")
        print("     or set model.provider: venice in config.yaml")
        return

    if args.cmd == "status":
        status()


if __name__ == "__main__":
    main()
