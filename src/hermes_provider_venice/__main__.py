"""Compatibility CLI for installing the Venice provider into Hermes.

Current Hermes releases can discover packaged plugins through entry points.
For older releases, or when this package is installed outside Hermes's Python
environment, this command copies the plugin into the user plugin directory.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

PLUGIN_NAME = "venice"
DISTRIBUTION_NAME = "hermes-provider-venice"


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


def _installed_version() -> str | None:
    try:
        return importlib.metadata.version(DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None


def update(*, pre: bool = False, source: str | None = None) -> None:
    """Upgrade this package via pip, then refresh any directory install.

    Runs ``pip install --upgrade`` for this distribution (or *source*, e.g. a
    ``git+https://…`` URL) in the current interpreter. When the compatibility
    installer has copied the plugin into ``$HERMES_HOME``, that copy is
    refreshed from the freshly upgraded files on disk so directory-plugin
    installs pick up the new version too.
    """
    target = source or DISTRIBUTION_NAME
    command = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if pre:
        command.append("--pre")
    command.append(target)

    print(f"Updating {DISTRIBUTION_NAME} (was {_installed_version() or 'unknown'})…")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not run pip. Install pip in this environment, or upgrade "
            "with the tool that installed this package (e.g. uv, pipx)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pip failed to upgrade {target} (exit {exc.returncode}).")

    dest = _hermes_home() / "plugins" / "model-providers" / PLUGIN_NAME
    if (dest / "__init__.py").is_file():
        refreshed = install(force=True)
        print(f"Refreshed directory install → {refreshed}")


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

    update_p = sub.add_parser(
        "update",
        help="Upgrade this package to the latest release via pip",
    )
    update_p.add_argument(
        "--pre",
        action="store_true",
        help="Include pre-release versions",
    )
    update_p.add_argument(
        "--source",
        metavar="SPEC",
        help=(
            "pip requirement to upgrade from instead of PyPI "
            "(e.g. 'git+https://github.com/joshua-mo/"
            "hermes-provider-venice.git@main')"
        ),
    )

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
        return

    if args.cmd == "update":
        try:
            update(pre=args.pre, source=args.source)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
        print()
        print("Update complete. Restart Hermes to load the new version.")
        return


if __name__ == "__main__":
    main()
