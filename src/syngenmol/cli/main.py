"""Top-level ``syngenmol`` command."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="syngenmol", description="SynGenMol molecular-generation workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Import command implementations lazily so help and core workflows do not
    # import optional PED or GeoDiff dependencies.
    from . import generate, prepare_search_space, prepare_warmup, score, train

    for command in (prepare_search_space, prepare_warmup, train, generate, score):
        command.add_parser(subparsers)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
