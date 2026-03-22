"""Allow running as: python -m jit_search [serve|search]

Examples:
    python -m jit_search serve              # start the FastAPI server
    python -m jit_search search "query" ... # run a CLI search
    python -m jit_search                    # default: start the server
"""

from __future__ import annotations

import sys


def main() -> None:
    # Determine sub-command (default to "serve")
    if len(sys.argv) < 2 or sys.argv[1] in ("serve", "--help", "-h"):
        if len(sys.argv) >= 2 and sys.argv[1] == "serve":
            # Strip the "serve" sub-command so uvicorn doesn't see it
            sys.argv = [sys.argv[0]] + sys.argv[2:]

        if "--help" in sys.argv or "-h" in sys.argv:
            print("Usage: python -m jit_search [serve|search]")
            print()
            print("Sub-commands:")
            print("  serve   Start the FastAPI server (default)")
            print("  search  Run a CLI search (see: python -m jit_search search --help)")
            return

        import uvicorn

        uvicorn.run(
            "jit_search.server:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )

    elif sys.argv[1] == "search":
        # Forward remaining args to the CLI
        from jit_search.cli import main as cli_main

        cli_main(sys.argv[2:])

    else:
        print(f"Unknown sub-command: {sys.argv[1]}", file=sys.stderr)
        print("Usage: python -m jit_search [serve|search]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
