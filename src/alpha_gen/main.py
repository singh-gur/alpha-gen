"""Main entry point for Alpha Gen application."""

from __future__ import annotations

import sys

from alpha_gen.cli.main import app


def main() -> None:
    """Entry point for the application."""
    try:
        sys.exit(app())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
