"""Open the launcher with ``python -m autodrive_rl``.

The launcher is the project's front door: every mode — drive, train, clone,
benchmark — is a form in that window. On a machine with no display (a headless
box, or SSH without X forwarding) it falls back to the original behaviour, the
rule-based visual demo, so the old entry point still behaves sensibly.
"""

from __future__ import annotations

import sys


def main() -> None:
    from .launcher import main as launcher_main

    try:
        launcher_main([])
        return
    except SystemExit as exit_error:
        if exit_error.code in (0, None):
            return
    except Exception as error:  # noqa: BLE001 - deliberate last-resort fallback
        print(f"Launcher unavailable ({error}).", file=sys.stderr)

    print("Falling back to the rule-based demo.", file=sys.stderr)
    from .play import main as play_main

    play_main([])


if __name__ == "__main__":
    main()
