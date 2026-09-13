"""Compatibility entry for explicitly requested historical Codex farm audits.

New work uses scripts.synthesize_image_text; this entry never starts new synthesis.
"""
import sys


def main():
    if len(sys.argv) == 1 or sys.argv[1] in {"--help", "-h"}:
        print(__doc__ + "\nCurrent entry: python -m scripts.synthesize_image_text --help")
        return
    if len(sys.argv) > 1 and sys.argv[1] in {"export", "recover-completed"}:
        from scripts.legacy.distill_b512_codex import main as historical
        return historical()
    raise SystemExit("Codex-only production has been retired. Use python -m scripts.synthesize_image_text --help. Historical tools are under scripts.legacy.")


if __name__ == "__main__":
    main()
