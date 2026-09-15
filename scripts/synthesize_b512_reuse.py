"""Reuse frozen captions locally; call SII only for explicit missing/invalid text."""
import argparse
import asyncio
import json
from pathlib import Path

from data_synthesis.config import DEFAULT_CONFIG, load_config
from data_synthesis.reuse_farm import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--acceptance', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.root or Path(config['state_root'])
    selection = args.selection or Path(config['prepared_selection'])
    acceptance = args.acceptance or root / 'user_acceptance.json'
    result = asyncio.run(run(root, config, selection, acceptance))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['state'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
