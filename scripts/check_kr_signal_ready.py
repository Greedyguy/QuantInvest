"""Check the latest published KR signal without loading a broker or pykrx."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signal_safety import validate_snapshot


def check_latest(directory, *, today=None):
    paths = sorted(Path(directory).glob('signal_kr_*.json'))
    if not paths:
        return False, None, 'No published KR signal snapshot'
    path = paths[-1]
    try:
        validate_snapshot(json.loads(path.read_text()), today=today, require_private_inputs=True)
    except (ValueError, TypeError, AttributeError, OSError) as exc:
        return False, path, str(exc)
    return True, path, 'Verified private inputs for the current KST decision date'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='reports/signals')
    parser.add_argument('--github-output')
    parser.add_argument('--require-ready', action='store_true')
    args = parser.parse_args()
    ready, path, reason = check_latest(args.directory)
    print(json.dumps(dict(ready=ready, snapshot=str(path) if path else None, reason=reason)))
    if args.github_output:
        with open(args.github_output, 'a') as output:
            output.write(f'ready={str(ready).lower()}\n')
    if args.require_ready and not ready:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
