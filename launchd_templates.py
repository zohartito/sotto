"""Render sealed LaunchAgent templates for the deploying user's home."""
from pathlib import Path
import argparse
import os
import plistlib
import tempfile


def render(template: bytes, *, home: Path, ffmpeg: str = '/opt/homebrew/bin/ffmpeg') -> bytes:
    if not home.is_absolute() or not Path(ffmpeg).is_absolute():
        raise ValueError('home and FFmpeg paths must be absolute')
    def replace(value):
        if isinstance(value, str):
            return value.replace('__SOTTO_HOME__', str(home)).replace('__SOTTO_FFMPEG__', ffmpeg)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value
    return plistlib.dumps(replace(plistlib.loads(template)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = render(args.template.read_bytes(), home=Path.home(),
                  ffmpeg=os.environ.get('SOTTO_FFMPEG', '/opt/homebrew/bin/ffmpeg'))
    fd, temporary = tempfile.mkstemp(dir=args.output.parent, prefix='.sotto-plist-')
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
