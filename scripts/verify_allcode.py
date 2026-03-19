from pathlib import Path
import os

IGNORED_DIRS = {"__pycache__", "pycache", ".git", ".venv", "venv", "node_modules"}
ALLOWED_EXT = {".py", ".txt", ".md", ".env", ".json", ".yaml", ".yml"}


def is_ignored_dir(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def collect_text_files(root: Path):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        if is_ignored_dir(p.relative_to(root)):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for fn in sorted(filenames):
            if fn == 'allcode.txt':
                continue
            fpath = p / fn
            if fpath.suffix.lower() in ALLOWED_EXT:
                files.append(fpath.relative_to(root).as_posix())
    return files


def verify(root: Path):
    out_path = root / 'allcode.txt'
    if not out_path.exists():
        print('allcode.txt not found')
        return 2

    text_files = collect_text_files(root)

    content = out_path.read_text(encoding='utf-8', errors='replace')

    missing = []
    for rel in text_files:
        if f'FILE: {rel}' not in content:
            missing.append(rel)

    if missing:
        print('Missing files in allcode.txt:')
        for m in missing:
            print(m)
        print(f"Total missing: {len(missing)}")
        return 1
    else:
        print('All text files are included in allcode.txt')
        print(f'Total files checked: {len(text_files)}')
        return 0


if __name__ == '__main__':
    import sys
    root = Path.cwd()
    rc = verify(root)
    sys.exit(rc)
