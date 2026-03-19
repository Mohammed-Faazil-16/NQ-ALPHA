import os
from pathlib import Path


IGNORED_DIRS = {"__pycache__", "pycache", ".git", ".venv", "venv", "node_modules"}
ALLOWED_EXT = {".py", ".txt", ".md", ".env", ".json", ".yaml", ".yml"}


def is_ignored_dir(path: Path) -> bool:
    # check each path part case-insensitively
    return any(part.lower() in IGNORED_DIRS for part in path.parts)


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXT


def build_structure(root: Path) -> str:
    lines = []
    root_name = root.name
    lines.append(f"{root_name}/")

    # Walk deterministically by sorting directory names and filenames
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        try:
            rel = p.relative_to(root)
        except Exception:
            rel = Path('.')

        if is_ignored_dir(rel):
            dirnames[:] = []
            continue

        dirnames[:] = sorted(d for d in dirnames if d.lower() not in IGNORED_DIRS)
        filenames = sorted(filenames)

        # compute indentation: top-level children (under root) have no indent
        if str(rel) != '.':
            indent_level = len(rel.parts)
            indent = ' ' * 4 * indent_level
            lines.append(f"{indent}{rel.name}/")

        # list allowed files in this directory
        files = [f for f in filenames if Path(f).suffix.lower() in ALLOWED_EXT and f != 'allcode.txt']
        if files:
            file_indent = ' ' * 4 * (len(rel.parts) + 1)
            for fn in files:
                lines.append(f"{file_indent}{fn}")

    return "\n".join(lines)


def write_file_contents(root: Path, out_path: Path):
    with out_path.open("w", encoding="utf-8") as out:
        out.write("PROJECT STRUCTURE\n")
        out.write("==================\n")
        out.write(build_structure(root))
        out.write("\n\nFILE CONTENTS:\n")
        out.write("==============\n")

        for dirpath, dirnames, filenames in os.walk(root):
            p = Path(dirpath)
            if is_ignored_dir(p.relative_to(root)):
                dirnames[:] = []
                continue

            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]

            for fname in sorted(filenames):
                if fname == out_path.name:
                    continue
                fpath = p / fname
                if not is_text_file(fpath):
                    continue

                rel = fpath.relative_to(root)
                out.write('\n' + '=' * 80 + '\n')
                out.write(f"FILE: {rel.as_posix()}\n")
                out.write('\n')

                try:
                    with fpath.open('r', encoding='utf-8', errors='replace') as fh:
                        out.write(fh.read())
                except Exception as e:
                    out.write(f"Could not read {rel.as_posix()}: {e}\n")


def main():
    root = Path.cwd()
    out = root / 'allcode.txt'
    write_file_contents(root, out)
    print('Repository snapshot exported to allcode.txt')


if __name__ == '__main__':
    main()
