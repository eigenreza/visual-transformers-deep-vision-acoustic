"""Replace the machine-specific roots in copied result files by placeholders.

Usage: python scripts/redact_paths.py --paths PATHS <directory> [<directory> ...]
The roots are the entries of the paths file (``${NAME}`` values are read from
the environment). In every JSON, YAML, CSV and text file below the given
directories each root is replaced by a placeholder such as ``<project_dir>``.
``scripts/verify_manifest.py --paths PATHS`` puts the roots back for the
comparison with the recorded hashes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from echolabel.config import expand_paths, load_yaml
from echolabel.manifest import replace_roots

SUFFIXES = {".json", ".yml", ".yaml", ".csv", ".txt"}


def main(arguments: list[str]) -> int:
    i = arguments.index("--paths")
    roots = {k: str(v) for k, v in expand_paths(load_yaml(arguments[i + 1])).items() if v}
    n_changed = 0
    for root in arguments[:i] + arguments[i + 2 :]:
        for path in sorted(Path(root).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUFFIXES:
                continue
            text = path.read_bytes().decode("utf-8", errors="surrogateescape")
            new = replace_roots(text, roots)
            if new != text:
                path.write_bytes(new.encode("utf-8", errors="surrogateescape"))
                n_changed += 1
    print(f"{n_changed} files with placeholders written")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
