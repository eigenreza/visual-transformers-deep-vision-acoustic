"""Check the output files of stage directories against the hashes in their manifests.

Usage: python scripts/verify_manifest.py [--strict] [--paths PATHS] <directory> [<directory> ...]
Every directory that holds a ``manifest.json`` below the given ones is checked.
A recorded output that is absent is listed as not present. A copy in which
machine-specific roots are replaced by placeholders is compared after the
placeholders are replaced by the roots of the paths file given with
``--paths``; without that file it is counted as not comparable. The exit
status is 1 when an output has changed, and with ``--strict`` also when one
is absent or not comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

from echolabel.config import expand_paths, load_yaml
from echolabel.manifest import MANIFEST_NAME, verify_outputs


def main(arguments: list[str]) -> int:
    strict = "--strict" in arguments
    arguments = [a for a in arguments if a != "--strict"]
    roots = None
    if "--paths" in arguments:
        i = arguments.index("--paths")
        roots = {k: str(v) for k, v in expand_paths(load_yaml(arguments[i + 1])).items() if v}
        arguments = arguments[:i] + arguments[i + 2 :]
    changed = absent = placeholder = 0
    for root in arguments:
        for manifest in sorted(Path(root).rglob(MANIFEST_NAME)):
            status = verify_outputs(manifest.parent, roots)
            n_ok = sum(s == "ok" for s in status.values())
            n_absent = sum(s == "missing" for s in status.values())
            n_placeholder = sum(s == "placeholder" for s in status.values())
            bad = [name for name, s in status.items() if s == "changed"]
            changed += len(bad)
            absent += n_absent
            placeholder += n_placeholder
            line = f"{manifest.parent.as_posix()}: {n_ok} of {len(status)} outputs match"
            line += f", {n_placeholder} with placeholders not compared" if n_placeholder else ""
            print(line + (f", {n_absent} not present" if n_absent else ""))
            for name in bad:
                print(f"  changed: {name}")
            if strict:
                for name, s in status.items():
                    if s in ("missing", "placeholder"):
                        print(f"  {s}: {name}")
    print(f"{changed} changed, {placeholder} with placeholders not compared, {absent} not present")
    return 1 if changed or (strict and (absent or placeholder)) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
