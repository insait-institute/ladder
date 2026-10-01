#!/usr/bin/env python3
"""Repository checks that need no Bazel, run by CI (.github/workflows/ci.yml).

    tools/ci/check_repo.py yaml     every YAML file parses (config tags such as !overwrite allowed)
    tools/ci/check_repo.py shell    every shell script parses (bash -n)
    tools/ci/check_repo.py links    every relative link and image in a Markdown file exists
"""
import os
import re
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP = {".git", "bzl-build", "node_modules"}


def files(*exts):
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in SKIP and not d.startswith("bzl-")]
        for f in fn:
            if f.endswith(exts):
                yield os.path.join(dp, f)


class Loader(yaml.SafeLoader):
    """Safe loading that accepts the configs' own tags (e.g. !overwrite) without constructing them."""


Loader.add_multi_constructor("!", lambda loader, suffix, node: None)


def check_yaml():
    bad = []
    for p in files(".yaml", ".yml", ".cff"):
        try:
            list(yaml.load_all(open(p), Loader=Loader))
        except yaml.YAMLError as e:
            bad.append(f"{os.path.relpath(p, ROOT)}: {str(e).splitlines()[0]}")
    return bad


def check_shell():
    bad = []
    for p in files(".sh"):
        r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
        if r.returncode:
            bad.append(f"{os.path.relpath(p, ROOT)}: {r.stderr.strip()}")
    return bad


LINK = re.compile(r"\]\(([^)\s]+)\)|\bsrc=\"([^\"]+)\"|\bhref=\"([^\"]+)\"")


def check_links():
    bad = []
    for p in files(".md"):
        text = re.sub(r"```.*?```", "", open(p).read(), flags=re.S)  # code blocks are not links
        for m in LINK.finditer(text):
            target = next(g for g in m.groups() if g)
            if re.match(r"^(https?:|mailto:|#)", target):
                continue
            path = target.split("#")[0]
            full = os.path.join(ROOT, path.lstrip("/")) if path.startswith("/") else os.path.join(os.path.dirname(p), path)
            if path and not os.path.exists(full):
                bad.append(f"{os.path.relpath(p, ROOT)}: {target}")
    return bad


def main():
    what = sys.argv[1]
    bad = {"yaml": check_yaml, "shell": check_shell, "links": check_links}[what]()
    for b in bad:
        print("FAIL", b)
    print(f"{what}: {len(bad)} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
