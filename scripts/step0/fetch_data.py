"""Download and verify the three public datasets behind Fig. 1 (step 0), then extract
what is used.

Outputs (git-ignored):
  data/external/pnas_murray2023/  Figshare 13072790 v1 (CC BY 4.0): Murray et al., PNAS
  2023
      extracted/derived/Derived/{Network,Embeddings/precedence}/ and
      extracted/additional/Additional/
  data/external/kddcup2003_hepth/ KDD Cup 2003 hep-th titles and abstracts (Schakel &
  Wilson's corpus)
      extracted/<year>/<paper>.abs
  data/external/blogcatalog/      BlogCatalog network (Tang & Liu 2009), the copy in the
      DeepWalk repository at a pinned commit: downloads/blogcatalog.mat (used as is)
Every file is checked against the SHA-256 recorded below before extraction; archive
members are
checked for absolute or parent-directory paths. Already-verified downloads are reused.
Run: python scripts/step0/fetch_data.py [--root data/external]
"""

import argparse
import hashlib
import tarfile
import urllib.request
import zipfile
from pathlib import Path

FILES = {
    "pnas_murray2023/downloads/additional/Additional.zip": (
        "https://ndownloader.figshare.com/files/25117640",
        "01e4d2bf0d3697c039e1afb84b345ff60ebd1b63a90d9a8f024a1cf265ca2425",
    ),
    "pnas_murray2023/downloads/derived/Derived.zip": (
        "https://ndownloader.figshare.com/files/25117865",
        "6224ecf080d4b9c4ced460594f4b23e7e164fc09e4a606a89b7b24c34ba39cc9",
    ),
    "kddcup2003_hepth/downloads/hep-th-abs.tar.gz": (
        "https://www.cs.cornell.edu/projects/kddcup/download/hep-th-abs.tar.gz",
        "d4f33c10ebae1d504328e62b5232ea67ac43ebaa1bd58fda13385ae29e610026",
    ),
    "blogcatalog/downloads/blogcatalog.mat": (
        "https://raw.githubusercontent.com/phanein/deepwalk/"
        "abe220b60dd08bb270aa08cb19f2050feb9b1664/example_graphs/blogcatalog.mat",
        "d4f4fb89ce1ccd4b7e2a183386c000773cc9362cc61f1be5b246a6d9c259da8f",
    ),
}
DERIVED_MEMBERS = ("Derived/Network/", "Derived/Embeddings/precedence/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(path: Path, url: str, expected: str) -> None:
    if path.exists() and sha256(path) == expected:
        print(f"verified {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual}")


def safe(name: str) -> bool:
    return not name.startswith("/") and ".." not in Path(name).parts


def extract_zip(archive: Path, target: Path, prefixes: tuple[str, ...] | None) -> None:
    with zipfile.ZipFile(archive) as z:
        members = [
            m
            for m in z.namelist()
            if safe(m)
            and "__MACOSX" not in m
            and not m.endswith(".DS_Store")
            and (prefixes is None or m.startswith(prefixes))
        ]
        z.extractall(target, members=members)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/external"))
    a = p.parse_args()
    for relative, (url, digest) in FILES.items():
        fetch(a.root / relative, url, digest)
    pnas = a.root / "pnas_murray2023"
    extract_zip(
        pnas / "downloads/derived/Derived.zip",
        pnas / "extracted/derived",
        DERIVED_MEMBERS,
    )
    extract_zip(
        pnas / "downloads/additional/Additional.zip",
        pnas / "extracted/additional",
        None,
    )
    hepth = a.root / "kddcup2003_hepth"
    with tarfile.open(hepth / "downloads/hep-th-abs.tar.gz") as t:
        members = [
            m for m in t.getmembers() if safe(m.name) and (m.isfile() or m.isdir())
        ]
        t.extractall(hepth / "extracted", members=members, filter="data")
    print("done")


if __name__ == "__main__":
    main()
