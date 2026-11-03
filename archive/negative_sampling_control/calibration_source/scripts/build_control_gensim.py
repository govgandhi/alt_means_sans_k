"""Build Gensim 4.4.0 with the two BLAS exception declarations corrected.

Inputs: SHA-256-verified official source archive and pinned build/runtime packages.
Outputs: a compiled Word2Vec extension and build_manifest.json; optionally install
into the active virtual environment. No system packages are modified.
Run with Python 3.13 after installing both negative_sampling_control requirements
files: python3 scripts/build_control_gensim.py --install
The patch matches the declaration fix proposed in Gensim PR #3668. It does not
repair integer-token sampling; that separate intervention stays in the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildConfig:
    work_dir: Path = Path(__file__).resolve().parents[1] / ".cache/gensim-control-build"
    url: str = (
        "https://files.pythonhosted.org/packages/1a/80/"
        "fe9d2e1ace968041814dbcfce4e8499a643a36c41267fa4b6c4f54cce420/"
        "gensim-4.4.0.tar.gz"
    )
    sha256: str = "a3f5b626da5518e79a479140361c663089fe7998df8ba52d56e1ded71ac5bdf5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(config: BuildConfig, install: bool) -> None:
    if install and sys.prefix == sys.base_prefix:
        raise RuntimeError("Installation is allowed only in a virtual environment.")
    work = config.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "gensim-4.4.0.tar.gz"
    if not archive.exists():
        with urllib.request.urlopen(config.url, timeout=60) as response:
            archive.write_bytes(response.read())
    if sha256(archive) != config.sha256:
        raise ValueError("Official source archive checksum mismatch.")
    source = work / "gensim-4.4.0"
    if not source.exists():
        with tarfile.open(archive) as handle:
            handle.extractall(work, filter="data")
    declarations = source / "gensim/models/word2vec_inner.pxd"
    with tarfile.open(archive) as handle:
        original_inputs = {
            name: handle.extractfile(f"gensim-4.4.0/gensim/models/{name}").read()
            for name in ("word2vec_inner.pyx", "word2vec_inner.pxd", "voidptr.h")
        }
    for name in ("word2vec_inner.pyx", "voidptr.h"):
        if (source / "gensim/models" / name).read_bytes() != original_inputs[name]:
            raise ValueError(
                f"Cached compilation input differs from official source: {name}"
            )
    before = original_inputs["word2vec_inner.pxd"].decode()
    after = before
    for name in ("sdot_ptr", "dsdot_ptr"):
        matching = [line for line in after.splitlines() if f"(*{name})" in line]
        if len(matching) != 1:
            raise ValueError(f"Unexpected declaration for {name}")
        line = matching[0]
        if "except -1 nogil" not in line and "noexcept nogil" not in line:
            raise ValueError(f"Unexpected exception specification for {name}")
        after = after.replace(line, line.replace("except -1 nogil", "noexcept nogil"))
    if declarations.read_text() not in (before, after):
        raise ValueError(
            "Cached declarations contain changes beyond the specified patch."
        )
    declarations.write_text(after)
    # The sdist lacks fastss.pyx, so build only the extension used by this study.
    # Its pyx and original pxd match those supplied in the official binary wheel.
    setup = source / "build_control_extension.py"
    setup.write_text(
        "from setuptools import Extension, setup\n"
        "from Cython.Build import cythonize\n"
        "import numpy\n"
        "setup(name='gensim-word2vec-blas-fix', ext_modules=cythonize([Extension(\n"
        "    'gensim.models.word2vec_inner', ['gensim/models/word2vec_inner.pyx'],\n"
        "    include_dirs=[numpy.get_include(), 'gensim/models'])],\n"
        "    force=True, language_level=3))\n"
    )
    library = work / "extension"
    subprocess.run(
        [
            sys.executable,
            str(setup),
            "build_ext",
            "--build-lib",
            str(library),
            "--build-temp",
            str(work / "objects"),
        ],
        cwd=source,
        check=True,
    )
    (extension,) = (library / "gensim/models").glob("word2vec_inner*.so")
    manifest = {
        "source_url": config.url,
        "source_sha256": config.sha256,
        "original_compile_inputs_sha256": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in original_inputs.items()
        },
        "patch": "sdot_ptr and dsdot_ptr: except -1 nogil -> noexcept nogil",
        "patched_declarations_sha256": sha256(declarations),
        "generated_word2vec_c_sha256": sha256(
            source / "gensim/models/word2vec_inner.c"
        ),
        "extension": extension.name,
        "extension_sha256": sha256(extension),
        "python": sys.version,
        "build_packages": {
            name: importlib.metadata.version(name)
            for name in ("Cython", "setuptools", "wheel", "pip", "numpy")
        },
    }
    (work / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if install:
        package = importlib.metadata.distribution("gensim")
        if package.version != "4.4.0":
            raise ValueError("This patch is restricted to Gensim 4.4.0.")
        destination = Path(package.locate_file("gensim/models"))
        if not destination.resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(
                "Refusing to patch Gensim outside this virtual environment."
            )
        if not (destination / extension.name).is_file():
            raise ValueError("Built extension does not match the installed Python ABI.")
        for artifact in (extension, declarations):
            target = destination / artifact.name
            backup = work / "original_installed" / artifact.name
            backup.parent.mkdir(exist_ok=True)
            if not backup.exists():
                shutil.copy2(target, backup)
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(artifact, temporary)
            temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=BuildConfig.work_dir)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    build(BuildConfig(work_dir=args.work_dir), args.install)


if __name__ == "__main__":
    main()
