"""Create the public native-wheel checksum manifest from verified build artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from packaging.utils import parse_wheel_filename

LLAMA_COMMIT = "ad6c66839af3c5646fba8c6c2e2087a1e4e38948"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    wheels = []
    seen = set()
    for path in sorted(args.directory.glob("*.whl")):
        name, version, build, tags = parse_wheel_filename(path.name)
        if name != "confucius-r2t2-native" or str(version) != "0.2.0":
            raise ValueError(f"Unexpected distribution/version: {path.name}")
        with zipfile.ZipFile(path) as package:
            info = json.loads(package.read("r2t2_native/build-info.json"))
            if info["llama_commit"] != LLAMA_COMMIT or info["api_version"] != 1:
                raise ValueError(f"Wrong source/ABI in {path.name}")
            if any(n.lower().endswith(".exe") for n in package.namelist()):
                raise ValueError(f"Executable included in {path.name}")
        backend = "cuda" if info["cuda"] else "cpu"
        for tag in tags:
            key = (backend, tag.interpreter, tag.platform)
            if key in seen:
                raise ValueError(f"Ambiguous native wheel: {key}")
            seen.add(key)
        wheels.append({"filename": path.name, "backend": backend, "tags": sorted(map(str, tags)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    if not wheels:
        raise ValueError("No native wheels found")
    manifest = {"version": "0.2.0", "llama_commit": LLAMA_COMMIT, "wheels": wheels}
    output = args.directory / "native-wheels.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
