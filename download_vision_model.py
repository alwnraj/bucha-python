"""Download a pinned ADE20K SegFormer ONNX checkpoint locally.

``b0`` is the small default. ``b5`` is about 340 MB and labels better; both
use NVIDIA's noncommercial SegFormer license.
"""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


MODELS = {
    "b0": {
        "repository": "optimum/segformer-b0-finetuned-ade-512-512",
        "revision": "312a6466f90f7057b711cf550d5659ed3a18d508",
        "folder": "segformer-b0-ade20k",
        # local name: (remote path, SHA-256)
        "files": {
            "model.onnx": ("model.onnx", "3a89102115fe3c16230502437b894844ba50cde6f7c800f9884e87c360bcbfc9"),
            "config.json": ("config.json", "8faac3a12302746f63d8a3a8c0bbf7f7f49bf459e5a3533bf71c067cb824e799"),
            "preprocessor_config.json": ("preprocessor_config.json",
                                         "b5fa2a5da2f4483d6f2f8ac96c2a81ff91a91a0815cfbc77cc2ab30fd7c7b45b"),
        },
    },
    "b5": {
        "repository": "Xenova/segformer-b5-finetuned-ade-640-640",
        "revision": "02e76595880a97cf256ceecb93acbc070e21fe97",  # 2025-07-01
        "folder": "segformer-b5-ade20k",
        "files": {
            "model.onnx": ("onnx/model.onnx", "ea8305f7c32e7d5ccdaa24d97161e4246e027cfef5835c90d0f63454d7fbfb41"),
            "config.json": ("config.json", "7353d2b4b1cbb8eb239891d0292f50637759bbe9d0446c39c165c5186d47066c"),
            "preprocessor_config.json": ("preprocessor_config.json",
                                         "7a6d83d17d6ebc8ba352b63988dd22741a2a2717d8d015cde0c6a08e86c3d12b"),
        },
    },
}


def model_for_hash(model_sha256):
    """Return the pinned model entry whose ONNX file has this SHA-256."""
    for entry in MODELS.values():
        if entry["files"]["model.onnx"][1] == model_sha256:
            return entry
    raise ValueError("ONNX model does not match any pinned SegFormer checkpoint.")


def download_model(destination, name="b0"):
    entry = MODELS[name]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{entry['repository']}/resolve/{entry['revision']}/"
    for local, (remote, expected_hash) in entry["files"].items():
        target = destination / local
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_hash:
            continue
        with urlopen(base + remote, timeout=600) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"Pinned model checksum mismatch for {local}; file was not saved.")
        target.write_bytes(data)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default="b0")
    parser.add_argument("--output", type=Path, help="Default: models/<model folder>")
    args = parser.parse_args()
    output = args.output or Path(__file__).resolve().parent / "models" / MODELS[args.model]["folder"]
    print(f"Verified model in {download_model(output, args.model)}")


if __name__ == "__main__":
    main()
