#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {
    "doc/chrome_HLNmgdrCSb.jpg": "36448a3d37ceecdeaef0383c74eed19c6475d4bfe80680ef7cf759bfd5688299",
    "artifacts/comfy_kitchen_gfx1100.so": "0328285d26d04786c0e72a463db5952bf8c9f9aab4544d88d71ce595fc45560d",
    "kernels/int8_attn.hip": "48ec2e4202f26dfb2870c2e0a3686a72271fdc937098d59edf9d24df499a1da1",
    "kernels/rms_rope.hip": "bc803be1d6e14c949a8539b3081e362d8ef5281cc02ce13b25ad6d7c15e5d9a1",
    "vendor/ComfyUI-d158420.tar.gz": "3c740366b49679e6b902345cd5d7bf38745e23c2a7ed036aa09a69f315f0c3ac",
    "vendor/comfy-kitchen-v0.2.35-b2a2972.tar.gz": "871453a59c0faf883bf42fcbb7fb4264479ac4cda6d799aaa47136229e216c97",
    "vendor/nanobind-3.1.0-py3-none-any.whl": "96984dbe8154ffb86e653b1b60aaeed6985c710c5e75ad0279eb87943c66292d",
    "vendor/cmake-4.4.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": "27b024e903ef985b37183d754a5c61230b56b41fe0971cd44b71b80c787ec594",
    "workflows/qwen_image_2_1_t2i_api.json": "be84790e39037bd04ef8602ead139565c52bee23cfda4610fabaac197a623c7d",
    "benchmarks/raw/ab_summary.json": "9f2fce3bb5a33cd655f40e58ddbb3d3da71533651354c50ac2f676a815df691f",
    "benchmarks/raw/image_comparison.json": "364a3b2e0ac156245d8e4a325078cf035149a2bcdc9d702aa55d150c2e6d034c",
    "benchmarks/images/reference_seed44.png": "e370c74216e58aefc683bf295637c9bcaff59b1764b87679614d5519e41b036d",
    "benchmarks/images/candidate_seed44.png": "0b4ef0e9cc184f29e60fad6747c2aaf713e60dff34f1195c5683c4a89c31d902",
    "artifacts/image-manifest.json": "f51e9db0cdae0ab8ba6ba6fa70b856d9d256c64243d7c516784482efdfbd7ddd",
    "benchmarks/reproduced/warm-seed43.json": "9d99fa7043314ec4eab59b7f4684393e215541734038f2899777af5e553257b7",
    "benchmarks/reproduced/measure-seed44.json": "769f21baf9dd0cf398e56bccf7c11ef97a794aae8fde918926a907b325c0f8f0",
    "benchmarks/reproduced/latest-summary.json": "6130d44d61c552fb499438bdc10a50c7e27349f098e5b9e525512781192aaf55",
    "benchmarks/reproduced/measure-seed44.png": "8f682a38ab5e0aaf7251c4b25bc2e773b61fc89737e072272b759d20e07a9b70",
    "benchmarks/reproduced/receipt.json": "34de55b347a1b6672f8ee0adf77631ffde881ce8d538513ebda1790bd0f3ff78",
    "benchmarks/reproduced/image-comparison.json": "dc8e710fc21955a305f6f309ff36698aba31c31d6da9c3319e4b110f81e35817",
    "benchmarks/reproduced/repeat-warm-seed43.json": "f2fd7242797b124ed2907c7001250376ed75a9bd15c885ec1e4ee229d38db5eb",
    "benchmarks/reproduced/repeat-measure-seed44.json": "7c7480da062a5c23fe982fded902adc8c32967ce9bd49794f2e74611fe748743",
    "benchmarks/reproduced/repeat-summary.json": "1ad8e8dd78b0bc313eb4059cfabf6d105a970c204b211ea3cd89bade813ba755",
    "benchmarks/reproduced/repeat-measure-seed44.png": "8f682a38ab5e0aaf7251c4b25bc2e773b61fc89737e072272b759d20e07a9b70",
    "benchmarks/reproduced/repeat-receipt.json": "796294a995eeab3a49d7e2813a90b45706cd0b308df2c6943a173e4523a4e31c",
    "benchmarks/reproduced/repeat-image-comparison.json": "4d45390f854a63c850a550bab2c5d0711ed41b81f3e158cf18ba091232bd0aab",
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


for relative, expected in EXPECTED.items():
    path = ROOT / relative
    assert path.is_file(), relative
    assert sha256(path) == expected, relative

for path in ROOT.rglob("*.json"):
    if any(part in {".git", "runtime", "output", "dist"} for part in path.parts):
        continue
    json.loads(path.read_text(encoding="utf-8"))

manifest = json.loads((ROOT / "models/manifest.json").read_text())
assert manifest["revision"] == "5dc8a8352312860d67e07cc9bee310dc7ed3e55c"
assert sum(item["size_bytes"] for item in manifest["files"]) == 17_283_091_112

api_graph = json.loads((ROOT / "workflows/qwen_image_2_1_t2i_api.json").read_text())
assert api_graph["459:458"]["inputs"] == {
    **api_graph["459:458"]["inputs"],
    "seed": 44,
    "steps": 25,
    "cfg": 1.0,
    "sampler_name": "euler",
    "scheduler": "simple",
}
assert api_graph["13"]["inputs"]["megapixels"] == 1.0

ab = json.loads((ROOT / "benchmarks/raw/ab_summary.json").read_text())
assert ab["overall"] == "PASS"
assert ab["wall_ms"]["candidate"] == 24981.79008
assert ab["wall_ms"]["reference"] == 32718.921377
ksampler = next(row for row in ab["node_table"] if row["class"] == "KSampler")
assert ksampler["candidate_ms"] == 23600.188111
assert ksampler["reference_ms"] == 31302.259783
assert ab["backend_warm_current_delta"]["candidate"]["fallback_delta"] == 0

composition = json.loads((ROOT / "benchmarks/raw/composition_summary.json").read_text())
assert composition["status"] == "verified_no_regression"
assert composition["combined_artifact"]["sha256"] == EXPECTED["artifacts/comfy_kitchen_gfx1100.so"]

reproduced = json.loads((ROOT / "benchmarks/reproduced/measure-seed44.json").read_text())
assert reproduced["wall_ms"] == 25064.687212
assert reproduced["cached_nodes"] == ["13", "459:451", "459:452", "459:453", "459:454", "459:456"]
assert reproduced["metadata"] == {"prompt_suffix_length": 0, "warmup": False}
reproduced_ksampler = next(row for row in reproduced["node_intervals"] if row["class_type"] == "KSampler")
assert reproduced_ksampler["duration_ms"] == 23680.349649
summary = json.loads((ROOT / "benchmarks/reproduced/latest-summary.json").read_text())
assert summary["candidate_calls_total"] == 1664
assert summary["fallback_calls"] == 0
comparison = json.loads((ROOT / "benchmarks/reproduced/image-comparison.json").read_text())
assert comparison["exact_rgba_values"] is True
assert comparison["mae"] == comparison["rmse"] == 0.0
assert comparison["identical_values_fraction"] == 1.0
image_manifest = json.loads((ROOT / "artifacts/image-manifest.json").read_text())
assert image_manifest["image_id"] == "sha256:3698286daa306924b4fbbc2e0e896355841cadf8e8fe17fab835f99663ab1018"
assert image_manifest["archive_sha256"] == "243d60e64108ccaec239f1a2ab4fde9adde76c48b658c6068abe830ba482e3a1"
repeat = json.loads((ROOT / "benchmarks/reproduced/repeat-measure-seed44.json").read_text())
assert repeat["wall_ms"] == 25196.804631
assert repeat["cached_nodes"] == reproduced["cached_nodes"]
repeat_ksampler = next(row for row in repeat["node_intervals"] if row["class_type"] == "KSampler")
assert repeat_ksampler["duration_ms"] == 23742.350545
repeat_summary = json.loads((ROOT / "benchmarks/reproduced/repeat-summary.json").read_text())
assert repeat_summary["candidate_calls"] == 1664
assert repeat_summary["fallback_calls"] == 0
repeat_comparison = json.loads((ROOT / "benchmarks/reproduced/repeat-image-comparison.json").read_text())
assert repeat_comparison["exact_rgba_values"] is True
assert repeat_comparison["identical_values_fraction"] == 1.0

for script in sorted((ROOT / "scripts").glob("*.sh")):
    subprocess.run(["bash", "-n", str(script)], check=True)

for markdown in (ROOT / "README.md", ROOT / "THIRD_PARTY.md", ROOT / "benchmarks/W7900D.md"):
    text = markdown.read_text(encoding="utf-8")
    assert text.count("```") % 2 == 0, markdown
    for target in re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", text):
        local = target.split("#", 1)[0]
        if not local or re.match(r"^[a-z]+://", local):
            continue
        assert (markdown.parent / local).resolve().exists(), (markdown, target)

credential = re.compile(r"(?:token|api[_-]?key|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{16,}", re.I)
for path in ROOT.rglob("*"):
    if not path.is_file() or any(part in {".git", "runtime", "output", "dist"} for part in path.parts):
        continue
    if path.suffix.lower() in {".md", ".json", ".py", ".sh", ".txt"} or path.name in {"Dockerfile", ".gitignore", ".dockerignore"}:
        assert not credential.search(path.read_text(errors="ignore")), path

unexpected = [path for path in ROOT.rglob("*.tmp*") if ".git" not in path.parts]
assert not unexpected, unexpected
print(f"PASS: {len(EXPECTED)} locked assets, JSON, workflow, W7900D evidence, shell, links, credentials")
