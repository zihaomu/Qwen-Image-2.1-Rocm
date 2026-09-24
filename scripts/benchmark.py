#!/usr/bin/env python3
import argparse
import asyncio
import copy
import json
import os
import statistics
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiohttp


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, 0o644)
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def is_api_graph(value):
    return (isinstance(value, dict) and bool(value) and
            all(isinstance(node_id, str) and isinstance(node, dict) and
                isinstance(node.get("class_type"), str) and
                bool(node["class_type"]) and
                isinstance(node.get("inputs"), dict)
                for node_id, node in value.items()))


def graph_from_history(item):
    if not isinstance(item, dict):
        raise RuntimeError("history entry schema type is not dict")
    raw = item.get("prompt")

    graph = None
    if isinstance(raw, dict):
        if is_api_graph(raw):
            graph = raw
        elif "prompt" in raw and is_api_graph(raw["prompt"]):
            graph = raw["prompt"]
    elif isinstance(raw, (list, tuple)):
        if len(raw) >= 3 and is_api_graph(raw[2]):
            graph = raw[2]
    if graph is None:
        raw_len = len(raw) if isinstance(raw, (list, tuple)) else "n/a"
        raise RuntimeError("history prompt schema type=%s len=%s has no API graph" %
                           (type(raw).__name__, raw_len))
    status = item.get("status", {})
    if status.get("status_str") != "success" or status.get("completed") is not True:
        raise RuntimeError("history is not successful")
    return copy.deepcopy(graph)


def nodes_of(graph, class_type):
    return [(str(node_id), node) for node_id, node in graph.items()
            if isinstance(node, dict) and node.get("class_type") == class_type]


def apply_prompt_suffix(graph, suffix):
    if not suffix:
        return
    nodes = nodes_of(graph, "TextEncodeQwenImage21")
    if len(nodes) != 1:
        raise RuntimeError("expected exactly one TextEncodeQwenImage21 node")
    _, node = nodes[0]
    prompt = node["inputs"].get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError("TextEncodeQwenImage21 prompt must be a string")
    node["inputs"]["prompt"] = prompt + suffix


def queue_is_empty(value):
    if not isinstance(value, dict):
        return False
    return not (value.get("queue_running") or value.get("queue_pending") or
                value.get("running") or value.get("pending"))


def output_files(outputs):
    found = []
    if isinstance(outputs, dict):
        for value in outputs.values():
            if isinstance(value, dict):
                images = value.get("images", [])
                if isinstance(images, list):
                    for image in images:
                        if isinstance(image, dict) and isinstance(image.get("filename"), str):
                            found.append(image["filename"])
    return found


async def request_json(session, method, url, **kwargs):
    async with session.request(method, url, **kwargs) as response:
        body = await response.json(content_type=None)
        if response.status >= 400:
            raise RuntimeError("HTTP %s" % response.status)
        return body


async def run(args):
    base = args.url.rstrip("/")
    history_base = (args.history_url or base).rstrip("/")
    if base.startswith("http://"):
        ws_base = "ws://" + base[len("http://"):]
    elif base.startswith("https://"):
        ws_base = "wss://" + base[len("https://"):]
    else:
        raise RuntimeError("url must start with http:// or https://")
    started_utc = utc_now()
    started_ns = time.monotonic_ns()
    client_id = str(uuid.uuid4())
    result = {
        "schema_version": 1,
        "source_prompt_id": args.history_prompt_id,
        "source_graph_file": args.graph_file,
        "new_prompt_id": None,
        "seed": args.seed,
        "prefix": args.prefix,
        "start": started_utc,
        "end": None,
        "wall_ms": None,
        "events": [],
        "node_intervals": [],
        "cached_nodes": [],
        "progress_summary": {},
        "binary_frames": {"count": 0, "bytes": 0},
        "history_status": None,
        "outputs": [],
        "error": None,
        "metadata": {
            "warmup": bool(args.warmup),
            "prompt_suffix_length": len(args.prompt_suffix),
        },
    }
    node_map = {}
    sampler_id = None
    save_id = None
    active = None
    progress = []
    try:
        timeout = aiohttp.ClientTimeout(total=args.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if args.graph_file:
                graph = json.loads(Path(args.graph_file).read_text(encoding="utf-8"))
                if not is_api_graph(graph):
                    raise RuntimeError("graph file is not an API graph")
            else:
                history = await request_json(
                    session, "GET", f"{history_base}/history/{args.history_prompt_id}"
                )
                item = history.get(args.history_prompt_id) if isinstance(history, dict) else None
                if not isinstance(item, dict):
                    raise RuntimeError("history prompt id not found")
                graph = graph_from_history(item)
            apply_prompt_suffix(graph, args.prompt_suffix)
            samplers = nodes_of(graph, "KSampler")
            saves = nodes_of(graph, "SaveImageAdvanced")
            if len(samplers) != 1 or len(saves) != 1:
                raise RuntimeError("KSampler and SaveImageAdvanced must each be unique")
            sampler_id, sampler = samplers[0]
            save_id, save = saves[0]
            sampler.setdefault("inputs", {})["seed"] = args.seed
            save.setdefault("inputs", {})["filename_prefix"] = args.prefix
            node_map = {str(k): v.get("class_type") for k, v in graph.items() if isinstance(v, dict)}
            queue = await request_json(session, "GET", f"{base}/queue")
            if not queue_is_empty(queue):
                raise RuntimeError("queue is non-empty")
            ws_url = f"{ws_base}/ws?clientId={client_id}"
            async with session.ws_connect(ws_url, heartbeat=30) as ws:
                payload = {"prompt": graph, "client_id": client_id}
                posted = await request_json(session, "POST", f"{base}/prompt", json=payload)
                prompt_id = posted.get("prompt_id")
                errors = posted.get("node_errors")
                if not prompt_id or errors:
                    raise RuntimeError("prompt rejected")
                result["new_prompt_id"] = prompt_id
                success = False
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    try:
                        message = await asyncio.wait_for(ws.receive(), max(0.1, deadline - time.monotonic()))
                    except asyncio.TimeoutError:
                        break
                    now_ns = time.monotonic_ns()
                    if message.type == aiohttp.WSMsgType.BINARY:
                        result["binary_frames"]["count"] += 1
                        result["binary_frames"]["bytes"] += len(message.data)
                        continue
                    if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE,
                                        aiohttp.WSMsgType.ERROR):
                        break
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    try:
                        event = json.loads(message.data)
                    except json.JSONDecodeError:
                        continue
                    kind = event.get("type")
                    data = event.get("data") or {}
                    if kind not in {"status", "execution_start", "execution_cached", "executing",
                                    "progress", "executed", "execution_success", "execution_error"}:
                        continue
                    if data.get("prompt_id") != prompt_id:
                        continue
                    result["events"].append({"type": kind, "time_ns": now_ns - started_ns})
                    if kind == "execution_cached":
                        for node in data.get("nodes", []) or []:
                            node = str(node)
                            if node not in result["cached_nodes"]:
                                result["cached_nodes"].append(node)
                    elif kind == "executing":
                        node = data.get("node")
                        if active is not None:
                            active["end_ns"] = now_ns
                            active["duration_ms"] = (now_ns - active.pop("start_ns")) / 1e6
                            result["node_intervals"].append(active)
                            active = None
                        if node is not None:
                            node = str(node)
                            active = {"node": node, "class_type": node_map.get(node), "start_ns": now_ns}
                    elif kind == "progress":
                        value, maximum = data.get("value"), data.get("max")
                        if isinstance(value, (int, float)) and isinstance(maximum, (int, float)):
                            progress.append((now_ns, value, maximum))
                    elif kind == "execution_success":
                        success = True
                        break
                    elif kind == "execution_error":
                        raise RuntimeError("execution_error")
                if active is not None:
                    active["end_ns"] = time.monotonic_ns()
                    active["duration_ms"] = (active["end_ns"] - active.pop("start_ns")) / 1e6
                    result["node_intervals"].append(active)
                if not success:
                    raise RuntimeError("timeout or execution did not succeed")
            final = await request_json(session, "GET", f"{base}/history/{result['new_prompt_id']}")
            final_item = final.get(result["new_prompt_id"], {}) if isinstance(final, dict) else {}
            result["history_status"] = final_item.get("status")
            if not isinstance(result["history_status"], dict) or result["history_status"].get("status_str") != "success":
                raise RuntimeError("final history is not successful")
            result["outputs"] = output_files(final_item.get("outputs", {}))
            if not result["outputs"]:
                raise RuntimeError("successful history has no output filename")
            if progress:
                intervals = [(progress[i][0] - progress[i - 1][0]) / 1e6 for i in range(1, len(progress))]
                result["progress_summary"] = {"node": sampler_id, "first": progress[0][1:],
                    "last": progress[-1][1:], "samples": len(progress),
                    "per_step_intervals_ms_median": statistics.median(intervals) if intervals else None,
                    "per_step_intervals_ms_p95": (statistics.quantiles(intervals, n=100, method="inclusive")[94]
                                                   if len(intervals) > 1 else (intervals[0] if intervals else None))}
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        result["end"] = utc_now()
        result["wall_ms"] = (time.monotonic_ns() - started_ns) / 1e6
        atomic_json(args.output_json, result)
    for interval in result["node_intervals"]:
        print("NODE_TIMING %s %s %.3f" % (interval["node"], interval["class_type"], interval["duration_ms"]))
    for node in result["cached_nodes"]:
        print("CACHED %s" % node)
    if result["progress_summary"]:
        print("PROGRESS " + json.dumps(result["progress_summary"], sort_keys=True))
    print("PROMPT_ID %s" % (result["new_prompt_id"] or ""))
    print("WALL %.3f" % result["wall_ms"])
    print("OUTPUT " + json.dumps(result["outputs"]))
    print("PROFILE_VALIDATED %s" % ("PASS" if result["error"] is None else "FAIL"))
    return 0 if result["error"] is None else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--history-url", default=None)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--history-prompt-id")
    source.add_argument("--graph-file")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--timeout", default=900, type=float)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--prompt-suffix", default="", help="suffix for cache-busting")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
