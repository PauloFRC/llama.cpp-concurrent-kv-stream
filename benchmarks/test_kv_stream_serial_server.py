#!/usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = ROOT / "build/bin/llama-server"


def request(port: int, path: str, payload: dict | None, timeout: int = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


class Server:
    def __init__(self, server: Path, model: Path, port: int, cache_mib: int, log: Path,
                 n_parallel: int = 1, ctx_size: int = 8448, extra: list[str] = (), env: dict = None):
        command = [
            str(server),
            "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(ctx_size), "-fa", "on", "-ctk", "q8_0", "-ctv", "q4_0",
            "-ngl", "all", "-b", "256", "-ub", "256", "-np", str(n_parallel),
            "--no-mmproj", "--no-warmup", "--reasoning-format", "none",
            "--kv-stream-stage-mib", "128", "--cache-ram", str(cache_mib),
            *extra,
        ]
        self.log_path = log
        self.log_file = log.open("wb")
        self.process = subprocess.Popen(
            command, stdout=self.log_file, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log_file.close()
                raise RuntimeError(f"server exited with {self.process.returncode}")
            try:
                if request(port, "/health", None, 2).get("status") == "ok":
                    self.port = port
                    return
            except Exception:
                time.sleep(0.25)
        self.process.send_signal(signal.SIGINT)
        self.process.wait(timeout=15)
        self.log_file.close()
        raise RuntimeError("server did not become ready")

    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=15)
        self.log_file.close()


def completion(server: Server, prompt: list[int], cache_prompt: bool, n_predict: int = 16) -> dict:
    return request(server.port, "/completion", {
        "prompt": prompt,
        "n_predict": n_predict,
        "ignore_eos": True,
        "cache_prompt": cache_prompt,
        "temperature": 0,
        "seed": 1,
        "reasoning_format": "none",
    })


def patterned(size: int, tokens: tuple[int, ...]) -> list[int]:
    return [tokens[i % len(tokens)] for i in range(size)]


def run_serial(binary: Path, model: Path, port: int, output: Path):
    server = Server(binary, model, port, 0, output / "serial.log")
    try:
        short = patterned(1024, (23066, 1000, 2000))
        medium = patterned(4096, (23066, 3000, 4000, 5000))
        streamed = patterned(6144, (23066, 6000, 7000, 8000, 9000))
        short_first = completion(server, short, False)
        medium_first = completion(server, medium, False)
        streamed_result = completion(server, streamed, False)
        short_second = completion(server, short, False)
        medium_second = completion(server, medium, False)
        serial_changed = (
            short_second["content"] != short_first["content"] or
            medium_second["content"] != medium_first["content"])
        streamed_invalid = not streamed_result["content"].strip("/")
        if serial_changed or streamed_invalid:
            details = {
                "short_first": short_first["content"],
                "short_second": short_second["content"],
                "medium_first": medium_first["content"],
                "streamed": streamed_result["content"],
                "medium_second": medium_second["content"],
            }
            raise RuntimeError(f"serial unrelated-prefill output changed: {json.dumps(details)}")
        print("serial unrelated-prefill test: PASS", flush=True)
    finally:
        server.stop()


def run_prompt_cache(binary: Path, model: Path, port: int, output: Path):
    server = Server(binary, model, port, 1536, output / "prompt-cache.log")
    try:
        cached = patterned(4096, (23066, 1100, 2100, 3100))
        unrelated = patterned(2048, (23066, 4100, 5100, 6100))
        expected = completion(server, cached, True)["content"]
        completion(server, unrelated, True)
        restored = completion(server, cached, True)
        if restored["content"] != expected:
            raise RuntimeError("prompt-cache restore output changed")
        if restored["timings"].get("cache_n", 0) == 0:
            raise RuntimeError("prompt-cache restore did not reuse cached tokens")
        print(f"prompt-cache restore test: PASS (cache_n={restored['timings']['cache_n']})", flush=True)
    finally:
        server.stop()


RESTORE_RUNS = re.compile(rb"state_read_data: restoring (\d+) cells in (\d+) runs")
CELL_PAGE = re.compile(rb"^([.0-9M]{256}) \*$", re.MULTILINE)


def mixed_pages(log: bytes) -> list[bytes]:
    return [page for page in CELL_PAGE.findall(log) if len(set(page) - {ord(".")}) > 1]


def run_parked_slots(binary: Path, model: Path, port: int, output: Path):
    server = Server(binary, model, port, 2048, output / "parked-slots.log",
                    n_parallel=2, ctx_size=12288, extra=["--kv-unified", "--cache-idle-slots", "-lv", "5"],
                    env={"LLAMA_KV_CACHE_DEBUG": "3"})
    try:
        parent = patterned(4096, (23066, 1200, 2200, 3200))
        child_a = patterned(2048, (23066, 4200, 5200, 6200))
        child_b = patterned(2048, (23066, 7200, 8200, 9200))

        expected = completion(server, parent, True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(completion, server, child_a, True, 64)
            fut_b = pool.submit(completion, server, child_b, True, 256)
            child_a_result = fut_a.result()
            restored = completion(server, parent, True)
            child_b_result = fut_b.result()

        restored_a = completion(server, child_a, True, 64)

        details = {
            "expected": expected["content"],
            "restored": restored["content"],
            "expected_slot": expected["id_slot"],
            "restored_slot": restored["id_slot"],
            "child_slots": [child_a_result["id_slot"], child_b_result["id_slot"]],
            "restored_timings": restored["timings"],
        }
        if restored["content"] != expected["content"]:
            raise RuntimeError(f"parked-slot restore output changed: {json.dumps(details)}")
        cache_n = restored["timings"].get("cache_n", 0)
        if cache_n < len(parent) - 256:
            raise RuntimeError(f"parked-slot restore reused only {cache_n} tokens: {json.dumps(details)}")

        server.log_file.flush()
        log = server.log_path.read_bytes()
        restores = RESTORE_RUNS.findall(log)
        if not restores:
            raise RuntimeError("no state_read_data restore found in the server log")
        if len(restores) < 2:
            raise RuntimeError(f"expected two restores, found {len(restores)}")
        cells, runs = (int(x) for x in restores[-2])
        mixed = mixed_pages(log)
        if mixed:
            raise RuntimeError(
                f"{len(mixed)} page(s) held cells of more than one sequence, first: {mixed[0].decode()}")
        cells_a, runs_a = (int(x) for x in restores[-1])
        if restored_a["content"] != child_a_result["content"]:
            raise RuntimeError("child restore output changed: "
                               f"{json.dumps([child_a_result['content'], restored_a['content']])}")
        if runs_a != 1:
            raise RuntimeError(f"child restore landed in {runs_a} runs, the idle parent was not parked first")
        print(f"parked-slot restore test: PASS (cache_n={cache_n}, slot {expected['id_slot']} -> "
              f"{restored['id_slot']}, {cells} cells in {runs} runs, "
              f"prompt_ms={restored['timings']['prompt_ms']:.1f}; child back in {cells_a} cells, {runs_a} run)", flush=True)
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--port", type=int, default=12358)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/serial-server")
    parser.add_argument("--only", choices=["serial", "prompt-cache", "parked-slots"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "serial"):
        run_serial(args.server, args.model, args.port, args.output)
    if args.only in (None, "prompt-cache"):
        run_prompt_cache(args.server, args.model, args.port, args.output)
    if args.only in (None, "parked-slots"):
        run_parked_slots(args.server, args.model, args.port, args.output)


if __name__ == "__main__":
    main()
