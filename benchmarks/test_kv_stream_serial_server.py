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
UBATCH = 256


def request(port: int, path: str, payload: dict | None, timeout: int = 1800) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


class Server:
    def __init__(self, server: Path, model: Path, port: int, cache_mib: int, log: Path,
                 n_parallel: int = 1, ctx_size: int = 8448, extra: list[str] = (), env: dict = None,
                 stage_mib: int = 128):
        command = [
            str(server),
            "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(ctx_size), "-fa", "on", "-ctk", "q8_0", "-ctv", "q4_0",
            "-ngl", "all", "-b", "256", "-ub", str(UBATCH), "-np", str(n_parallel),
            "--no-mmproj", "--no-warmup", "--reasoning-format", "none",
            "--kv-stream-stage-mib", str(stage_mib), "--cache-ram", str(cache_mib),
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
TRACE_LINE = re.compile(rb"kv_stream_adapt: active (\d+), resident \d+, ring \d+, (?:layout \d+, )?samples \d+, misses \d+, "
                        rb"copy busy ([0-9.]+)%, peak \d+, skipped (\d+), resident attended (\d+)")
CACHE_EVICT = re.compile(rb"removing oldest entry|exceeds cache size limit")
DISK_SPILL = re.compile(rb"spilled entry with (\d+) tokens to disk \(([0-9.]+) MiB in ([0-9.]+) ms\)")
DISK_RESTORE = re.compile(rb"restored entry with (\d+) tokens from disk \(([0-9.]+) MiB in ([0-9.]+) ms\)")
DISK_DIRECT = re.compile(rb"saved entry with (\d+) tokens to disk \(([0-9.]+) MiB in ([0-9.]+) ms\)")
DISK_FILES = re.compile(rb"disk files: (\d+)")
STATE_SIZE = re.compile(rb"saving prompt with length (\d+), total state size = ([0-9.]+) MiB")


def rss_mib(pid: int) -> float:
    with open(f"/proc/{pid}/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def spill_files(pid: int, directory: Path) -> int:
    n = 0
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            if os.readlink(fd).startswith(f"{directory.resolve()}/"):
                n += 1
        except OSError:
            pass
    return n


def parse_fill(value: str) -> tuple[int, int]:
    scale = {"k": 1024, "m": 1024 * 1024}
    parked, active = value.lower().split("+")
    return tuple(int(part.rstrip("km")) * scale.get(part[-1], 1) for part in (parked, active))


def summarize_trace(span: bytes) -> dict:
    rows = TRACE_LINE.findall(span)
    return {
        "active_max": max((int(row[0]) for row in rows), default=0),
        "copy_busy_max": max((float(row[1]) for row in rows), default=0.0),
        "skipped_pages": sum(int(row[2]) for row in rows),
        "resident_pages_attended": sum(int(row[3]) for row in rows),
    }


def mixed_pages(log: bytes) -> list[bytes]:
    return [page for page in CELL_PAGE.findall(log) if len(set(page) - {ord(".")}) > 1]


def run_parked_slots(binary: Path, model: Path, port: int, output: Path, args):
    parked_n, active_n = args.fill
    # unified cache holds the parked sequence, both active ones and their decode tokens
    ctx_size = max(12288, (parked_n + 2 * active_n + 1024 + 255) // 256 * 256)
    cache_ram = args.cache_ram if args.cache_ram is not None else max(2048, (parked_n + 2 * active_n) * 64 // 1024 + 2048)
    env = {"LLAMA_KV_STREAM_TRACE": "1"}
    if args.debug_cells:
        env["LLAMA_KV_CACHE_DEBUG"] = "3"
    extra = ["--kv-unified", "--cache-idle-slots", "-lv", "5"]
    if args.cache_disk_dir:
        extra.extend(["--cache-disk-dir", str(args.cache_disk_dir)])
    server = Server(binary, model, port, cache_ram, output / "parked-slots.log",
                    n_parallel=2, ctx_size=ctx_size, extra=extra,
                    env=env, stage_mib=args.stage_mib)
    t0 = time.monotonic()
    marks = []

    def mark(name: str):
        marks.append((name, server.log_path.stat().st_size, time.monotonic() - t0, rss_mib(server.process.pid)))

    try:
        parked = patterned(parked_n, (23066, 1200, 2200, 3200))
        active_a = patterned(active_n, (23066, 4200, 5200, 6200))
        active_b = patterned(active_n, (23066, 7200, 8200, 9200))
        diverged = parked[:-UBATCH // 2] + patterned(UBATCH // 2, (23066, 1300, 2300, 3300))

        mark("prefill-parked")
        expected = completion(server, parked, True)
        if args.checkpoint_hit:
            mark("hit-baseline")
            expected_hit = completion(server, diverged, True)
            mark("prefill-parked-again")
            again = completion(server, parked, True)
            if again["content"] != expected["content"]:
                raise RuntimeError("re-sending the parent after a checkpoint hit changed its output")
        mark("active-concurrent")
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(completion, server, active_a, True, 64)
            fut_b = pool.submit(completion, server, active_b, True, 256)
            result_a = fut_a.result()
            if args.cache_disk_dir and cache_ram > 0:
                # parking A spills the parent here
                mark("spill-parked")
                completion(server, patterned(512, (23066, 1400, 2400, 3400)), True, 4)
            mark("restore-parked")
            restored = completion(server, parked, True)
            mark("active-tail")
            result_b = fut_b.result()
        mark("restore-active")
        restored_a = completion(server, active_a, True, 64)
        if args.checkpoint_hit:
            # the parent came back from disk
            mark("hit-after-disk")
            hit_after = completion(server, diverged, True)
        mark("end")
        open_spills = spill_files(server.process.pid, args.cache_disk_dir) if args.cache_disk_dir else 0

        details = {
            "expected": expected["content"],
            "restored": restored["content"],
            "expected_slot": expected["id_slot"],
            "restored_slot": restored["id_slot"],
            "active_slots": [result_a["id_slot"], result_b["id_slot"]],
            "restored_timings": restored["timings"],
        }
        if restored["content"] != expected["content"]:
            raise RuntimeError(f"parked-slot restore output changed: {json.dumps(details)}")
        cache_n = restored["timings"].get("cache_n", 0)
        if cache_n < parked_n - 256:
            raise RuntimeError(f"parked-slot restore reused only {cache_n} tokens: {json.dumps(details)}")

        log = server.log_path.read_bytes()
        if CACHE_EVICT.search(log):
            raise RuntimeError(f"prompt cache evicted a state, pass --cache-ram above {cache_ram}")
        windows = {}
        for (name, start, t_start, rss_start), (_, end, t_end, rss_end) in zip(marks, marks[1:]):
            span = log[start:end]
            windows[name] = {
                "span": [start, end],
                "seconds": t_end - t_start,
                "rss_mib": [rss_start, rss_end],
                "spills": [[int(n), float(mib), float(ms)] for n, mib, ms in DISK_SPILL.findall(span)],
                "disk_restores": [[int(n), float(mib), float(ms)] for n, mib, ms in DISK_RESTORE.findall(span)],
                "direct_saves": [[int(n), float(mib), float(ms)] for n, mib, ms in DISK_DIRECT.findall(span)],
                "restores": [[int(cells), int(runs)] for cells, runs in RESTORE_RUNS.findall(span)],
                **summarize_trace(span),
            }
        if args.cache_disk_dir:
            if not any(n >= parked_n for n, _, _ in windows["restore-parked"]["disk_restores"]):
                raise RuntimeError("parent was not restored from disk")
            if any(n >= parked_n for n, _, _ in windows["restore-parked"]["spills"]):
                raise RuntimeError("parent was spilled by its own restore")
            if cache_ram > 0 and not any(n >= parked_n for n, _, _ in windows["spill-parked"]["spills"]):
                raise RuntimeError("parent did not spill in spill-parked, lower --cache-ram")
            if cache_ram == 0 and not any(n >= parked_n for n, _, _ in windows["active-concurrent"]["direct_saves"]):
                raise RuntimeError("parent was not saved direct to disk in active-concurrent")
            # the last cache-state trace lists every blob still open
            on_disk = sum(int(n) for n in DISK_FILES.findall(log[log.rfind(b"cache state:"):]))
            if open_spills != on_disk:
                raise RuntimeError(f"{open_spills} spill files open for {on_disk} blobs on disk")
            if any(Path(args.cache_disk_dir).iterdir()):
                raise RuntimeError("named files left in --cache-disk-dir")
        if args.checkpoint_hit:
            if hit_after["content"] != expected_hit["content"]:
                raise RuntimeError("checkpoint hit after the disk round trip changed output: "
                                   f"{json.dumps([expected_hit['content'], hit_after['content']])}")
            if args.cache_disk_dir and not windows["hit-after-disk"]["disk_restores"]:
                raise RuntimeError("the checkpoint hit did not restore the parent from disk")
            for name in ("hit-baseline", "hit-after-disk"):
                start, end = windows[name]["span"]
                if b"restored context checkpoint" not in log[start:end]:
                    raise RuntimeError(f"{name} did not restore a checkpoint")
        if not windows["restore-parked"]["restores"] or not windows["restore-active"]["restores"]:
            raise RuntimeError("no state_read_data restore found in a restore window")
        cells, runs = windows["restore-parked"]["restores"][-1]
        cells_a, runs_a = windows["restore-active"]["restores"][-1]
        if args.debug_cells:
            mixed = mixed_pages(log)
            if mixed:
                raise RuntimeError(
                    f"{len(mixed)} page(s) held cells of more than one sequence, first: {mixed[0].decode()}")
        if restored_a["content"] != result_a["content"]:
            raise RuntimeError("active restore output changed: "
                               f"{json.dumps([result_a['content'], restored_a['content']])}")
        if runs_a != 1:
            raise RuntimeError(f"active restore landed in {runs_a} runs, the idle sequence was not parked first")

        summary = {
            "shape": {"parked_tokens": parked_n, "active_tokens": active_n, "ctx_size": ctx_size,
                      "cache_ram_mib": cache_ram, "stage_mib": args.stage_mib,
                      "state_sizes": [[int(n), float(mib)] for n, mib in STATE_SIZE.findall(log)]},
            "requests": {name: {"id_slot": r["id_slot"], "timings": r["timings"]} for name, r in (
                ("parked", expected), ("active_a", result_a), ("active_b", result_b),
                ("parked_resumed", restored), ("active_a_resumed", restored_a))},
            "windows": windows,
        }
        (output / "parked-slots-summary.json").write_text(json.dumps(summary, indent=2))
        disk_info = ""
        if "spill-parked" in windows and windows["spill-parked"]["spills"]:
            disk_info += f", spill {windows['spill-parked']['spills'][-1][2]:.1f} ms"
        if windows["restore-parked"]["disk_restores"]:
            disk_info += f", disk restore {windows['restore-parked']['disk_restores'][-1][2]:.1f} ms"
        if windows["active-concurrent"]["direct_saves"]:
            disk_info += f", direct save {windows['active-concurrent']['direct_saves'][-1][2]:.1f} ms"

        print(f"parked-slot restore test: PASS (cache_n={cache_n}, slot {expected['id_slot']} -> "
              f"{restored['id_slot']}, {cells} cells in {runs} runs, "
              f"prompt_ms={restored['timings']['prompt_ms']:.1f}{disk_info}, "
              f"skipped {windows['restore-parked']['skipped_pages']} pages; "
              f"active back in {cells_a} cells, {runs_a} run)", flush=True)
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--port", type=int, default=12358)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/serial-server")
    parser.add_argument("--only", choices=["serial", "prompt-cache", "parked-slots"])
    shape = parser.add_argument_group("parked-slots shape")
    shape.add_argument("--fill", type=parse_fill, default=(4096, 2048),
                       help="parked and active sequence depths, e.g. 100K+10K (default: 4096+2048)")
    shape.add_argument("--stage-mib", type=int, default=128, help="resident + staging pool in MiB")
    shape.add_argument("--cache-ram", type=int, help="host prompt cache cap in MiB (default: sized from --fill)")
    shape.add_argument("--debug-cells", action="store_true",
                       help="LLAMA_KV_CACHE_DEBUG=3 and the mixed-page check; slow, off for timing runs")
    shape.add_argument("--cache-disk-dir", type=Path,
                       help="spill parked states here; set --cache-ram between one and two parked states so the parent spills, or 0 for direct-to-disk only")
    shape.add_argument("--checkpoint-hit", action="store_true",
                       help="also resume the parent with a prompt that diverges ub/2 tokens before its end (hits a context checkpoint)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "serial"):
        run_serial(args.server, args.model, args.port, args.output)
    if args.only in (None, "prompt-cache"):
        run_prompt_cache(args.server, args.model, args.port, args.output)
    if args.only in (None, "parked-slots"):
        run_parked_slots(args.server, args.model, args.port, args.output, args)


if __name__ == "__main__":
    main()
