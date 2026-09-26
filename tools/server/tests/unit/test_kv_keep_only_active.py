import os
import sys
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()

class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


# idle slot cleared on launch should restore from cache-ram
def test_clear_and_restore():
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # Slot 0 is the only slot with KV — should NOT be cleared
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # Launching slot 1 clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.drain()

    # Re-send same prompt — should restore from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "updating prompt cache" in log.drain()
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n

    # Follow-up — slot 0 kept its KV, no clearing needed
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT + " The knight finally reached the castle gates.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def test_disabled_with_flag():
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


linux_only = pytest.mark.skipif(sys.platform != "linux", reason="--cache-disk-dir is Linux only")


def token_prompt(seed: int, n: int, base: int = 20) -> list[int]:
    return [(seed * 7 + i) % 400 + base for i in range(n)]


def complete(prompt: list[int], id_slot: int = -1) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": id_slot,
        "cache_prompt": True,
    })
    assert res.status_code == 200, res.body
    return res.body


def spill_files(directory) -> int:
    n = 0
    for fd in os.listdir(f"/proc/{server.process.pid}/fd"):
        try:
            if os.readlink(f"/proc/{server.process.pid}/fd/{fd}").startswith(f"{directory}/"):
                n += 1
        except OSError:
            pass
    return n


@linux_only
def test_disk_spill_restore(tmp_path):
    global server
    server.n_ctx = 2048
    server.cache_ram = 1
    server.cache_disk_dir = str(tmp_path)
    server.start()
    log = LogReader(server.log_path)

    prompts = [token_prompt(k, 700) for k in range(4)]

    # each launch on the other slot parks the idle one
    first = [complete(p, id_slot) for p, id_slot in zip(prompts, (0, 1, 0, 1))]
    content = log.drain()
    assert "__TEST_TAG_CACHE_DISK_SPILL__" in content
    assert "removing oldest entry" not in content
    assert spill_files(tmp_path) == 1
    assert not os.listdir(tmp_path)

    res = complete(prompts[0])
    assert "__TEST_TAG_CACHE_DISK_RESTORE__" in log.drain()
    assert res["timings"]["cache_n"] > 0
    assert res["timings"]["prompt_n"] < first[0]["timings"]["prompt_n"]
    assert res["content"] == first[0]["content"]
    assert spill_files(tmp_path) == 1
    assert not os.listdir(tmp_path)


@linux_only
def test_disk_limit(tmp_path):
    global server
    server.n_ctx = 4096
    server.cache_ram = 2
    server.cache_disk_dir = str(tmp_path)
    server.cache_disk = 1
    server.start()
    log = LogReader(server.log_path)

    # cache: A (1.25 MiB) and B (0.19 MiB), C (0.69 MiB) idle in slot 0
    complete(token_prompt(0, 2000), 0)
    complete(token_prompt(1, 300), 1)
    complete(token_prompt(2, 1100), 0)
    log.drain()

    # parking C needs 0.12 MiB: A is too big for the disk tier, B is not
    complete(token_prompt(3, 1100))
    content = log.drain()
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in content
    assert "__TEST_TAG_CACHE_DISK_SPILL__" in content
    assert "removing oldest entry" not in content

    # parking P3 spills C, then parking P4 finds no room on disk and drops A
    complete(token_prompt(4, 1100), 0)
    complete(token_prompt(5, 1100), 1)
    content = log.drain()
    assert "disk tier is full" in content
    assert "removing oldest entry" in content
    assert spill_files(tmp_path) == 2


@linux_only
def test_disk_needs_ram_limit(tmp_path):
    global server
    server.n_ctx = 1024
    server.cache_ram = -1
    server.cache_disk_dir = str(tmp_path)
    server.start()
    log = LogReader(server.log_path)

    for k, id_slot in ((0, 0), (1, 1), (2, 0), (3, 1)):
        complete(token_prompt(k, 400), id_slot)
    assert "cache token limit" in log.drain()


@linux_only
def test_disk_direct_checkpoints(tmp_path):
    global server
    server = ServerPreset.tinygemma3()
    server.no_mmproj = True
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.kv_unified = True
    server.cache_ram = 0
    server.cache_disk_dir = str(tmp_path)
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    server.start()
    log = LogReader(server.log_path)

    # gemma ids below 1000 are special tokens
    prompt = token_prompt(0, 400, base=1000)
    other = token_prompt(1, 400, base=1000)

    complete(prompt, 0)
    assert "created context checkpoint" in log.drain()

    # launching slot 1 parks slot 0 with no RAM copy: main and ckpt blobs
    complete(other, 1)
    assert "__TEST_TAG_CACHE_DISK_DIRECT__" in log.drain()
    assert spill_files(tmp_path) == 2

    # diverge two tokens before the end: restore from disk, then the checkpoint at n-4; slot 1 parks direct
    body = complete(prompt[:-2] + token_prompt(9, 30, base=1000))
    content = log.drain()
    assert "__TEST_TAG_CACHE_DISK_RESTORE__" in content
    assert "restored context checkpoint" in content
    assert body["timings"]["cache_n"] >= 390
    assert spill_files(tmp_path) == 2
    assert not os.listdir(tmp_path)


@linux_only
def test_disk_direct_full(tmp_path):
    global server
    server.n_ctx = 2048
    server.cache_ram = 0
    server.cache_disk_dir = str(tmp_path)
    server.cache_disk = 1
    server.start()
    log = LogReader(server.log_path)

    for k, id_slot in ((0, 0), (1, 1), (2, 0), (3, 1)):
        complete(token_prompt(k, 700), id_slot)
    content = log.drain()
    assert "__TEST_TAG_CACHE_DISK_DIRECT__" in content
    assert "disk tier is full" in content
    assert "removing oldest entry" in content
    assert "exceeds cache size limit" not in content
    assert spill_files(tmp_path) == 2


@linux_only
def test_disk_pin_pending_load(tmp_path):
    global server
    server.n_ctx = 2048
    server.cache_ram = 1
    server.cache_disk_dir = str(tmp_path)
    server.start()
    log = LogReader(server.log_path)

    prompts = [token_prompt(k, 700) for k in range(4)]
    for p, id_slot in zip(prompts, (0, 1, 0, 1)):
        complete(p, id_slot)
    log.drain()

    body = complete(prompts[1])
    content = log.drain()
    assert "__TEST_TAG_CACHE_DISK_SPILL__" in content
    assert "__TEST_TAG_CACHE_DISK_RESTORE__" not in content
    # only the last prompt token is evaluated again
    assert body["timings"]["prompt_n"] == 1

    complete(token_prompt(4, 1100), 1)
    log.drain()
    body = complete(prompts[3])
    content = log.drain()
    assert "__TEST_TAG_CACHE_DISK_DIRECT__" in content
    assert "__TEST_TAG_CACHE_DISK_SPILL__" not in content
    assert "__TEST_TAG_CACHE_DISK_RESTORE__" not in content
    assert "removing oldest entry" not in content
    assert body["timings"]["prompt_n"] == 1
