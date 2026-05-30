"""
SHM Reopen Visibility Probe — tests H1 for Issue 1.

QUESTION: On Windows, after a sender closes+unlinks a SharedMemory segment and
creates a new one at the SAME name but DIFFERENT size, does a receiver that is
still holding the OLD handle:
  (a) read the old version from its stale mapping (H1 CONFIRMED), or
  (b) somehow see the new version (H1 refuted)?

This is a pure-Python, no-GPU, deterministic probe.  Run it before touching any
CUDA code.  The result tells us whether the Issue-1 root cause lives in the
SHM lifecycle layer or higher.

Usage:
    python scripts/probes/shm_reopen_probe.py

Expected output on Windows (H1 confirmed — bug confirmed):
    [1] Sender created SHM 'cudalink_probe_XXX' (size=33177600), wrote version 1
    [2] Receiver opened SHM 'cudalink_probe_XXX', reads version 1 OK
    [3] Sender close+unlink old SHM
    [4] Sender created NEW SHM 'cudalink_probe_XXX' (size=2073600), wrote version 2
    [5] Receiver reads version via OLD handle: 1  <-- stale mapping, H1 CONFIRMED
    [6] Fresh reader opens SHM 'cudalink_probe_XXX', reads version 2 OK
    RESULT: H1 CONFIRMED — old handle reads stale version 1, not new version 2
    Root cause: sender's set_version writes to a NEW segment the receiver cannot observe.

Expected output if H1 is refuted:
    [5] Receiver reads version via OLD handle: 2
    RESULT: H1 REFUTED — old handle sees new version (unexpected on Windows)
"""

from __future__ import annotations

import contextlib
import sys
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

# Allow running from the project root or from scripts/probes/
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))

from cuda_link.shm_protocol import read_version, set_version  # noqa: E402

# Realistic segment sizes: float32 RGBA 1080p vs uint8 mono 1080p
_FLOAT32_SIZE = 1920 * 1080 * 4 * 4  # 33,177,600
_UINT8_SIZE = 1920 * 1080 * 1 * 1  # 2,073,600

SHM_NAME = f"cudalink_probe_{uuid.uuid4().hex[:8]}"


def main() -> None:
    print(f"Probe: SHM reopen visibility on {sys.platform!r}")
    print(f"       Name={SHM_NAME!r}  float32_size={_FLOAT32_SIZE}  uint8_size={_UINT8_SIZE}")
    print()

    sender_shm = None
    receiver_shm = None
    sender_shm2 = None

    try:
        # [1] Sender creates the initial large segment and writes version=1
        sender_shm = SharedMemory(create=True, name=SHM_NAME, size=_FLOAT32_SIZE)
        set_version(sender_shm.buf, 1)
        print(f"[1] Sender created SHM {SHM_NAME!r} (size={_FLOAT32_SIZE}), wrote version 1")

        # [2] Receiver opens the segment (simulating receiver connect)
        receiver_shm = SharedMemory(name=SHM_NAME)
        v = read_version(receiver_shm.buf)
        ok = "OK" if v == 1 else f"UNEXPECTED (got {v})"
        print(f"[2] Receiver opened SHM {SHM_NAME!r}, reads version {v} {ok}")

        # [3] Sender closes + unlinks (simulates Exporter.close() in export_frame reopen)
        sender_shm.close()
        sender_shm = None
        # Re-open same name just to unlink (mirrors Exporter._do_cleanup step 7)
        try:
            _tmp = SharedMemory(name=SHM_NAME)
            _tmp.close()
            _tmp.unlink()
        except FileNotFoundError:
            pass
        print("[3] Sender close+unlink old SHM")

        # [4] Sender creates NEW segment at the SAME name but DIFFERENT (smaller) size
        try:
            sender_shm2 = SharedMemory(create=True, name=SHM_NAME, size=_UINT8_SIZE)
            set_version(sender_shm2.buf, 2)
            print(f"[4] Sender created NEW SHM {SHM_NAME!r} (size={_UINT8_SIZE}), wrote version 2")
        except FileExistsError as e:
            # On some OS/Windows configs the old segment (held by receiver_shm) is still alive
            print(f"[4] Sender FAILED to create new SHM — FileExistsError: {e}")
            print("    H1 mechanism variant: old segment blocked new creation (receiver pin)")
            _print_result_h1_confirmed()
            return

        # [5] KEY QUESTION: what does the receiver's OLD handle read?
        v_old = read_version(receiver_shm.buf)
        print(f"[5] Receiver reads version via OLD handle: {v_old}", end="  ")
        if v_old == 1:
            print("<-- stale mapping, H1 CONFIRMED")
        elif v_old == 2:
            print("<-- saw new version (H1 REFUTED)")
        else:
            print(f"<-- UNEXPECTED value {v_old!r}")

        # [6] Fresh reader — should see version 2
        try:
            fresh = SharedMemory(name=SHM_NAME)
            v_fresh = read_version(fresh.buf)
            fresh.close()
            ok2 = "OK" if v_fresh == 2 else f"UNEXPECTED (got {v_fresh})"
            print(f"[6] Fresh reader opens SHM {SHM_NAME!r}, reads version {v_fresh} {ok2}")
        except FileNotFoundError:
            print("[6] Fresh reader: SHM not found (receiver holds the only surviving reference)")

        print()
        if v_old == 1:
            _print_result_h1_confirmed()
        elif v_old == 2:
            print("RESULT: H1 REFUTED — old handle sees new version (unexpected on Windows)")
            print("        Bug must be elsewhere (par.format application, TD runtime, etc.)")
        else:
            print(f"RESULT: INCONCLUSIVE — unexpected version {v_old!r}")

    finally:
        for shm in (receiver_shm, sender_shm, sender_shm2):
            if shm is not None:
                with contextlib.suppress(OSError, BufferError):
                    shm.close()
        # Best-effort cleanup
        for name in (SHM_NAME,):
            with contextlib.suppress(FileNotFoundError, OSError):
                _tmp = SharedMemory(name=name)
                _tmp.close()
                _tmp.unlink()


def _print_result_h1_confirmed() -> None:
    print("RESULT: H1 CONFIRMED — old handle reads stale version 1, not new version 2")
    print("        Root cause: sender's set_version writes to a NEW segment the receiver")
    print("        cannot observe. The in-place refresh path (_refresh_on_version_change)")
    print("        can never fire because acquire_slot never sees VERSION_CHANGED.")
    print("        Fix: keep the SHM mapping open across dtype changes (in-place reopen).")


if __name__ == "__main__":
    main()
