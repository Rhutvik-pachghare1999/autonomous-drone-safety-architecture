import mmap
import os
import struct
import time

class VLASharedMemoryPublisher:
    """
    Zero-copy IPC bridge for the Vision-Language-Action (VLA) model.
    Writes nominal velocity vectors to a memory-mapped file that the C++
    safety kernel reads instantly.

    Struct layout (C-compatible, 64-byte aligned — must match
    src/rt/watchdog.h VLACommand):
      offset  0: uint64 seq_head   (seqlock: odd = write in progress)
      offset  8: double vx_nom
      offset 16: double vy_nom
      offset 24: double vz_nom
      offset 32: bool is_new_data (+7 pad)
      offset 40: uint64 seq_tail   (torn-read check: must equal seq_head)
      offset 48: 16 pad
    Total: 64 bytes

    Seqlock protocol — mmap writes are plain byte copies with NO atomicity
    against concurrent mmap readers: a single 33-byte write could be
    observed mid-copy as new seq + stale velocities (torn frame). The
    reader (vla_shm_snapshot in src/rt/watchdog.c) rejects torn frames;
    the safety filter then treats the cycle as no-new-data and hovers.
    """
    def __init__(self, shm_name="/dev/shm/aisp_vla_cmd"):
        self.shm_name = shm_name
        self.size = 64
        self.seq = 0

        # Create/open the file in /dev/shm (tmpfs, strictly RAM)
        fd = os.open(self.shm_name, os.O_CREAT | os.O_RDWR)
        os.ftruncate(fd, self.size)

        # Memory map
        self.shm = mmap.mmap(fd, self.size, mmap.MAP_SHARED, mmap.PROT_WRITE)
        os.close(fd)

    def publish(self, vx: float, vy: float, vz: float):
        # Seqlock: seq_head odd (in progress) -> payload + tail -> seq_head
        # even (stable). Three writes; reader detects any mid-copy state.
        s_write  = self.seq + 1   # odd:  write in progress
        s_stable = self.seq + 2   # even: stable
        self.shm.seek(0)
        self.shm.write(struct.pack('=Q', s_write))
        self.shm.seek(8)
        self.shm.write(struct.pack('=ddd?7xQ', vx, vy, vz, True, s_stable))
        self.shm.seek(0)
        self.shm.write(struct.pack('=Q', s_stable))
        self.seq = s_stable

    def close(self):
        self.shm.close()

if __name__ == "__main__":
    pub = VLASharedMemoryPublisher()
    print("Publishing dummy VLA commands (CTRL+C to stop)...")
    try:
        while True:
            # Simulate a VLA generating an adversarial downward vector
            pub.publish(2.0, 0.0, -5.0)
            time.sleep(0.5) # 2 Hz
    except KeyboardInterrupt:
        pub.close()
