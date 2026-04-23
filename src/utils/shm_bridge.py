import mmap
import os
import struct
import time

class VLASharedMemoryPublisher:
    """
    Zero-copy IPC bridge for the Vision-Language-Action (VLA) model.
    Writes nominal velocity vectors to a memory-mapped file that the C++
    safety kernel reads instantly.
    
    Struct layout (C-compatible, 64-byte aligned):
    - uint64_t sequence_number (8 bytes)
    - double vx_nom            (8 bytes)
    - double vy_nom            (8 bytes)
    - double vz_nom            (8 bytes)
    - bool is_new_data         (1 byte)
    - 31 bytes padding         (31 bytes)
    Total: 64 bytes
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
        self.seq += 1
        # Pack data: Q (uint64), d (double), d (double), d (double), ? (bool)
        # Using native byte order '='
        data = struct.pack('=Qddd?', self.seq, vx, vy, vz, True)
        self.shm.seek(0)
        self.shm.write(data)
        
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
