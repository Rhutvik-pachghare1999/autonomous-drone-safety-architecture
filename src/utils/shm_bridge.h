#pragma once

#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <cstring>

namespace aisp {
namespace utils {

// Struct layout must exactly match Python's struct.pack('=Qddd?')
struct alignas(64) VLACommand {
    std::atomic<uint64_t> sequence_number;
    double vx_nom;
    double vy_nom;
    double vz_nom;
    std::atomic<bool> is_new_data;
    char padding[31]; // Pad to 64 bytes for cache line alignment
};

class VLASharedMemorySubscriber {
private:
    int fd_;
    VLACommand* shm_ptr_;
    uint64_t last_seq_;
    const char* shm_name_ = "/dev/shm/aisp_vla_cmd";
    const size_t size_ = 64;

public:
    VLASharedMemorySubscriber() : last_seq_(0) {
        fd_ = open(shm_name_, O_RDWR);
        if (fd_ == -1) {
            // If the python node hasn't created it, we can create it
            fd_ = open(shm_name_, O_CREAT | O_RDWR, 0666);
            ftruncate(fd_, size_);
        }
        
        shm_ptr_ = (VLACommand*)mmap(NULL, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (shm_ptr_ == MAP_FAILED) {
            perror("mmap failed");
            exit(1);
        }
    }

    ~VLASharedMemorySubscriber() {
        munmap(shm_ptr_, size_);
        close(fd_);
    }

    bool fetch_latest(double& vx, double& vy, double& vz) {
        // Atomic read to avoid torn reads
        if (shm_ptr_->is_new_data.load(std::memory_order_acquire)) {
            uint64_t seq = shm_ptr_->sequence_number.load(std::memory_order_relaxed);
            if (seq > last_seq_) {
                vx = shm_ptr_->vx_nom;
                vy = shm_ptr_->vy_nom;
                vz = shm_ptr_->vz_nom;
                
                last_seq_ = seq;
                
                // Clear the flag
                shm_ptr_->is_new_data.store(false, std::memory_order_release);
                return true;
            }
        }
        return false;
    }
};

} // namespace utils
} // namespace aisp
