#pragma once
#include "des/types.cuh"
#include "des/engine.cuh"

#include <vector>
#include <cstdint>

namespace des{
// multiGPUConfig
struct MultiGPUConfig{
    // CUDA device IDs to use. Defaults to all visible devices
    std::vector<int> device_ids;

    // Applied identically to each GPU
    EngineConfig per_gpu;

    // If true, attempt cudaDeviceEnbalePeerAccess between GPU pairs so that cudaMemcpyPeerAsync can skip the host relay (NVLink/PCIe P2P)
    bool enable_peer_access = true;
};

// Entity GPU mapping
// Default: round-robin by entity_id, Override with a custom function if user function has non-uniform entity distributions
using EntityToGPUFn = int (*)(uint32_t entity_id, int num_gpus);

inline __host__ __device__
int default_entity_to_gpu(uint32_t entity_id, int num_gpus){
    return static_cast<int>(entity_id % static_cast<uint32_t>(num_gpus));
}

// MultiGPU stats - per-GPU break down
struct MultiGPUStats {
    Stats aggregate;
    std::vector<uint64_t> events_per_gpu; // events processed on each GPU
    std::vector<uint64_t> cross_gpu_events; // events transferred to each GPU
};

// MultiGPUEngine
// Orchestrates a conservative parallel DES across N CUDA devices
//  Loop per step:
//    1. Host reads each GPU's queue front → global t_min.
//    2. Each GPU pops its front batch (events at t_min).
//    3. Each GPU launches the user handler kernel.
//    4. Staging buffers are DtoH copied; events are partitioned by destination
//       GPU (entity_id % num_gpus) on the host.
//    5. Local slices are HtoD copied back and merged into each GPU's queue.
//    6. Cross-GPU slices are sent to destination GPUs (P2P if available).

class MultiGPUEngine{
public:
    explicit MultiGPUEngine(const MultiGPUConfig& cfg = {});
    ~MultiGPUEngine();

    // Disable copy
    MultiGPUEngine(const MultiGPUEngine&) = delete;
    MultiGPUEngine& operator=(const MultiGPUEngine&) = delete;

    // Setup
    // Schedule initial events from host, Events are auto-routed to their owning GPU by entity_id% num_gpus
    void schedule(const Event* h_events, uint32_t count);

    // Set per-GPU device state pointer, Must already reside on device g
    void set_user_state(int gpu_idx, void* m_ptr);

    // Run
    MultiGPUStats run(HandlerKernelFn handler);

    // Inspection
    int num_gpus()const{ return static_cast<int>(gpus_.size()); }
    const MultiGPUConfig& config() const { return cfg_; }

private:
    // ── Per-GPU state ─────────────────────────────────────────────────────────
    struct GPUData {
        int device_id  = 0;
        void* user_state = nullptr;
        bool peer_ok    = false;   // peer access enabled to every other GPU

        // Device event queue
        EventQueue* queue = nullptr;

        // ── Device buffers (all on this GPU's device) ──
        //  batch: AoS output from dequeue
        Event* m_batch = nullptr;
        uint32_t batch_cap = 0;

        //  batch SoA: handler input columns (scattered from m_batch)
        SimTime* m_b_time = nullptr;
        uint32_t* m_b_entity = nullptr;
        EventType* m_b_type = nullptr;
        uint32_t* m_b_payload = nullptr;

        //  staging SoA: handler output columns
        SimTime* m_s_time = nullptr;
        uint32_t* m_s_entity = nullptr;
        EventType* m_s_type    = nullptr;
        uint32_t* m_s_payload = nullptr;
        int* m_s_count = nullptr;  ///< atomic counter
        uint32_t stag_cap = 0;

        //  incoming AoS: events received from other GPUs / local new events
        Event* m_incoming  = nullptr;
        uint32_t incoming_cap = 0;

        // ── Pinned host mirrors (for DtoH staging copy) ──
        SimTime* h_s_time = nullptr;
        uint32_t*  h_s_entity = nullptr;
        EventType* h_s_type = nullptr;
        uint32_t* h_s_payload = nullptr;
        int h_s_count = 0;

        cudaStream_t stream = nullptr;
    };

    void init_gpu(GPUData& gd, int device_id, uint32_t capacity);
    void free_gpu(GPUData& gd);
    void ensure_stag(GPUData &gd, uint32_t needed);
    void ensure_incoming(GPUData& gd, uint32_t needed);

    // Exchange phase: partition staging-destination GPUs, enqueue everywhere
    void exchange_and_enqueue(
        std::vector<std::vector<Event>>& inbox,
        MultiGPUStats& stats
    );
    MultiGPUConfig cfg_;
    std::vector<GPUData> gpus_;
};
} // namespace des