#pragma once

#include "types.cuh"
#include "event_queue.cuh"

#include <cstdint>
#include <functional>

namespace des{
    // HandlerContext - passed to every device event-handler kernel
    // Handler write new events via the emit helpers below
    // All writes are lock-free through atmocAdd on new_count

struct HandlerContext{
    // input batch
    const SimTime* in_time;
    const uint32_t* in_entity;
    const EventType* in_type;
    const uint32_t* in_payload;
    uint32_t in_count;

    // Output staging buffer
    SimTime* out_time;
    uint32_t* out_entity;
    EventType* out_type;
    uint32_t* out_payload;
    int* out_count; // atomic counter
    uint32_t out_capacity; // hard cap to prevent overflow

    // Simulation state pointer
    void* user_state;

    // Multi-GPU metadata (single GPU engine sets these to 1 and 0)
    uint32_t num_gpus;
    uint32_t local_gpu_id;
};

// Call from device code inside a handler kernel
__device__ __forceinline__
void emit_event(HandlerContext& ctx,
    SimTime time,
    uint32_t entity_id,
    EventType type,
    uint32_t payload=0){
    int slot = atomicAdd(ctx.out_count, 1);
    if(slot < static_cast<int>(ctx.out_capacity)){
        ctx.out_time[slot] = time;
        ctx.out_entity[slot] = entity_id;
        ctx.out_type[slot] = type;
        ctx.out_payload[slot] = payload;
    }
}

// HandlerKernelFn - the users CUDA kernel prototype
// The engine calls it as, user_kernel<<<grid, block, 0, stream>>>(ctx)
//  Typical body:
//      __global__ void my_kernel(des::HandlerContext ctx) {
//          int i = blockIdx.x * blockDim.x + threadIdx.x;
//          if (i >= ctx.in_count) return;
//          // read event i, emit new events via des::emit_event(ctx, ...)
//      }

using HandlerKernelFn = void(*)(HandlerContext);

// Statistics collected by the engine
struct Stats{
    uint64_t events_processed = 0;
    uint64_t steps = 0; // number of time-advanced steps
    SimTime sim_time = 0.0;
};

// DES engine
class DESEngine{
public:
    explicit DESEngine(const EngineConfig& cfg = {});
    ~DESEngine();

    // Disable copy
    DESEngine(const DESEngine&) = delete;
    DESEngine& operator=(const DESEngine&)=delete;

    // Setup
    // Schedule initial events before run()
    void schedule(const Event* h_events, uint32_t count);

    // Provide a device pointer to application state (passed through to kernels)
    void set_user_state(void* m_ptr){ user_state_ = m_ptr; }

    // Run
    // Execute the simulation loop, calling handler for event time-step batch
    Stats run(HandlerKernelFn handler);

    // Inspect
    const EngineConfig& config() const { return cfg_; }

private:
    void ensure_staging(uint32_t needed);

    EngineConfig cfg_;
    EventQueue queue_;

    // Device staging buffer for newly emitted events
    SimTime* m_staging_time_ = nullptr;
    uint32_t* m_staging_entity_ = nullptr;
    EventType* m_staging_type_ = nullptr;
    uint32_t* m_staging_payload_ = nullptr;
    int* m_staging_count_ = nullptr;
    uint32_t staging_capacity_ = 0;

    // Device batch buffer (events popped each step)
    SimTime* m_batch_time_ = nullptr;
    uint32_t* m_batch_entity_ = nullptr;
    EventType* m_batch_type_ = nullptr;
    uint32_t* m_batch_payload_ = nullptr;
    uint32_t batch_capacity_ = 0;

    void* user_state_ = nullptr;

    cudaStream_t stream_ = nullptr;
};
} // namespace des