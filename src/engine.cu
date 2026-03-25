#include "des/engine.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <cstdio>

#define CUDA_CHECK(call) \
    do{ \
        cudaError_t _e = (call); \
        if(_e != cudaSuccess){ \
            char buf[256]; \
            std::snprintf(buf, sizeof(buf), \
                "CUDA error %s (%d) at %s:%d", \
                cudaGetErrorString(_e), static_cast<int>(_e), \
                __FILE__, __LINE__); \
            throw std::runtime_error(buf); \
        } \
    }while(0)

namespace des {
// Kernel: scatter an AoS host-side initial event array onto the GPU
__global__ static void k_aos_to_device(
    const Event* src, 
    uint32_t n,
    SimTime* t,
    uint32_t* ent,
    EventType* typ,
    uint32_t* pay
){
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i>=n) return;
    t[i] = src[i].time;
    ent[i] = src[i].entity_id;
    typ[i] = src[i].type;
    pay[i] = src[i].payload;
}

// Construction
DESEngine::DESEngine(const EngineConfig& cfg): cfg_(cfg), queue_(cfg.max_events){
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

DESEngine::~DESEngine()
{
    if(m_staging_time_) cudaFree(m_staging_time_);
    if(m_staging_entity_) cudaFree(m_staging_entity_);
    if(m_staging_type_) cudaFree(m_staging_type_);
    if(m_staging_payload_) cudaFree(m_staging_payload_);
    if(m_staging_count_) cudaFree(m_staging_count_);

    if(m_batch_time_) cudaFree(m_batch_time_);
    if(m_batch_entity_) cudaFree(m_batch_entity_);
    if(m_batch_type_) cudaFree(m_batch_type_);
    if(m_batch_payload_) cudaFree(m_batch_payload_);

    if(stream_) cudaStreamDestroy(stream_);
}

// ensure_staging() - (re)allocate device buffers for a give batch size
void DESEngine::ensure_staging(uint32_t needed){
    if(needed <= staging_capacity_){ return; }
    // Free up the old memory
    if(m_staging_time_){
        cudaFree(m_staging_time_);
        cudaFree(m_staging_entity_);
        cudaFree(m_staging_type_);
        cudaFree(m_staging_payload_);
        cudaFree(m_staging_count_);
    }
    staging_capacity_ = needed;
    CUDA_CHECK(cudaMalloc(&m_staging_time_, needed*sizeof(SimTime)));
    CUDA_CHECK(cudaMalloc(&m_staging_entity_, needed*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&m_staging_type_, needed*sizeof(EventType)));
    CUDA_CHECK(cudaMalloc(&m_staging_payload_, needed*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&m_staging_count_, sizeof(int)));

    // Allocate batch buffers with the same size
    if(m_batch_time_){
        cudaFree(m_batch_time_);
        cudaFree(m_batch_entity_);
        cudaFree(m_batch_type_);
        cudaFree(m_batch_payload_);
    }
    batch_capacity_ = needed;
    CUDA_CHECK(cudaMalloc(&m_batch_time_, needed * sizeof(SimTime)));
    CUDA_CHECK(cudaMalloc(&m_batch_entity_, needed * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&m_batch_type_, needed * sizeof(EventType)));
    CUDA_CHECK(cudaMalloc(&m_batch_payload_, needed * sizeof(uint32_t)));
}

// schedule() - enqueue initial events from host memory
void DESEngine::schedule(const Event* h_events, uint32_t count){
    if(count == 0){ return; }
    // Temporary device buffer for AoS upload
    Event* m_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&m_temp, count*sizeof(Event)));
    CUDA_CHECK(cudaMemcpy(m_temp, h_events, count*sizeof(Event), cudaMemcpyHostToDevice));

    queue_.enqueue_batch(m_temp, count);
    cudaFree(m_temp);
}
// ─────────────────────────────────────────────────────────────────────────────
//  Staging → EventQueue bridge
//  The staging buffer is AoS-ish (SoA columns); EventQueue.enqueue_batch
//  expects AoS Events.  We reuse a tiny kernel.
// ─────────────────────────────────────────────────────────────────────────────
__global__ static void k_staging_to_aos(
    const SimTime* t,
    const uint32_t* ent,
    const EventType* typ,
    const uint32_t* pay,
    Event* out,
    uint32_t n){
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    out[i].time = t[i];
    out[i].entity_id = ent[i];
    out[i].type = typ[i];
    out[i].payload = pay[i];
}

// Run() - main simulation loop
Stats DESEngine::run(HandlerKernelFn handler){
    Stats stats{};
    // Initialise buffer allocation (max_events is upper bound)
    uint32_t max_staging = cfg_.max_events * cfg_.max_new_per_event;
    max_staging = std::min(max_staging, cfg_.max_events);
    ensure_staging(max_staging);

    // Tmp AoS staging used when merging back
    Event* m_aos_staging = nullptr;
    CUDA_CHECK(cudaMalloc(&m_aos_staging, max_staging*sizeof(Event)));

    while (!queue_.empty()) {
        SimTime t_now = queue_.min_time();
        if (t_now >= cfg_.end_time) break;
        uint32_t batch_sz = queue_.dequeue_front_batch(
            // Dequeue writes into m_batch_* SoA, but the queue API returns AoS.
            // We use a temporary host-visible device buffer.
            m_aos_staging   // reuse as batch output buffer
        );
        if (batch_sz == 0) break;

        // Make sure staging capacity is sufficient for outputs
        uint32_t needed = batch_sz * cfg_.max_new_per_event;
        if (needed > staging_capacity_) { ensure_staging(needed); }

        // Reset atomic counter on device
        CUDA_CHECK(cudaMemsetAsync(m_staging_count_, 0, sizeof(int), stream_));
        //  We need per-event SoA columns for inputs.
        //  For simplicity, scatter the AoS batch buffer into temp SoA.
        //  (A future optimisation: keep batch in SoA from the start.)
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        {
            int tpb    = cfg_.threads_per_block;
            int blocks = (batch_sz + tpb - 1) / tpb;
            k_aos_to_device<<<blocks, tpb, 0, stream_>>>(
                m_aos_staging, batch_sz,
                m_batch_time_, m_batch_entity_, m_batch_type_, m_batch_payload_
            );
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        HandlerContext ctx{};
        ctx.in_time    = m_batch_time_;
        ctx.in_entity  = m_batch_entity_;
        ctx.in_type    = m_batch_type_;
        ctx.in_payload = m_batch_payload_;
        ctx.in_count   = batch_sz;

        ctx.out_time    = m_staging_time_;
        ctx.out_entity  = m_staging_entity_;
        ctx.out_type    = m_staging_type_;
        ctx.out_payload = m_staging_payload_;
        ctx.out_count   = m_staging_count_;
        ctx.out_capacity = staging_capacity_;

        ctx.user_state = user_state_;
        ctx.num_gpus     = 1;
        ctx.local_gpu_id = 0;
        {
            int tpb    = cfg_.threads_per_block;
            int blocks = (batch_sz + tpb - 1) / tpb;
            handler<<<blocks, tpb, 0, stream_>>>(ctx);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaGetLastError());
        
        int h_new_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_new_count, m_staging_count_, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_new_count > 0) {
            uint32_t n_new = static_cast<uint32_t>(h_new_count);
            if (n_new > staging_capacity_) {
                std::fprintf(stderr,
                    "[DES] Warning: handler emitted %u events, cap is %u. "
                    "Truncating.\n", n_new, staging_capacity_);
                n_new = staging_capacity_;
            }
            {
                int tpb    = cfg_.threads_per_block;
                int blocks = (n_new + tpb - 1) / tpb;
                k_staging_to_aos<<<blocks, tpb, 0, stream_>>>(
                    m_staging_time_,
                    m_staging_entity_,
                    m_staging_type_,
                    m_staging_payload_,
                    m_aos_staging,
                    n_new
                );
            }
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            queue_.enqueue_batch(m_aos_staging, n_new);
        }
        stats.events_processed += batch_sz;
        stats.steps += 1;
        stats.sim_time = t_now;
    }
    cudaFree(m_aos_staging);
    return stats;
}
} // namespace des