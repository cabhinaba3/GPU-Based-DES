#include "des/multigpu_engine.cuh"
#include "des/event_queue.cuh"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <numeric>

//  Error helpers
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                  \
            char buf[256];                                                        \
            std::snprintf(buf, sizeof(buf), "CUDA error %s (%d) at %s:%d",      \
                cudaGetErrorString(_e), static_cast<int>(_e), __FILE__, __LINE__);\
            throw std::runtime_error(buf);                                        \
        }                                                                         \
    } while(0)

namespace des {

//  Device kernels shared with the single-GPU engine

// Scatter AoS Event array → SoA columns (handler input)
__global__ static void k_aos_to_soa(
    const Event* src, uint32_t n,
    SimTime* t, uint32_t* ent, EventType* typ, uint32_t* pay) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    t  [i] = src[i].time;
    ent[i] = src[i].entity_id;
    typ[i] = src[i].type;
    pay[i] = src[i].payload;
}

// Pack SoA columns → AoS Event array (for EventQueue::enqueue_batch)
__global__ static void k_soa_to_aos(
    const SimTime* t, const uint32_t* ent,
    const EventType* typ, const uint32_t* pay,
    Event* dst, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = { t[i], ent[i], typ[i], pay[i] };
}

//  Construction helpers
static std::vector<int> enumerate_devices(){
    int n = 0;
    cudaGetDeviceCount(&n);
    std::vector<int> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    return ids;
}

MultiGPUEngine::MultiGPUEngine(const MultiGPUConfig& cfg): cfg_(cfg){
    if (cfg_.device_ids.empty()) { cfg_.device_ids = enumerate_devices(); }
    if (cfg_.device_ids.empty()){
        throw std::runtime_error("MultiGPUEngine: no CUDA devices found");
    }
    int N = static_cast<int>(cfg_.device_ids.size());
    gpus_.resize(N);

    if (cfg_.enable_peer_access && N > 1) {
        for (int a = 0; a < N; ++a) {
            for (int b = 0; b < N; ++b) {
                if (a == b){ continue; }
                int can = 0;
                cudaDeviceCanAccessPeer(&can, cfg_.device_ids[a], cfg_.device_ids[b]);
                if (can) {
                    cudaSetDevice(cfg_.device_ids[a]);
                    cudaError_t e = cudaDeviceEnablePeerAccess(cfg_.device_ids[b], 0);
                    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                        std::fprintf(stderr, "[MultiGPU] P2P %d→%d failed: %s\n",
                            cfg_.device_ids[a], cfg_.device_ids[b], cudaGetErrorString(e));
                }
            }
        }
    }

    uint32_t cap = cfg_.per_gpu.max_events;
    for (int g = 0; g < N; ++g) { init_gpu(gpus_[g], cfg_.device_ids[g], cap); }
}

void MultiGPUEngine::init_gpu(GPUData& gd, int device_id, uint32_t capacity){
    gd.device_id = device_id;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&gd.stream));

    gd.queue = new EventQueue(capacity);

    // Batch buffers
    gd.batch_cap = capacity;
    CUDA_CHECK(cudaMalloc(&gd.m_batch, capacity * sizeof(Event)));
    CUDA_CHECK(cudaMalloc(&gd.m_b_time, capacity * sizeof(SimTime)));
    CUDA_CHECK(cudaMalloc(&gd.m_b_entity, capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&gd.m_b_type, capacity * sizeof(EventType)));
    CUDA_CHECK(cudaMalloc(&gd.m_b_payload, capacity * sizeof(uint32_t)));

    // Staging — sized at max_new_per_event * capacity (capped)
    uint32_t sc = std::min(capacity * cfg_.per_gpu.max_new_per_event, capacity);
    ensure_stag(gd, sc);
    ensure_incoming(gd, capacity);
}

void MultiGPUEngine::ensure_stag(GPUData& gd, uint32_t needed){
    if (needed <= gd.stag_cap) return;
    CUDA_CHECK(cudaSetDevice(gd.device_id));
    if (gd.m_s_time) {
        cudaFree(gd.m_s_time);
        cudaFree(gd.m_s_entity);
        cudaFree(gd.m_s_type);    
        cudaFree(gd.m_s_payload);
        cudaFree(gd.m_s_count);
        cudaFreeHost(gd.h_s_time); 
        cudaFreeHost(gd.h_s_entity);
        cudaFreeHost(gd.h_s_type); 
        cudaFreeHost(gd.h_s_payload);
    }
    gd.stag_cap = needed;

    CUDA_CHECK(cudaMalloc(&gd.m_s_time, needed * sizeof(SimTime)));
    CUDA_CHECK(cudaMalloc(&gd.m_s_entity, needed * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&gd.m_s_type, needed * sizeof(EventType)));
    CUDA_CHECK(cudaMalloc(&gd.m_s_payload, needed * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&gd.m_s_count, sizeof(int)));

    // Pinned host mirrors for fast DtoH
    CUDA_CHECK(cudaMallocHost(&gd.h_s_time, needed * sizeof(SimTime)));
    CUDA_CHECK(cudaMallocHost(&gd.h_s_entity, needed * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&gd.h_s_type, needed * sizeof(EventType)));
    CUDA_CHECK(cudaMallocHost(&gd.h_s_payload, needed * sizeof(uint32_t)));
}

void MultiGPUEngine::ensure_incoming(GPUData& gd, uint32_t needed){
    if (needed <= gd.incoming_cap) return;
    CUDA_CHECK(cudaSetDevice(gd.device_id));
    if (gd.m_incoming) cudaFree(gd.m_incoming);
    gd.incoming_cap = needed;
    CUDA_CHECK(cudaMalloc(&gd.m_incoming, needed * sizeof(Event)));
}

void MultiGPUEngine::free_gpu(GPUData& gd)
{
    if (!gd.stream) { return; }
    CUDA_CHECK(cudaSetDevice(gd.device_id));
    delete gd.queue;  gd.queue = nullptr;
    cudaFree(gd.m_batch);
    cudaFree(gd.m_b_time);
    cudaFree(gd.m_b_entity);
    cudaFree(gd.m_b_type);
    cudaFree(gd.m_b_payload);

    if (gd.m_s_time) {
        cudaFree(gd.m_s_time);
        cudaFree(gd.m_s_entity);
        cudaFree(gd.m_s_type);
        cudaFree(gd.m_s_payload);
        cudaFree(gd.m_s_count);
        cudaFreeHost(gd.h_s_time);
        cudaFreeHost(gd.h_s_entity);
        cudaFreeHost(gd.h_s_type); 
        cudaFreeHost(gd.h_s_payload);
    }
    if (gd.m_incoming) cudaFree(gd.m_incoming);
    cudaStreamDestroy(gd.stream);
    gd.stream = nullptr;
}

MultiGPUEngine::~MultiGPUEngine(){ for (auto& gd : gpus_) free_gpu(gd); }

//  schedule()  –  route initial events to their owning GPUs
void MultiGPUEngine::schedule(const Event* h_events, uint32_t count){
    if (count == 0) { return; }
    int N = num_gpus();
    // Partition host-side by destination GPU
    std::vector<std::vector<Event>> per_gpu(N);
    for (uint32_t i = 0; i < count; ++i) {
        int dst = default_entity_to_gpu(h_events[i].entity_id, N);
        per_gpu[dst].push_back(h_events[i]);
    }

    for (int g = 0; g < N; ++g) {
        if (per_gpu[g].empty()) continue;
        auto& gd = gpus_[g];
        CUDA_CHECK(cudaSetDevice(gd.device_id));

        uint32_t n = static_cast<uint32_t>(per_gpu[g].size());
        ensure_incoming(gd, n);
        CUDA_CHECK(cudaMemcpy(gd.m_incoming, per_gpu[g].data(), n * sizeof(Event), cudaMemcpyHostToDevice));
        gd.queue->enqueue_batch(gd.m_incoming, n);
    }
}

//  set_user_state()
void MultiGPUEngine::set_user_state(int gpu_idx, void* m_ptr){
    assert(gpu_idx >= 0 && gpu_idx < num_gpus());
    gpus_[gpu_idx].user_state = m_ptr;
}

//  exchange_and_enqueue()
//  Pre: each gd.h_s_* is filled and gd.h_s_count is set.
//  Post: events delivered to each GPU's d_incoming; queue merged.
void MultiGPUEngine::exchange_and_enqueue(
    std::vector<std::vector<Event>>& inbox,
    MultiGPUStats& stats){
    int N = num_gpus();

    for (int g = 0; g < N; ++g) {
        auto& gd = gpus_[g];
        int cnt  = gd.h_s_count;
        if (cnt <= 0) continue;

        for (int i = 0; i < cnt; ++i) {
            uint32_t eid = gd.h_s_entity[i];
            int      dst = default_entity_to_gpu(eid, N);
            inbox[dst].push_back({
                gd.h_s_time   [i],
                eid,
                gd.h_s_type   [i],
                gd.h_s_payload[i]
            });
        }
    }
    for (int g = 0; g < N; ++g) {
        if (inbox[g].empty()){ continue; }
        auto& gd   = gpus_[g];
        uint32_t n = static_cast<uint32_t>(inbox[g].size());

        CUDA_CHECK(cudaSetDevice(gd.device_id));
        ensure_incoming(gd, n);

        CUDA_CHECK(cudaMemcpyAsync(gd.m_incoming, inbox[g].data(), n * sizeof(Event), cudaMemcpyHostToDevice, gd.stream));
        CUDA_CHECK(cudaStreamSynchronize(gd.stream));

        gd.queue->enqueue_batch(gd.m_incoming, n);

        // Accumulate cross-GPU stats
        for (const auto& ev : inbox[g]) {
            int src = default_entity_to_gpu(ev.entity_id, N);
            if (src != g) stats.cross_gpu_events[g]++;
        }
        inbox[g].clear();
    }
}

//  run()  –  main multi-GPU simulation loop
MultiGPUStats MultiGPUEngine::run(HandlerKernelFn handler){
    int N = num_gpus();
    MultiGPUStats stats{};
    stats.events_per_gpu.assign(N, 0);
    stats.cross_gpu_events.assign(N, 0);
    const uint32_t tpb = cfg_.per_gpu.threads_per_block;

    // Reusable per-destination inbox (avoids repeated allocation)
    std::vector<std::vector<Event>> inbox(N);

    while (true) {
        SimTime global_t_min = TIME_INFINITY;
        for (int g = 0; g < N; ++g) {
            CUDA_CHECK(cudaSetDevice(gpus_[g].device_id));
            SimTime lt = gpus_[g].queue->min_time();
            global_t_min = std::min(global_t_min, lt);
        }
        if (global_t_min >= cfg_.per_gpu.end_time || global_t_min == TIME_INFINITY) { break; }
        std::vector<uint32_t> batch_sizes(N, 0);
        for (int g = 0; g < N; ++g) {
            auto& gd = gpus_[g];
            CUDA_CHECK(cudaSetDevice(gd.device_id));

            batch_sizes[g] = gd.queue->dequeue_front_batch(gd.m_batch);
            if (batch_sizes[g] == 0){ continue; }
            uint32_t bs = batch_sizes[g];
            int blk = (bs + tpb - 1) / tpb;
            k_aos_to_soa<<<blk, tpb, 0, gd.stream>>>(
                gd.m_batch, bs,
                gd.m_b_time, gd.m_b_entity, gd.m_b_type, gd.m_b_payload);
            // Reset staging counter
            CUDA_CHECK(cudaMemsetAsync(gd.m_s_count, 0, sizeof(int), gd.stream));
        }
        // Sync all GPUs after queue ops
        for (int g = 0; g < N; ++g) {
            CUDA_CHECK(cudaSetDevice(gpus_[g].device_id));
            CUDA_CHECK(cudaStreamSynchronize(gpus_[g].stream));
        }
        for (int g = 0; g < N; ++g) {
            auto& gd = gpus_[g];
            if (batch_sizes[g] == 0) { continue; }
            CUDA_CHECK(cudaSetDevice(gd.device_id));
            uint32_t bs  = batch_sizes[g];
            uint32_t sc  = bs * cfg_.per_gpu.max_new_per_event;
            sc = std::max(sc, 1u);
            ensure_stag(gd, sc);

            HandlerContext ctx{};
            ctx.in_time = gd.m_b_time;
            ctx.in_entity = gd.m_b_entity;
            ctx.in_type = gd.m_b_type;
            ctx.in_payload = gd.m_b_payload;
            ctx.in_count = bs;

            ctx.out_time = gd.m_s_time;
            ctx.out_entity = gd.m_s_entity;
            ctx.out_type = gd.m_s_type;
            ctx.out_payload = gd.m_s_payload;
            ctx.out_count = gd.m_s_count;
            ctx.out_capacity = gd.stag_cap;

            ctx.user_state = gd.user_state;
            ctx.num_gpus = static_cast<uint32_t>(N);
            ctx.local_gpu_id = static_cast<uint32_t>(g);
            int blk = (bs + tpb - 1) / tpb;
            handler<<<blk, tpb, 0, gd.stream>>>(ctx);
        }
        for (int g = 0; g < N; ++g) {
            auto& gd = gpus_[g];
            if (batch_sizes[g] == 0) { gd.h_s_count = 0; continue; }

            CUDA_CHECK(cudaSetDevice(gd.device_id));
            CUDA_CHECK(cudaStreamSynchronize(gd.stream));
            CUDA_CHECK(cudaGetLastError());

            int h_cnt = 0;
            CUDA_CHECK(cudaMemcpy(&h_cnt, gd.m_s_count, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_cnt < 0) h_cnt = 0;
            if (static_cast<uint32_t>(h_cnt) > gd.stag_cap) {
                std::fprintf(stderr,
                    "[MultiGPU] GPU %d: handler emitted %d events, cap %u. Clipping.\n",
                    g, h_cnt, gd.stag_cap);
                h_cnt = static_cast<int>(gd.stag_cap);
            }
            gd.h_s_count = h_cnt;
            if (h_cnt > 0) {
                // Async DtoH using pinned buffers
                CUDA_CHECK(cudaMemcpyAsync(gd.h_s_time,
                    gd.m_s_time, h_cnt * sizeof(SimTime), cudaMemcpyDeviceToHost, gd.stream));
                CUDA_CHECK(cudaMemcpyAsync(gd.h_s_entity,
                    gd.m_s_entity, h_cnt * sizeof(uint32_t),  cudaMemcpyDeviceToHost, gd.stream));
                CUDA_CHECK(cudaMemcpyAsync(gd.h_s_type,
                    gd.m_s_type, h_cnt * sizeof(EventType),  cudaMemcpyDeviceToHost, gd.stream));
                CUDA_CHECK(cudaMemcpyAsync(gd.h_s_payload,
                    gd.m_s_payload,h_cnt * sizeof(uint32_t),  cudaMemcpyDeviceToHost, gd.stream));
            }
        }
        for (int g = 0; g < N; ++g) {
            CUDA_CHECK(cudaSetDevice(gpus_[g].device_id));
            CUDA_CHECK(cudaStreamSynchronize(gpus_[g].stream));
        }
        exchange_and_enqueue(inbox, stats);
        for (int g = 0; g < N; ++g) {
            stats.events_per_gpu[g] += batch_sizes[g];
            stats.aggregate.events_processed += batch_sizes[g];
        }
        stats.aggregate.steps += 1;
        stats.aggregate.sim_time = global_t_min;
    }
    return stats;
}

} // namespace des
