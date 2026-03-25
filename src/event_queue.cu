#include "des/event_queue.cuh"

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/count.h>

#include <stdexcept>
#include <algorithm>

namespace des {
EventQueue::EventQueue(uint32_t capacity): capacity_(capacity){
    m_time_.reserve(capacity);
    m_entity_.reserve(capacity);
    m_type_.reserve(capacity);
    m_payload_.reserve(capacity);
}
namespace {
struct CompareByTime {
    __device__ bool operator()(const thrust::tuple<SimTime, uint32_t, EventType, uint32_t>& a,
                               const thrust::tuple<SimTime, uint32_t, EventType, uint32_t>& b) const {
        return thrust::get<0>(a) < thrust::get<0>(b);
    }
};
/// Zip-iterator tuple order: (time, entity, type, payload)
auto make_zip(thrust::device_vector<SimTime>&   t,
              thrust::device_vector<uint32_t>&  e,
              thrust::device_vector<EventType>& ty,
              thrust::device_vector<uint32_t>&  p,
              uint32_t n){
    return thrust::make_zip_iterator(thrust::make_tuple(
        t.begin(), e.begin(), ty.begin(), p.begin()
    ));
}

} // anonymous namespace
//  sort()  –  sort all resident events ascending by time
void EventQueue::sort()
{
    if (size_ < 2) return;
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
        m_time_.begin(),
        m_entity_.begin(),
        m_type_.begin(),
        m_payload_.begin()
    ));
    thrust::sort(thrust::device, zip_begin, zip_begin + size_, CompareByTime{});
}
//  enqueue_batch()
void EventQueue::enqueue_batch(const Event* m_events, uint32_t count){
    if (count == 0) return;

    uint32_t new_size = size_ + count;
    if (new_size > capacity_){
        throw std::runtime_error("EventQueue capacity exceeded");
    }
    // Resize (Thrust resize preserves existing data)
    m_time_.resize(new_size);
    m_entity_.resize(new_size);
    m_type_.resize(new_size);
    m_payload_.resize(new_size);

    // Scatter incoming AoS Event array into our SoA columns
    // We do this with a small kernel rather than 4 separate thrust::transforms
    // to avoid 4 passes over m_events.
    auto* raw_time    = thrust::raw_pointer_cast(m_time_.data())    + size_;
    auto* raw_entity  = thrust::raw_pointer_cast(m_entity_.data())  + size_;
    auto* raw_type    = thrust::raw_pointer_cast(m_type_.data())    + size_;
    auto* raw_payload = thrust::raw_pointer_cast(m_payload_.data()) + size_;

    // Use thrust counting_iterator + for_each for the scatter
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(count),
        [m_events, raw_time, raw_entity, raw_type, raw_payload]
        __device__ (uint32_t i){
            const Event& e = m_events[i];
            raw_time   [i] = e.time;
            raw_entity [i] = e.entity_id;
            raw_type   [i] = e.type;
            raw_payload[i] = e.payload;
        }
    );
    size_ = new_size;
    sort();
}

//  min_time()
SimTime EventQueue::min_time() const{
    if (size_ == 0) return TIME_INFINITY;
    return m_time_[0];   // m_time_[0] is a proxy; host-read triggers DtoH
}

//  dequeue_front_batch()
//  Counts how many leading events share the minimum timestamp, copies them
//  out as AoS Events, then erases them from the queue.
__global__ static void scatter_to_aos(
    const SimTime*   time,
    const uint32_t*  entity,
    const EventType* type,
    const uint32_t*  payload,
    Event* out,
    uint32_t n){
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i].time= time[i];
    out[i].entity_id = entity[i];
    out[i].type = type[i];
    out[i].payload = payload[i];
}

uint32_t EventQueue::dequeue_front_batch(Event* m_out)
{
    if (size_ == 0) return 0;
    SimTime t_min = min_time();
    uint32_t batch_size = static_cast<uint32_t>(
        thrust::count_if(
            thrust::device,
            m_time_.begin(),
            m_time_.begin() + size_,
            [t_min] __device__ (SimTime t) { return t == t_min; }
        )
    );

    // Scatter prefix into AoS output
    int tpb = 256;
    int blocks = (batch_size + tpb - 1) / tpb;
    scatter_to_aos<<<blocks, tpb>>>(
        thrust::raw_pointer_cast(m_time_.data()),
        thrust::raw_pointer_cast(m_entity_.data()),
        thrust::raw_pointer_cast(m_type_.data()),
        thrust::raw_pointer_cast(m_payload_.data()),
        m_out,
        batch_size
    );
    cudaDeviceSynchronize();

    // Erase the prefix (shift remaining elements left)
    uint32_t remaining = size_ - batch_size;
    if (remaining > 0) {
        thrust::copy(thrust::device,
                     m_time_.begin()    + batch_size, m_time_.begin()    + size_,
                     m_time_.begin());
        thrust::copy(thrust::device,
                     m_entity_.begin()  + batch_size, m_entity_.begin()  + size_,
                     m_entity_.begin());
        thrust::copy(thrust::device,
                     m_type_.begin()    + batch_size, m_type_.begin()    + size_,
                     m_type_.begin());
        thrust::copy(thrust::device,
                     m_payload_.begin() + batch_size, m_payload_.begin() + size_,
                     m_payload_.begin());
    }
    m_time_.resize(remaining);
    m_entity_.resize(remaining);
    m_type_.resize(remaining);
    m_payload_.resize(remaining);
    size_ = remaining;
    return batch_size;
}

} // namespace des
