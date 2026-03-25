#pragma once
#include "types.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace des{
    // EventQueue - GPU-resident priority queue backed by a sorted device_vector

    // Strategy: keep events sorted according by time
        // dequeue_front_batch()
        // enqueue_batch()
    // Re-sorting the whole array after each merge in O(n log n) but simple and cache friendly.

class EventQueue{
public:
    explicit EventQueue(uint32_t capacity);
    // Bulk enqueue
    // Append events from a device buffer and re-sort
    void enqueue_batch(const Event* m_events, uint32_t count);

    // Dequeue
    // Remove and return all events whose time == current minimum
    // Returns the number of events written to m_out (caller must pre-allocate)
    uint32_t dequeue_front_batch(Event* m_out);

    // Inspection
    bool empty() const { return size_ == 0; }
    uint32_t size() const { return size_; }
    SimTime min_time() const;

private:
    void sort();

    thrust::device_vector<SimTime> m_time_;
    thrust::device_vector<uint32_t> m_entity_;
    thrust::device_vector<EventType> m_type_;
    thrust::device_vector<uint32_t> m_payload_;

    uint32_t capacity_;
    uint32_t size_ = 0;
};
} // namespace des

// Todo: for larger n, replace with GPU-friendly heap or calender queue in a future pass
