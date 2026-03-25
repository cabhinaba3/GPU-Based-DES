#pragma once
#include <cstdint>
#include <cfloat>

namespace des{
    // Simulation time
    using SimTime = double;
    static constexpr SimTime TIME_INFINITY = DBL_MAX;
    static constexpr SimTime TIME_ZERO = 0.0;

    // Event
    // Keep this small - it lives in GPU Global memory by milliseconds
    // SoA layout is used in EventQueue; this struct is the logical view
    using EventType = uint32_t;

    namespace event_type {
        static constexpr EventType ARRIVAL = 0;
        static constexpr EventType DEPARTURE = 1;
        static constexpr EventType TIMEOUT   = 2;
        static constexpr EventType USER_BASE = 100;   // extend from here
    }

    struct Event {
        SimTime time; // scheduled timestamp
        uint32_t entity_id; // source/ target entity
        EventType type; // event kind
        uint32_t payload; // App-defined index into a side table
    };

    // Engine configuration
    struct EngineConfig {
        SimTime end_time = TIME_INFINITY; // Stop after this sim-time
        uint32_t max_events = 1 << 22; // Hardcap on queue size (~4G)
        uint32_t max_new_per_event = 4; // Max events one handler emits
        uint32_t threads_per_block = 256; 
    };
} // namespace des