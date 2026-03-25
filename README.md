# GPU-Based-DES

A minimal GPU accelerated Discrete Event Simulation (DES) engine written in CUDA C++

## Tools
CUDA Toolkit >= 12.0
CMake >= 3.25
C++ Compiler C++17
GPU >= Turing

## Building
```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURE="86"
cmake --build build -j$(nproc)
```

## Note:
I do admit, I had to take help from LLM Models for certain implementations and GPU Libraries.
This repo is created for fun and can be used for learning purposes. I am not going to contribute here regularly!