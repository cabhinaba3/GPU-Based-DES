# GPU-Based-DES

`tickdes` (working name) is a discrete-event simulation library. Models are written in typed Python and compiled to a C++ engine. CUDA and multi-GPU backends (MPI, NCCL) are planned.

Status: early scaffolding. Nothing simulates yet.

## Building

Needs Python 3.10 or later, CMake 3.25 or later, a C++17 compiler and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run python -c "import tickdes"
```

## Licence

GPL-2.0-only. See `LICENSE`.

## Note:
I do admit, I had to take help from LLM Models for certain implementations and GPU Libraries.
This repo is created for fun and can be used for learning purposes. I am not going to contribute here regularly!
