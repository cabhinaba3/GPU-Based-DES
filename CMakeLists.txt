# SPDX-License-Identifier: GPL-2.0-only
cmake_minimum_required(VERSION 3.25)
project(tickdes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core cpp/src/bindings/core.cpp)
target_compile_options(_core PRIVATE -Wall -Wextra)
install(TARGETS _core LIBRARY DESTINATION tickdes)
