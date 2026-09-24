# llama-parallel-decision is a developer tool, not a shipped binary. Enable it with
# -DLLAMA_BUILD_DECISION_CLI=ON. The /v1/decision endpoint in llama-server stays always on.
option(LLAMA_BUILD_DECISION_CLI "build llama-parallel-decision, the decision CLI worker (developer tool)" OFF)

# bench-decision is a developer tool, not a shipped binary. Enable it with
# -DLLAMA_BUILD_DECISION_BENCH=ON. To regenerate a baseline report:
#   cmake -B build -DLLAMA_BUILD_DECISION_BENCH=ON && cmake --build build --target bench-decision -j
#   ./build/bin/bench-decision --bench
option(LLAMA_BUILD_DECISION_BENCH "build bench-decision, the decision benchmark (developer tool)" OFF)

# The shared engine is only produced when something links it: llama-server, or one of the opt-in
# developer tools. test-decision-engine compiles the engine sources directly, so a tests-only or
# LLAMA_BUILD_SERVER=OFF build must not compile this library.
if (LLAMA_BUILD_SERVER OR LLAMA_BUILD_DECISION_CLI OR LLAMA_BUILD_DECISION_BENCH)
    # shared engine: used by llama-parallel-decision and by llama-server's /decision endpoint
    add_library(llama-decision STATIC
        decision-engine.cpp decision-engine.h
        decision-protocol.cpp decision-protocol.h
        labels.cpp labels.h
        letter_readout.cpp letter_readout.h)
    target_include_directories(llama-decision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(llama-decision PUBLIC llama-common llama)
    target_compile_features(llama-decision PUBLIC cxx_std_17)
    # llama-server links it into a shared library (libllama-server-impl)
    set_target_properties(llama-decision PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if (LLAMA_BUILD_DECISION_CLI)
    set(TARGET llama-parallel-decision)
    add_executable(${TARGET} parallel-decision.cpp)
    target_link_libraries(${TARGET} PRIVATE llama-decision llama-common llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()

if (LLAMA_BUILD_DECISION_BENCH)
    add_executable(bench-decision bench-decision.cpp)
    target_link_libraries(bench-decision PRIVATE llama-decision llama-common llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(bench-decision PRIVATE cxx_std_17)
endif()
