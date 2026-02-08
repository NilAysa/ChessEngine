cmake_minimum_required(VERSION 3.20)
project(ChessSophiaCpp CXX)

option(USE_CUDA_NNUE "Enable CUDA NNUE batch eval" OFF)

if (USE_CUDA_NNUE)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(chess_sophia
    engine/bitboards.cpp
    engine/board.cpp
    engine/evaluation.cpp
    engine/fen.cpp
    engine/magics.cpp
    engine/movegen.cpp
    engine/moveorderer.cpp
    engine/san.cpp
    engine/search.cpp
    engine/zobrist.cpp
    engine/uci.cpp
    engine/utils.cpp
    engine/mcts.cpp
    engine/experience.cpp
    engine/batch_eval.cpp
    engine/nnue.cpp
)

# Add CUDA file only when enabled
if (USE_CUDA_NNUE)
    target_sources(chess_sophia PRIVATE engine/nnue_cuda.cu)
    target_compile_definitions(chess_sophia PRIVATE USE_CUDA_NNUE=1)
endif()

target_include_directories(chess_sophia PRIVATE engine)

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    target_link_libraries(chess_sophia PRIVATE OpenMP::OpenMP_CXX)
endif()
