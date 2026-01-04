cmake_minimum_required(VERSION 3.20)
project(ChessSophiaCpp CXX)

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
    engine/tt.cpp
    engine/zobrist.cpp
    engine/uci.cpp
    engine/utils.cpp
    engine/mcts.cpp
    engine/experience.cpp)


target_include_directories(chess_sophia PRIVATE engine)
find_package(OpenMP REQUIRED)

if (OpenMP_CXX_FOUND)
    target_link_libraries(chess_sophia PRIVATE OpenMP::OpenMP_CXX)
endif()



