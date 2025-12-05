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
)


target_include_directories(chess_sophia PRIVATE engine)


