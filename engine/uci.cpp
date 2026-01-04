#include <iostream>
#include <string>
#include <cstring>
#include <ctime>
#include <cctype>   // zbog std::isdigit

#include "board.hpp"
#include "fen.hpp"
#include "search.hpp"
#include "zobrist.hpp"
#include "san.hpp"
#include "bitboards.hpp"
#include "movegen.hpp"
#include "evaluation.hpp"
#include "tt.hpp"
#include "experience.hpp"


void parsePosition(const std::string& command, Board* board);
void getBestMove(Board board, int depth);
void printEngineInfo();

// Iz search.cpp
extern int SEARCH_NODES_SEARCHED;
extern Move SEARCH_BEST_MOVE;

int main() {
    // Init engine
    initZobrist();
    initBitboards();
    initMoveGeneration();
    initEvaluation();
    std::cout << "info string book file: experience_book.bin\n";
    ExperienceBook::instance().load("experience_book.bin");


    std::cin.sync_with_stdio(false);
    std::cout.sync_with_stdio(false);

    const size_t bufferSize = 2000;
    char input[bufferSize];

    Board board{}; // value-initialize
    while (true) {
        std::memset(input, 0, sizeof(input));
        std::cout.flush();

        if (!std::cin.getline(input, bufferSize))
            continue;

        if (input[0] == '\0')
            continue;

        std::string command(input);

        if (command.rfind("isready", 0) == 0) {
            std::cout << "readyok\n";
        }
        else if (command.rfind("position", 0) == 0) {
            parsePosition(command, &board);
        }
        else if (command.rfind("ucinewgame", 0) == 0) {
            parsePosition("position startpos", &board);
        }
        else if (command.rfind("go", 0) == 0) {
            // default dubina ako GUI ne pošalje "depth X"
            int depth = 7;

            auto depthPos = command.find("depth");
            if (depthPos != std::string::npos) {
                std::string afterDepth = command.substr(depthPos + 5);

                size_t i = 0;
                while (i < afterDepth.size() && afterDepth[i] == ' ')
                    ++i;

                if (i < afterDepth.size() &&
                    std::isdigit(static_cast<unsigned char>(afterDepth[i]))) {
                    depth = std::stoi(afterDepth.substr(i));
                }
            }

            getBestMove(board, depth);
        }
        else if (command.rfind("quit", 0) == 0) {
            std::cout << "info string book file: experience_book.bin\n";
            ExperienceBook::instance().save("experience_book.bin");
            break;
        }
        else if (command.rfind("uci", 0) == 0) {
            printEngineInfo();
        }
    }

    return 0;
}

void parsePosition(const std::string& command, Board* board) {
    setFen(board, START_FEN);

    auto fenPos = command.find("fen");
    if (fenPos != std::string::npos) {
        std::string fenStr = command.substr(fenPos + 4);
        setFen(board, fenStr.data());
    }

    auto movesPos = command.find("moves");
    if (movesPos != std::string::npos) {
        std::string movesStr = command.substr(movesPos + 6);
        size_t idx = 0;
        while (idx < movesStr.size()) {
            std::string sanStr;
            while (idx < movesStr.size() && movesStr[idx] != ' ') {
                sanStr.push_back(movesStr[idx]);
                idx++;
            }
            pushSan(board, sanStr.data());
            idx++; // skip space
        }
    }

    board->hash = hash(*board);
}

void getBestMove(Board board, int depth) {
    std::clock_t start = std::clock();

    int eval = search(board, depth);

    std::clock_t end = std::clock();
    double time_ms =
        static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;

    std::cout << "info depth " << depth
        << " time " << static_cast<int>(time_ms)
        << " nodes " << SEARCH_NODES_SEARCHED
        << " score cp " << eval
        << "\n";

    char san[6]{};
    moveToSan(SEARCH_BEST_MOVE, san);
    std::cout << "bestmove " << san << "\n";
}

void printEngineInfo() {
    std::cout << "uciok\n";
}
