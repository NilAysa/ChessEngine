#include <iostream>
#include <string>
#include <cstring>
#include <ctime>
#include <cctype>   // zbog std::isdigit

#ifdef _OPENMP
#include <omp.h>
#endif

#include "board.hpp"
#include "fen.hpp"
#include "search.hpp"
#include "movegen.hpp"
#include "evaluation.hpp"
#include "zobrist.hpp"
#include "san.hpp"
#include "bitboards.hpp"
#include "tt.hpp"

void parsePosition(const std::string& command, Board* board);
void getBestMove(Board board, int depth);
void printEngineInfo();

// UCI opcija: broj niti
int UCI_THREADS = 1;

// Iz search.cpp
extern int SEARCH_NODES_SEARCHED;
extern int SEARCH_THREADS_USED;
extern Move SEARCH_BEST_MOVE;

int main() {
    // Init engine
    initZobrist();
    initBitboards();
    initMoveGeneration();
    initEvaluation();

#ifdef _OPENMP
    omp_set_dynamic(0);            // zabranimo OpenMP-u da sam mijenja broj niti
    omp_set_num_threads(UCI_THREADS);
#endif

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
        else if (command.rfind("setoption", 0) == 0) {
            // UCI: setoption name Threads value N
            if (command.find("Threads") != std::string::npos) {
                auto valuePos = command.find("value");
                if (valuePos != std::string::npos) {
                    std::string after = command.substr(valuePos + 5);
                    // preskoci razmake
                    size_t i = 0;
                    while (i < after.size() && after[i] == ' ')
                        ++i;

                    if (i < after.size() &&
                        std::isdigit(static_cast<unsigned char>(after[i]))) {
                        int threads = std::stoi(after.substr(i));
                        if (threads < 1)  threads = 1;
                        if (threads > 64) threads = 64;
                        UCI_THREADS = threads;
#ifdef _OPENMP
                        omp_set_num_threads(UCI_THREADS);
#endif
                    }
                }
            }
        }
        else if (command.rfind("go", 0) == 0) {
            // default dubina ako GUI ne pošalje "depth X"
            int depth = 7;

            // tražimo podstring "depth" u komandi
            auto depthPos = command.find("depth");
            if (depthPos != std::string::npos) {
                // uzmi sve nakon "depth " i pretvori u broj
                // npr. kod "go depth 4" ovo će dati 4
                std::string afterDepth = command.substr(depthPos + 5);
                // preskoči razmake
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

    // bitno: update Zobrist hash-a
    board->hash = hash(*board);
}

void getBestMove(Board board, int depth) {
    int threads_used = 1;

#ifdef _OPENMP
    // ako po tvojoj logici nema paralelizacije, prijavi 1 nit
    if (depth > 3 && UCI_THREADS > 1)
        threads_used = UCI_THREADS;
#endif
#ifdef _OPENMP
    double start = omp_get_wtime();
#else
    std::clock_t start = std::clock();
#endif

    int eval = search(board, depth);

#ifdef _OPENMP
    double end = omp_get_wtime();
    double time_ms = (end - start) * 1000.0;
#else
    std::clock_t end = std::clock();
    double time_ms =
        static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
#endif

    std::cout << "info depth " << depth
        << " time " << static_cast<int>(time_ms)
        << " nodes " << SEARCH_NODES_SEARCHED
        << " score cp " << eval
        << " threads " << threads_used << "\n";

    char san[6]{};
    moveToSan(SEARCH_BEST_MOVE, san);
    std::cout << "bestmove " << san << "\n";
}

void printEngineInfo() {
    std::cout << "option name Threads type spin default 1 min 1 max 64\n";
    std::cout << "uciok\n";
}
