#include "utils.hpp"
#include "san.hpp"
#include <iostream>
#include <bitset>
#include <iomanip>


// ispis koraka u SAN formatu
void printMoves(const Move moves[], int length) {
    std::cout << "Moves: " << length << "\n";
    for (int i = 0; i < length; i++) {
        char san[6] = {0};
        moveToSan(moves[i], san);
        std::cout << san << "\n";
    }
}

// Crta sahovsku plocu sa popunjenim poljima Bitboard-a
void printBitboard(Bitboard bb) {
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int loc = 63 - ((y * 8) + x); // H8 = index 0
            std::cout << ((bb & (1ULL << loc)) ? 'x' : '.') << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

// ispusuje direktno vrijednosti bita BItboard-a
void printBits(Bitboard bb) {
    std::bitset<64> bits(bb);
    std::cout << bits << '\n';
}

