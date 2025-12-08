#include "zobrist.hpp"
#include <cstdlib>
#include <random>
#include <cstdint>
#include "utils.hpp"


static Bitboard randomBitboard() {
    static std::mt19937_64 rng(std::random_device{}());
    return rng(); //vraca randomizirani 64-bitni broj
}


Bitboard PIECES[12][64];
Bitboard EN_PASSANT[64];
Bitboard CASTLING[16];
Bitboard WHITE_TO_MOVE;

//funkcija koja definise kako se azurira hash
//hash se korsiti kao kombinacija informacija iz Board-a, da se zabilježi neko stanje ploce, koristi se u poredjenjima
//brzo se mijenja stanje, samo XOR starihsa novim vrijednostima
Bitboard hash(const Board& board) {
    //inicijalizacija sa Bitboard jer je svakako to 64-bitni broj
    Bitboard h = 0;

    // upisuje castling pravila
    h ^= CASTLING[board.castling];

    // prolazi korz sve Bitboardove iz Board
    for (int i = 0; i < 12; i++) {
        Bitboard bb = *(&(board.pawn_W) + i); 

        while (bb) {
            //locira polje na kojem se nalazi figura
            int square = countTrailingZeros(bb);
            //upisuje lokaciju u hash
            h ^= PIECES[i][square];
            //uklanja tu figuru sa Bitboarda
            bb &= bb - 1;
        }
    }

    // Upisuje se en passant polje
    if (board.epSquare != -1) {
        h ^= EN_PASSANT[board.epSquare];
    }
    //upisuje se ciji je potez
    if (board.turn) {
        h ^= WHITE_TO_MOVE;
    }

    return h;
}
//inicijalizacija Zobrist hasha, razliciti 64-bitni brojevi osiguravaju da se dva stanja uvijek razlikuju (skoro uvijek)
void initZobrist() {
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 64; j++)
            PIECES[i][j] = randomBitboard();

    for (int i = 0; i < 64; i++)
        EN_PASSANT[i] = randomBitboard();

    for (int i = 0; i < 16; i++)
        CASTLING[i] = randomBitboard();

    WHITE_TO_MOVE = randomBitboard();
}
