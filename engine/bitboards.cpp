#include "bitboards.hpp"

//niz od 64 Bitboarda, svaki predstavlja po jedno zauzeto polje
Bitboard SQUARE_BITBOARDS[64];
//bijeli i crni pjesaci na ploci, predefinisane maske
Bitboard RANK_1 = 0xFF00ULL;
Bitboard RANK_7 = 0x00FF000000000000ULL;

//inicijalizira 64 Bitboarda gdje svaki Bitboard ima na po jednom polju od 64 postavljen bit na 1, ostali su nula
//koristi se za provjeru da li je neko polje zauzeto, recimo brze se radi bitwise operacija izmedju nekog Bitboarda i jednog iz ove liste
//nego da se svaki put pristupa memoriji u for petlji ili slicno
void initBitboards() {
    for (int i = 0; i < 64; ++i) {
        SQUARE_BITBOARDS[i] = (1ULL << i);
    }
}
