#pragma once


#include <cstdint>

//kodiranje boja - flag ko je na redu
constexpr int BLACK = 0;
constexpr int WHITE = 1;

//Bitboard je ustvari 64-bitni int, bit za svako polje
using Bitboard = std::uint64_t;

// Tip figure (redoslijed važan jer se koristi pointer aritmetika na Board)
enum Piece {
    PAWN_W, KNIGHT_W, BISHOP_W, ROOK_W, QUEEN_W, KING_W,
    PAWN_B, KNIGHT_B, BISHOP_B, ROOK_B, QUEEN_B, KING_B
};

// Ploča H1..A8 
enum Square {
    H1, G1, F1, E1, D1, C1, B1, A1,
    H2, G2, F2, E2, D2, C2, B2, A2,
    H3, G3, F3, E3, D3, C3, B3, A3,
    H4, G4, F4, E4, D4, C4, B4, A4,
    H5, G5, F5, E5, D5, C5, B5, A5,
    H6, G6, F6, E6, D6, C6, B6, A6,
    H7, G7, F7, E7, D7, C7, B7, A7,
    H8, G8, F8, E8, D8, C8, B8, A8
};
//moguca stanja poteza
enum MoveValidation { NOT_VALIDATED, LEGAL, ILLEGAL };

//moguci ishodi igre
enum GameResult { UN_DETERMINED, WHITE_WIN, BLACK_WIN, DRAW };

// Rokade kao bit-maska
enum Castling { K = 1, Q = 2, k = 4, q = 8 };

//za prolazak korz kolone
enum File { H, G, F, E, D, C, B, A };

//struktura Board
struct Board {
    Bitboard pawn_W;
    Bitboard knight_W;
    Bitboard bishop_W;
    Bitboard rook_W;
    Bitboard queen_W;
    Bitboard king_W;

    Bitboard pawn_B;
    Bitboard knight_B;
    Bitboard bishop_B;
    Bitboard rook_B;
    Bitboard queen_B;
    Bitboard king_B;

    int turn;               // ciji je red na potez
    int castling;           // bitmaska (K|Q|k|q)
    int epSquare;           // en passant polje 
    int halfmoves;          // za 50-move rule
    int fullmoves;          // broj punih poteza
    int whiteKingSq;        // polje bijelog kralja
    int blackKingSq;        // polje crnog kralja

    Bitboard occupancy;         // svi zauzeti
    Bitboard occupancyWhite;    // zauzeti bijelih
    Bitboard occupancyBlack;    // zauzeti crnih
    Bitboard hash;              // Zobrist hash
    Bitboard attacks;           // maske napada protivnika
};

//move sturktura, vazan redoslijed zbog poziva funkcija
struct Move {
    int fromSquare;     // početno polje figure
    int toSquare;       // ciljno polje figure
    int promotion;      // -1 ako nema; inače Piece (QUEEN_W/… zavisno od boje)
    int castle;         // Castling flags (K/Q/k/q) ili 0
    int pieceType;      // koja figura se krece
    int validation;     // MoveValidation (LEGAL/ILLEGAL/NOT_VALIDATED)
    int score;          // za move ordering, koliko je dobar potez
    bool exhausted;     // pomoćno u pretrazi
};

//za imenovanje polja
extern char SQUARE_NAMES[64][3];
