#include "san.hpp"
#include <cstdlib>
#include <cstring>
#include <cctype>      
#include "board.hpp"
#include "utils.hpp"   

// prevodi unos poteza iz san formata u lokaciju na ploci 
static inline int squareFrom(char fileCh, char rankCh) {
    fileCh = static_cast<char>(std::tolower(static_cast<unsigned char>(fileCh)));
    if (fileCh < 'a' || fileCh > 'h') return -1;
    if (rankCh < '1' || rankCh > '8') return -1;
    int file = 'h' - fileCh;        
    int rank = (rankCh - '1');      
    return 8 * rank + file;
}
//pretvara string u Move strukturu i izvršava potez
void pushSan(Board* board, const char* san) {
    Move move{};
    sanToMove(*board, &move, san);   
    pushMove(board, move);
}

void sanToMove(const Board& board, Move* move, const char* san) {
    // Očekuje UCI stil: "e2e4" ili "e7e8q"
    // za inicijalizaciju Move strukture
    const size_t len = std::strlen(san);
    move->fromSquare = 0;
    move->toSquare   = 0;
    move->castle     = 0;
    move->promotion  = -1;
    move->pieceType  = -1;   
    move->validation = NOT_VALIDATED;
    move->score      = 0;
    move->exhausted  = false;

    if (len < 4) return;

    // parsiranje ulaznog stringa
    char ffile = static_cast<char>(std::tolower(static_cast<unsigned char>(san[0])));
    char frank = san[1];
    char tfile = static_cast<char>(std::tolower(static_cast<unsigned char>(san[2])));
    char trank = san[3];
    //vraca polja na ploci
    int fromSq = squareFrom(ffile, frank);
    int toSq   = squareFrom(tfile, trank);
    if (fromSq < 0 || toSq < 0) return;

    //dodjela vrijednosti u Move strukturu
    move->fromSquare = fromSq;
    move->toSquare   = toSq;

    // Castling
    if (move->fromSquare == E1 && move->toSquare == G1 && (board.castling & K)) move->castle = K;
    else if (move->fromSquare == E1 && move->toSquare == C1 && (board.castling & Q)) move->castle = Q;
    else if (move->fromSquare == E8 && move->toSquare == G8 && (board.castling & k)) move->castle = k;
    else if (move->fromSquare == E8 && move->toSquare == C8 && (board.castling & q)) move->castle = q;

    // Promocija (5. znak ako postoji)
    if (len >= 5) {
        char p = static_cast<char>(std::tolower(static_cast<unsigned char>(san[4])));
        const bool whiteToMove = (board.turn == WHITE);
        switch (p) {
            case 'q': move->promotion = whiteToMove ? QUEEN_W  : QUEEN_B;  break;
            case 'r': move->promotion = whiteToMove ? ROOK_W   : ROOK_B;   break;
            case 'n': move->promotion = whiteToMove ? KNIGHT_W : KNIGHT_B; break;
            case 'b': move->promotion = whiteToMove ? BISHOP_W : BISHOP_B; break;
            default:  move->promotion = -1;                                break;
        }
    }

    // Odredi tip figure koja stoji na fromSquare
    // Prolazi kroz sve Bitboard-ove
    for (int i = 0; i < 12; ++i) {
        Bitboard bb = *(&board.pawn_W + i);
        if (getBit(bb, move->fromSquare)) {
            move->pieceType = i;
            break;
        }
    }
}

void moveToSan(const Move& move, char san[]) {
    // Castling u UCI zapis
    if (move.castle) {
        if      (move.castle == K) std::strcpy(san, "e1g1");
        else if (move.castle == Q) std::strcpy(san, "e1c1");
        else if (move.castle == k) std::strcpy(san, "e8g8");
        else if (move.castle == q) std::strcpy(san, "e8c8");
        return;
    }

    // from/to koordinata
    san[0] = SQUARE_NAMES[move.fromSquare][0];
    san[1] = SQUARE_NAMES[move.fromSquare][1];
    san[2] = SQUARE_NAMES[move.toSquare][0];
    san[3] = SQUARE_NAMES[move.toSquare][1];

    // Promocija (ako postoji)
    switch (move.promotion) {
        case QUEEN_W:
        case QUEEN_B:  san[4] = 'q'; san[5] = '\0'; return;
        case ROOK_W:
        case ROOK_B:   san[4] = 'r'; san[5] = '\0'; return;
        case BISHOP_W:
        case BISHOP_B: san[4] = 'b'; san[5] = '\0'; return;
        case KNIGHT_W:
        case KNIGHT_B: san[4] = 'n'; san[5] = '\0'; return;
        default:       san[4] = '\0'; return;
    }
}
