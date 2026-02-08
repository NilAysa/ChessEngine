#include "fen.hpp"
#include "board.hpp"
#include "zobrist.hpp"
#include "utils.hpp"
#include "movegen.hpp"
#include <cstdlib>

// Vraca pokazivac na odredjeni Bitboard u Board strukturi
Bitboard* pieceBitboard(Board* board, int pieceType) {
    return reinterpret_cast<Bitboard*>(board) + pieceType;
}

// Resetuje Bitboardove na prazno
void reset(Board* board) {
    Bitboard* bb = reinterpret_cast<Bitboard*>(board);
    for (int i = 0; i < 12; i++) {
        bb[i] = 0;
    }
    board->turn = WHITE;
    board->castling = 0;
    board->epSquare = -1;
}

// Funkcija koja računa sva polja trenutnog igraca koja protivnik moze napasti
Bitboard computeAttacks(Board board) {
    Bitboard attackMask = 0;
    //inicijalizira protivnikovim figurama
    Bitboard pawn   = board.turn ? board.pawn_B   : board.pawn_W;
    Bitboard king   = board.turn ? board.king_B   : board.king_W;
    Bitboard knight = board.turn ? board.knight_B : board.knight_W;
    Bitboard bishop = board.turn ? board.bishop_B : board.bishop_W;
    Bitboard rook   = board.turn ? board.rook_B   : board.rook_W;
    Bitboard queen  = board.turn ? board.queen_B  : board.queen_W;

    //za svaku figuru pronalazi sva polja koja protivnicka figura moze napasti
    while (queen) {
        //daje lokaciju figure na ploci
        int sq = countTrailingZeros(queen);
        //posto se kraljica moze kretati dijagonalno i horizontalno/vertiklano, kombinujemo ta dva bitboarda za njeno kretanje
        Bitboard attacks = getBishopAttacks(sq, board.occupancy);
        attacks |= getRookAttacks(sq, board.occupancy);
        attackMask |= attacks;//upisuje se u masku, tj globalni bitboard
        queen &= queen - 1; //vise kraljica ako se desi promocija pjesaka
    }
    while (bishop) {
        int sq = countTrailingZeros(bishop);
        Bitboard attacks = getBishopAttacks(sq, board.occupancy);
        attackMask |= attacks;
        bishop &= bishop - 1; //prelazak na sljedećeg lovca ako je jos uvijek na ploci
    }
    while (rook) {
        int sq = countTrailingZeros(rook);
        Bitboard attacks = getRookAttacks(sq, board.occupancy);
        attackMask |= attacks;
        rook &= rook - 1;
    }
    while (knight) {
        int sq = countTrailingZeros(knight);
        //unaprijed pripremljeni Bitboard za konja, posto on moze preskakati figure i ima fiksan obrazac kretanja
        attackMask |= KNIGHT_MOVEMENT[sq];
        knight &= knight - 1;
    }
    while (pawn) {
        int sq = countTrailingZeros(pawn);
        if (board.turn) { //pjesaci napadaju dijagonalno pa se koriste unaprijed preracunate maske
            attackMask |= PAWN_B_ATTACKS_EAST[sq];
            attackMask |= PAWN_B_ATTACKS_WEST[sq];
        } else {
            attackMask |= PAWN_W_ATTACKS_EAST[sq];
            attackMask |= PAWN_W_ATTACKS_WEST[sq];
        }
        pawn &= pawn - 1;
    }
    while (king) {
        int sq = countTrailingZeros(king);
        //isto unaprijed preracunati pokreti kralja
        attackMask |= KING_MOVEMENT[sq];
        king &= king - 1;
    }
    //vraca krajnju masku
    return attackMask;
}

//Postavljanje Board strukture na osnovi FEN stringa
void setFen(Board* board, const char* fen) {
    reset(board);
    //kako se krece korz plocu
    int rank = 7;
    int file = 0;
    int piece = -1;
    int turn = WHITE;
    //razmaci oznacavaju sekcije u fen zapisu
    int spacesEncountered = 0;
    //potencijalno enpassant polje
    char enPassantFile = '0';
    char enPassantRank = '0';

    while (*fen) {
        piece = -1;
        //postavljanje vrijendosti varijabli u odnosu na FEN zapis
		switch (*fen) {
            case 'b': 
                piece = BISHOP_B; 
                turn = BLACK;
                enPassantFile = *fen;
                break;
            case 'w':
                turn = WHITE;
                break;
            case 'p': piece = PAWN_B; break;
            case 'n': piece = KNIGHT_B; break;
            case 'r': piece = ROOK_B; break;
            case 'q': piece = QUEEN_B; break;
            case 'k': piece = KING_B; break;
            case 'P': piece = PAWN_W; break;
            case 'N': piece = KNIGHT_W; break;
            case 'B': piece = BISHOP_W; break;
            case 'R': piece = ROOK_W; break;
            case 'Q': piece = QUEEN_W; break;
            case 'K': piece = KING_W; break;

            case 'a':
            case 'c':
            case 'd':
            case 'e':
            case 'f':
            case 'g':
            case 'h': 
                enPassantFile = *fen; 
                break;
            //broj praznih polja, file se pomjera za taj broj
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
                enPassantRank = *fen;
                file += atoi(fen);
                break;
            //prelazak na sljedeci rang
            case '/':
                rank--;
                file = 0;
                fen++;
                continue;
            //prelazak na sljedecu sekciju
            case ' ':
                spacesEncountered++;
                enPassantFile = '0';
                enPassantRank = '0';
                fen++;
                continue;
            case '-':
                fen++;
                continue;
        }

        // ako ima validna figura, racuna se indeks polja gdje se nalazi, pronalazi se pokazivac na njen bitboard i upisuje se bit na njenu lokaciju
        if (spacesEncountered == 0 && piece != -1 && rank >= 0) {
            int square = ((rank+1)*8) - file - 1;
            file++;
            Bitboard* bb = pieceBitboard(board, piece); 
            *bb |= 1ULL << square;
        // upisuje se ciji je potez
        } else if (spacesEncountered == 1) {
            board->turn = turn;

        // Upisuju se castling pravila
        } else if (spacesEncountered == 2) {
            if (piece == KING_W) board->castling |= 1;
            else if (piece == QUEEN_W) board->castling |= 1 << 1;
            else if (piece == KING_B) board->castling |= 1 << 2;
            else if (piece == QUEEN_B) board->castling |= 1 << 3;
        
        // upisuje se enpassant polje
        } else if (spacesEncountered == 3) {
            if (enPassantRank != '0' && enPassantFile != '0') {
                int file = 'h' - enPassantFile;
                int rank = enPassantRank - '1';
                board->epSquare = 8 * rank + file;
            }
        }
		
		fen++;
	}

    // postavljaju se pozicije kraljeva, za provjeru saha
    for (int i = 0; i < 64; i++) {
        if (getBit(board->king_W, i)) board->whiteKingSq = i;
        if (getBit(board->king_B, i)) board->blackKingSq = i;
    }

    // racunanje maski
    computeOccupancyMasks(board);

    // postavljanje hasha
    board->hash = hash(*board);

    // racunanje napada
    board->attacks = computeAttacks(*board);
}
