#include "board.hpp"
#include "movegen.hpp"
#include "zobrist.hpp"
#include "utils.hpp"

#include <iostream>
#include <cstdlib>
#include <cinttypes>
#include <cmath>
#include <fen.hpp>

#define TWO_RANKS 16

//mapa indeksa u sahovsku nomenklaturu
char SQUARE_NAMES[64][3] = {
    "h1", "g1", "f1", "e1", "d1", "c1", "b1", "a1",
    "h2", "g2", "f2", "e2", "d2", "c2", "b2", "a2",
    "h3", "g3", "f3", "e3", "d3", "c3", "b3", "a3",
    "h4", "g4", "f4", "e4", "d4", "c4", "b4", "a4",
    "h5", "g5", "f5", "e5", "d5", "c5", "b5", "a5",
    "h6", "g6", "f6", "e6", "d6", "c6", "b6", "a6",
    "h7", "g7", "f7", "e7", "d7", "c7", "b7", "a7",
    "h8", "g8", "f8", "e8", "d8", "c8", "b8", "a8"
};

//za ispis i maske za castling pravila
char CASTLING_RIGHTS[4][2] = { "K", "Q", "k", "q" };

int ALL_CASTLE_W = 0b0011;
int ALL_CASTLE_B = 0b1100;

//provjerava da li je za trenutni potez moguce postici mat, tj da li se nastavlja igra ili ne
static bool isInsufficientMaterial(const Board& board) {
    //ako ima dovoljno figura moze se nastaviti igra
    if (board.pawn_W || board.pawn_B || board.rook_W || board.rook_B || board.queen_W || board.queen_B)
        return false;

    int knights = 0;
    int bishops = 0;
    Bitboard knightsBB = board.knight_W | board.knight_B;
    Bitboard bishopsBB = board.bishop_W | board.bishop_B;

    while (knightsBB) {
        knightsBB &= knightsBB - 1;
        knights++;
    }
    while (bishopsBB) {
        bishopsBB &= bishopsBB - 1;
        bishops++;
    }
    //broji broj konja i lovaca za potencijalni draw
    if (!bishops && knights) return true;      // KN or KNN vs K
    if (!knights && bishops == 1) return true; // KB vs K

    return false;
}

int result(Board board, Move legal[], int length) {
    if (isInsufficientMaterial(board))
        return DRAW;

    bool noLegalMoves = (length == 0);
    if (noLegalMoves) {
        // Izracunaj napade PROTIVNIKA trenutnog igraca
        Bitboard attackers = computeAttacks(board);

        int kingSq = board.turn ? board.whiteKingSq : board.blackKingSq;
        Bitboard kingInCheck = SQUARE_BITBOARDS[kingSq] & attackers;

        if (kingInCheck)
            return board.turn ? BLACK_WIN : WHITE_WIN; // ako je bijeli na potezu i u šahu → crni win

        return DRAW;
    }

    return UN_DETERMINED;
}

void computeOccupancyMasks(Board* board) {
    //upisuje sve figure bijelog igraca u jedan Bitboard
    board->occupancyWhite = board->pawn_W | board->knight_W | board->bishop_W | board->rook_W | board->queen_W | board->king_W;
    //upisuje sve figure crnog igraca u jedan Bitboard
    board->occupancyBlack = board->pawn_B | board->knight_B | board->bishop_B | board->rook_B | board->queen_B | board->king_B;
    //upisuje sve figure oba igraca u jedan Bitboard
    board->occupancy = board->occupancyBlack | board->occupancyWhite;
}
//funkcija za enpassant potez
static void makeEnPassantMove(Board* board, const Move& move) {
    //gleda se koje je boje pjesak koji uzima protivnickog pjesaka da se moze izracunati indeks zarobljenog pjesaka (rang iznad ili ispod na ploci)
    //zato sto krajnji polozaj friendly pjesaka nije isti kao i polozaj tog zarobljenog pjesaka
    int capturedSquare = board->epSquare + (board->turn ? -8 : 8);

    //uklanjaju se informacije iz zobrist hasha za to polje da je bilo en-passant polje
    board->hash ^= EN_PASSANT[board->epSquare];
    //uklanja se stara pozicija friendly pjesaka
    board->hash ^= PIECES[move.pieceType][move.fromSquare];
    //dodaje se nova pozicija firendly pjesaka
    board->hash ^= PIECES[move.pieceType][move.toSquare];
    //uklanja se suprotnicki pjesak sa ploce
    board->hash ^= PIECES[board->turn ? PAWN_B : PAWN_W][capturedSquare];
    //radi se toggle boje u hashu, ciji je potez
    board->hash ^= WHITE_TO_MOVE;

    //postavljamo pokazivace na ispravne boje, koristi se kasnije u proracunu poteza, da se ne mora provjeravati boja
    Bitboard* friendlyPawns = board->turn ? &(board->pawn_W) : &(board->pawn_B);
    Bitboard* opponentPawns = board->turn ? &(board->pawn_B) : &(board->pawn_W);

    //uklanja se zarobljeni pjesak 
    *opponentPawns = toggleBit(*opponentPawns, capturedSquare);

    // pomjera friendly pjesaka sa stare na novu poziciju
    *friendlyPawns = toggleBit(*friendlyPawns, move.fromSquare);
    *friendlyPawns = setBit(*friendlyPawns, board->epSquare);

    //kraj en-passant poteza
    board->epSquare = -1;
    //zamjena igraca
    board->turn = board->turn ? 0 : 1;
    //racunaju se novi Bitboardovi
    computeOccupancyMasks(board);
}

static void makeCastleMove(Board* board, const Move& move) {
    //funkcija koja definise castling potez tj rokada korištenjem enum vrijednosti
    //bijela kratka rokada
    if (move.castle == K) {
        //brišu se stare pozicije topa i kralja i upisuju nove pozicije u hash
        board->hash ^= PIECES[KING_W][E1];
        board->hash ^= PIECES[KING_W][G1];
        board->hash ^= PIECES[ROOK_W][H1];
        board->hash ^= PIECES[ROOK_W][F1];
        //Za Bitboard se kralj pomjera za dva mjesta, zato >>2
        board->king_W >>= 2;
        //Top se pomjera iza kralja
        board->rook_W = toggleBit(board->rook_W, H1);
        board->rook_W = setBit(board->rook_W, F1);
        //biljezimo novu poziciju kralja (koristi se za provjere da li je sah)
        board->whiteKingSq = G1;
    } else if (move.castle == Q) {
        //bijela duga rokada, ostatak koda na isti princip
        board->hash ^= PIECES[KING_W][E1];
        board->hash ^= PIECES[KING_W][C1];
        board->hash ^= PIECES[ROOK_W][A1];
        board->hash ^= PIECES[ROOK_W][D1];
        board->king_W <<= 2;
        board->rook_W = toggleBit(board->rook_W, A1);
        board->rook_W = setBit(board->rook_W, D1);
        board->whiteKingSq = C1;
    } else if (move.castle == k) {
        //crna kratka rokada
        board->hash ^= PIECES[KING_B][E8];
        board->hash ^= PIECES[KING_B][G8];
        board->hash ^= PIECES[ROOK_B][H8];
        board->hash ^= PIECES[ROOK_B][F8];
        board->king_B >>= 2;
        board->rook_B = toggleBit(board->rook_B, H8);
        board->rook_B = setBit(board->rook_B, F8);
        board->blackKingSq = G8;
    } else if (move.castle == q) {
        //crna duga rokada
        board->hash ^= PIECES[KING_B][E8];
        board->hash ^= PIECES[KING_B][C8];
        board->hash ^= PIECES[ROOK_B][A8];
        board->hash ^= PIECES[ROOK_B][D8];
        board->king_B <<= 2;
        board->rook_B = toggleBit(board->rook_B, A8);
        board->rook_B = setBit(board->rook_B, D8);
        board->blackKingSq = C8;
    }
    //uklanja mogucnost en passant poteza ako je postojala
    if (board->epSquare != -1)
        board->hash ^= EN_PASSANT[board->epSquare];
    //zamjena igraca u hashu
    board->hash ^= WHITE_TO_MOVE;
    //uklanja se iz hasha castling mogucnost
    board->hash ^= CASTLING[board->castling];

    //Mijenja stanje castling prava u Bitboard nakon sto se napravi taj potez i upisujemo ih u hash
    board->castling &= board->turn ? ALL_CASTLE_B : ALL_CASTLE_W;
    board->hash ^= CASTLING[board->castling];
    //racuna se novo stanje ploce, tj azuriraju Bitboardovi za sve figure
    computeOccupancyMasks(board);
    //brise se en passant mogucnost
    board->epSquare = -1;
    //zamjena igraca
    board->turn = board->turn ? 0 : 1;
}

// uklanja protivničku figuru (ako je ima) sa polja toSquare
static void capturePieceIfAny(Board* board, int toSquare) {
    if (board->turn) {
        // bijeli je bio na potezu → protivnik su crne figure
        if (getBit(board->pawn_B, toSquare)) {
            board->pawn_B = toggleBit(board->pawn_B, toSquare);
            board->hash ^= PIECES[PAWN_B][toSquare];
        }
        else if (getBit(board->knight_B, toSquare)) {
            board->knight_B = toggleBit(board->knight_B, toSquare);
            board->hash ^= PIECES[KNIGHT_B][toSquare];
        }
        else if (getBit(board->bishop_B, toSquare)) {
            board->bishop_B = toggleBit(board->bishop_B, toSquare);
            board->hash ^= PIECES[BISHOP_B][toSquare];
        }
        else if (getBit(board->rook_B, toSquare)) {
            board->rook_B = toggleBit(board->rook_B, toSquare);
            board->hash ^= PIECES[ROOK_B][toSquare];

            // castling prava kad se skine crni top
            if (toSquare == H8) board->castling &= 0b1011;
            if (toSquare == A8) board->castling &= 0b0111;
        }
        else if (getBit(board->queen_B, toSquare)) {
            board->queen_B = toggleBit(board->queen_B, toSquare);
            board->hash ^= PIECES[QUEEN_B][toSquare];
        }
    }
    else {
        // crni je bio na potezu → protivnik su bijele figure
        if (getBit(board->pawn_W, toSquare)) {
            board->pawn_W = toggleBit(board->pawn_W, toSquare);
            board->hash ^= PIECES[PAWN_W][toSquare];
        }
        else if (getBit(board->knight_W, toSquare)) {
            board->knight_W = toggleBit(board->knight_W, toSquare);
            board->hash ^= PIECES[KNIGHT_W][toSquare];
        }
        else if (getBit(board->bishop_W, toSquare)) {
            board->bishop_W = toggleBit(board->bishop_W, toSquare);
            board->hash ^= PIECES[BISHOP_W][toSquare];
        }
        else if (getBit(board->rook_W, toSquare)) {
            board->rook_W = toggleBit(board->rook_W, toSquare);
            board->hash ^= PIECES[ROOK_W][toSquare];

            // castling prava kad se skine bijeli top
            if (toSquare == H1) board->castling &= 0b1110;
            if (toSquare == A1) board->castling &= 0b1101;
        }
        else if (getBit(board->queen_W, toSquare)) {
            board->queen_W = toggleBit(board->queen_W, toSquare);
            board->hash ^= PIECES[QUEEN_W][toSquare];
        }
    }
}


void pushMove(Board* board, const Move& move) {

    // funkcija koja primijenjuje bilo koji potez i azurira Board strukturu

    // 1) En passant i rokada se rješavaju posebnim funkcijama
    bool isEnPassantMove = (move.toSquare == board->epSquare) &&
        (move.pieceType == PAWN_W || move.pieceType == PAWN_B);

    if (isEnPassantMove) {
        makeEnPassantMove(board, move);
        return;
    }

    if (move.castle) {
        makeCastleMove(board, move);
        return;
    }

    // 2) Svi ostali normalni potezi

    // skini figuru koja se pomjera sa starog polja iz hash-a
    board->hash ^= PIECES[move.pieceType][move.fromSquare];
    // promijeni stranu na potezu u hash-u
    board->hash ^= WHITE_TO_MOVE;
    // skini stara castling prava iz hash-a
    board->hash ^= CASTLING[board->castling];
    // skini stari en passant kvadrat iz hash-a (ako postoji)
    if (board->epSquare != -1)
        board->hash ^= EN_PASSANT[board->epSquare];

    // više nema starog en-passant polja
    board->epSquare = -1;

    // 3) da li je pješak napravio potez od 2 polja (mogući novi en-passant)
    bool starterPawnMoved =
        (move.pieceType == PAWN_W && (move.fromSquare > A1 && move.fromSquare < H3)) ||
        (move.pieceType == PAWN_B && (move.fromSquare > A6 && move.fromSquare < H8));

    if (starterPawnMoved) {
        int distanceCovered = std::abs(move.fromSquare - move.toSquare);
        if (distanceCovered == TWO_RANKS) {
            // bijeli ide "gore", crni "dolje"
            board->epSquare = (move.pieceType == PAWN_W)
                ? move.fromSquare + 8
                : move.fromSquare - 8;
        }
    }

    // upiši novo en-passant polje u hash (ako postoji)
    if (board->epSquare != -1)
        board->hash ^= EN_PASSANT[board->epSquare];

    // 4) Ažuriraj castling prava na osnovu figure koja se pomjera
    if (move.pieceType == KING_W) {
        // bijeli kralj se pomjerio → gubi K i Q rokadu
        board->castling &= ALL_CASTLE_B;   // zadrži samo crna prava
    }
    else if (move.pieceType == KING_B) {
        // crni kralj se pomjerio → gubi k i q
        board->castling &= ALL_CASTLE_W;   // zadrži samo bijela prava
    }
    else if (move.pieceType == ROOK_W) {
        if (move.fromSquare == A1) board->castling &= 0b1101; // gubi Q
        if (move.fromSquare == H1) board->castling &= 0b1110; // gubi K
    }
    else if (move.pieceType == ROOK_B) {
        if (move.fromSquare == A8) board->castling &= 0b0111; // gubi q
        if (move.fromSquare == H8) board->castling &= 0b1011; // gubi k
    }

    // 5) Nađi tačan bitboard figure koja se pomjera
    Bitboard* pieceThatMoved = nullptr;
    switch (move.pieceType) {
    case PAWN_W:   pieceThatMoved = &board->pawn_W;   break;
    case KNIGHT_W: pieceThatMoved = &board->knight_W; break;
    case BISHOP_W: pieceThatMoved = &board->bishop_W; break;
    case ROOK_W:   pieceThatMoved = &board->rook_W;   break;
    case QUEEN_W:  pieceThatMoved = &board->queen_W;  break;
    case KING_W:   pieceThatMoved = &board->king_W;   break;

    case PAWN_B:   pieceThatMoved = &board->pawn_B;   break;
    case KNIGHT_B: pieceThatMoved = &board->knight_B; break;
    case BISHOP_B: pieceThatMoved = &board->bishop_B; break;
    case ROOK_B:   pieceThatMoved = &board->rook_B;   break;
    case QUEEN_B:  pieceThatMoved = &board->queen_B;  break;
    case KING_B:   pieceThatMoved = &board->king_B;   break;
    default: break;
    }

    // skini figuru sa pocetne pozicije
    if (pieceThatMoved)
        *pieceThatMoved = toggleBit(*pieceThatMoved, move.fromSquare);

    // 6) Upis figure na novo polje / promocija
    if (move.promotion <= 0) {
        // običan potez (bez promocije)
        if (pieceThatMoved)
            *pieceThatMoved = setBit(*pieceThatMoved, move.toSquare);

        // update pozicije kralja, ako je kralj
        if (pieceThatMoved == &board->king_W)      board->whiteKingSq = move.toSquare;
        else if (pieceThatMoved == &board->king_B) board->blackKingSq = move.toSquare;

        // novi položaj figure u hash-u
        board->hash ^= PIECES[move.pieceType][move.toSquare];
    }
    else {
        // PROMOCIJA – postavi novu figuru na toSquare
        Bitboard* targetMask = nullptr;
        switch (move.promotion) {
        case QUEEN_W:  targetMask = &board->queen_W;  break;
        case ROOK_W:   targetMask = &board->rook_W;   break;
        case BISHOP_W: targetMask = &board->bishop_W; break;
        case KNIGHT_W: targetMask = &board->knight_W; break;

        case QUEEN_B:  targetMask = &board->queen_B;  break;
        case ROOK_B:   targetMask = &board->rook_B;   break;
        case BISHOP_B: targetMask = &board->bishop_B; break;
        case KNIGHT_B: targetMask = &board->knight_B; break;
        default: break;
        }

        if (targetMask)
            *targetMask = setBit(*targetMask, move.toSquare);

        board->hash ^= PIECES[move.promotion][move.toSquare];
    }

    // 7) Ukloni eventualnu protivničku figuru na ciljnom polju
    capturePieceIfAny(board, move.toSquare);

    // 8) Upisi nova castling prava u hash
    board->hash ^= CASTLING[board->castling];

    // 9) Zamjena strane na potezu u samom board-u
    board->turn = board->turn ? 0 : 1;

    // 10) Proračun novih occupancy maski
    computeOccupancyMasks(board);
}


//ispisuje izgled ploce prolazeci kroz sve bitboardove, ispisuje ciji je potez, prava na castling, i potencijalno en passant polje
void printBoard(const Board& board) {
    const char pieceSymbols[] = "PNBRQKpnbrqk";//redoslijed bitboardova u strukturi
    std::cout << "\n";

    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int loc = 63 - ((y * 8) + x);
            bool printed = false;

            for (int i = 0; i < 12; i++) {
                Bitboard* bb = ((Bitboard*)&board) + i;
                if (getBit(*bb, loc)) {
                    std::cout << pieceSymbols[i];
                    printed = true;
                    break;
                }
            }

            if (!printed) std::cout << ".";
            std::cout << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Turn: " << (board.turn ? "White" : "Black") << "\n";
    std::cout << "Castling: ";
    for (int i = 0; i < 4; i++) {
        if ((board.castling & (1 << i)) >> i)
            std::cout << CASTLING_RIGHTS[i][0];
    }
    std::cout << "\nEp square: "
              << (board.epSquare == -1 ? "None" : SQUARE_NAMES[board.epSquare])
              << "\n\n";
}
