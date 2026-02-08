#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "typedefs.hpp"

// Jednostavan "book" baziran na iskustvu:
// key = zobrist hash pozicije
// value = statistika poteza u toj poziciji (plays, qSum u centipawnima)

struct BookChildStat {
    Move m{};
    int  visits = 0;      // koliko puta je potez istražen u MCTS-u
    double meanQ = 0.0;   // prosjek (centipawns) iz perspektive strane na potezu u toj poziciji
};

class ExperienceBook {
public:
    static ExperienceBook& instance();

    bool load(const std::string& path);
    bool save(const std::string& path) const;

    // bonus koji dodajemo na move.score u ordering-u
    int bonusForMove(uint64_t posHash, const Move& m) const;

    // update nakon search-a: ubaci root child stats u book
    void updateFromRootStats(uint64_t posHash, const std::vector<BookChildStat>& stats);

    // korisno za debug
    size_t positionsStored() const;
    // Debug: vrati bonus za move i broj poznatih poteza u toj poziciji
    int debugKnownMovesCount(uint64_t posHash) const;


private:
    ExperienceBook() = default;

private:
    struct MoveKey {
        int fromSquare = 0;
        int toSquare = 0;
        int promotion = -1;
        int castle = 0;
        int pieceType = 0;

        bool operator==(const MoveKey& o) const {
            return fromSquare == o.fromSquare &&
                toSquare == o.toSquare &&
                promotion == o.promotion &&
                castle == o.castle &&
                pieceType == o.pieceType;
        }
    };

    struct MoveKeyHash {
        size_t operator()(const MoveKey& k) const noexcept {
            // mali hash
            size_t h = 1469598103934665603ull;
            auto mix = [&](int v) {
                h ^= (size_t)v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
                };
            mix(k.fromSquare);
            mix(k.toSquare);
            mix(k.promotion);
            mix(k.castle);
            mix(k.pieceType);
            return h;
        }
    };

    struct MoveStat {
        int plays = 0;
        double qSum = 0.0; // cp * plays
    };

    struct PositionEntry {
        std::unordered_map<MoveKey, MoveStat, MoveKeyHash> moves;
    };

private:
    static MoveKey keyFromMove(const Move& m);

    // map: posHash -> entry
    std::unordered_map<uint64_t, PositionEntry> book;

    mutable std::mutex mtx;
};
