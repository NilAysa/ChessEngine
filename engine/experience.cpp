#include "experience.hpp"

#include <fstream>
#include <cmath>
#include <algorithm>

// mali clamp helper (radi u C++11/14)
static double clampd(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

ExperienceBook& ExperienceBook::instance() {
    static ExperienceBook inst;
    return inst;
}

ExperienceBook::MoveKey ExperienceBook::keyFromMove(const Move& m) {
    MoveKey k;
    k.fromSquare = m.fromSquare;
    k.toSquare = m.toSquare;
    k.promotion = m.promotion;
    k.castle = m.castle;
    k.pieceType = m.pieceType;
    return k;
}

size_t ExperienceBook::positionsStored() const {
    std::lock_guard<std::mutex> lock(mtx);
    return book.size();
}

int ExperienceBook::bonusForMove(uint64_t posHash, const Move& m) const {
    std::lock_guard<std::mutex> lock(mtx);

    std::unordered_map<uint64_t, PositionEntry>::const_iterator it = book.find(posHash);
    if (it == book.end()) return 0;

    const MoveKey k = keyFromMove(m);

    std::unordered_map<MoveKey, MoveStat, MoveKeyHash>::const_iterator itM =
        it->second.moves.find(k);

    if (itM == it->second.moves.end()) return 0;

    const MoveStat& s = itM->second;
    if (s.plays <= 0) return 0;

    double mean = s.qSum / (double)s.plays; // cp

    // confidence ~ 0..1 (više plays -> veće povjerenje)
    double conf = std::log((double)s.plays + 1.0) / std::log(50.0);
    conf = clampd(conf, 0.0, 1.0);

    // mean cp -> bonus
    double raw = mean / 2.0;          // 200cp => +100
    raw = clampd(raw, -250.0, 250.0);

    // ✅ NOVO: samo pozitivni bonus (book kao "prior", ne kazna)
    if (raw < 0.0) raw = 0.0;

    int bonus = (int)std::lround(raw * conf);
    return bonus;

}

void ExperienceBook::updateFromRootStats(uint64_t posHash, const std::vector<BookChildStat>& stats) {
    if (stats.empty()) return;

    std::lock_guard<std::mutex> lock(mtx);

    PositionEntry& entry = book[posHash];

    for (size_t i = 0; i < stats.size(); ++i) {
        const BookChildStat& st = stats[i];
        if (st.visits <= 0) continue;

        MoveKey k = keyFromMove(st.m);
        MoveStat& ms = entry.moves[k];

        ms.plays += st.visits;
        ms.qSum += st.meanQ * (double)st.visits;
    }
}

bool ExperienceBook::save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mtx);

    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out) return false;

    // header
    const char magic[4] = { 'E','X','P','B' };
    uint32_t version = 1;
    out.write(magic, 4);
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    uint32_t nPos = (uint32_t)book.size();
    out.write(reinterpret_cast<const char*>(&nPos), sizeof(nPos));

    for (std::unordered_map<uint64_t, PositionEntry>::const_iterator it = book.begin();
        it != book.end(); ++it) {

        uint64_t hash = it->first;
        const PositionEntry& entry = it->second;

        out.write(reinterpret_cast<const char*>(&hash), sizeof(hash));

        uint32_t nMoves = (uint32_t)entry.moves.size();
        out.write(reinterpret_cast<const char*>(&nMoves), sizeof(nMoves));

        for (std::unordered_map<MoveKey, MoveStat, MoveKeyHash>::const_iterator itM = entry.moves.begin();
            itM != entry.moves.end(); ++itM) {

            const MoveKey& k = itM->first;
            const MoveStat& st = itM->second;

            out.write(reinterpret_cast<const char*>(&k.fromSquare), sizeof(int));
            out.write(reinterpret_cast<const char*>(&k.toSquare), sizeof(int));
            out.write(reinterpret_cast<const char*>(&k.promotion), sizeof(int));
            out.write(reinterpret_cast<const char*>(&k.castle), sizeof(int));
            out.write(reinterpret_cast<const char*>(&k.pieceType), sizeof(int));

            out.write(reinterpret_cast<const char*>(&st.plays), sizeof(int));
            out.write(reinterpret_cast<const char*>(&st.qSum), sizeof(double));
        }
    }

    return true;
}

bool ExperienceBook::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(mtx);

    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) return false;

    char magic[4]{};
    in.read(magic, 4);
    if (!in || magic[0] != 'E' || magic[1] != 'X' || magic[2] != 'P' || magic[3] != 'B')
        return false;

    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!in || version != 1) return false;

    uint32_t nPos = 0;
    in.read(reinterpret_cast<char*>(&nPos), sizeof(nPos));
    if (!in) return false;

    book.clear();
    book.reserve(nPos);

    for (uint32_t i = 0; i < nPos; ++i) {
        uint64_t hash = 0;
        in.read(reinterpret_cast<char*>(&hash), sizeof(hash));

        uint32_t nMoves = 0;
        in.read(reinterpret_cast<char*>(&nMoves), sizeof(nMoves));

        PositionEntry entry;
        entry.moves.reserve(nMoves);

        for (uint32_t j = 0; j < nMoves; ++j) {
            MoveKey k{};
            MoveStat st{};

            in.read(reinterpret_cast<char*>(&k.fromSquare), sizeof(int));
            in.read(reinterpret_cast<char*>(&k.toSquare), sizeof(int));
            in.read(reinterpret_cast<char*>(&k.promotion), sizeof(int));
            in.read(reinterpret_cast<char*>(&k.castle), sizeof(int));
            in.read(reinterpret_cast<char*>(&k.pieceType), sizeof(int));

            in.read(reinterpret_cast<char*>(&st.plays), sizeof(int));
            in.read(reinterpret_cast<char*>(&st.qSum), sizeof(double));

            entry.moves.emplace(k, st);
        }

        book.emplace(hash, std::move(entry));
    }

    return true;
}

int ExperienceBook::debugKnownMovesCount(uint64_t posHash) const {
    std::lock_guard<std::mutex> lock(mtx);
    std::unordered_map<uint64_t, PositionEntry>::const_iterator it = book.find(posHash);
    if (it == book.end()) return 0;
    return (int)it->second.moves.size();
}
