#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>

using Hand = std::vector<int>;

constexpr int SEED = 42;

constexpr int ACE_VALUE = 11;
constexpr int BLACKJACK_VALUE = 21;
inline constexpr std::array<int, 13> CARD_VALUE =
    std::array<int, 13>{ACE_VALUE, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10};

struct HandInfo {
  int value;
  bool is_soft;
};

inline std::mt19937 &get_rng() {
  static std::mt19937 rng{SEED};
  return rng;
}

static std::uniform_int_distribution<int> dist(0, 12);
inline int get_random_card() { return dist(get_rng()); };

HandInfo get_hand_info(const Hand &hand);
int get_hand_type(const Hand &hand);
bool has_blackjack(const Hand &hand);
bool is_soft_17(const Hand &hand);

inline int get_card_value(int card) { return CARD_VALUE[card]; };
inline int get_hand_value(const Hand &hand) {
  return get_hand_info(hand).value;
};
inline bool is_bust(const Hand &hand) {
  return get_hand_value(hand) > BLACKJACK_VALUE;
};
