#pragma once

#include <array>

// Nearly impossible to have 15 or more cards
constexpr int MAX_CARDS = 22;
// Nearly impossible to re-split 12 times
constexpr int MAX_HANDS = 12;

constexpr std::array<int, 13> CARD_VALUES = {
    11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10,
};

constexpr int BLACKJACK_VALUE = 21;
constexpr int ACE_VALUE = 11;

constexpr int HARD = 0;
constexpr int SOFT = 1;
constexpr int PAIR = 2;

struct HandInfo {
  int bet;
  int value;
  bool useable_ace;
  bool can_double;
  bool can_split;

  bool bust() const { return value > BLACKJACK_VALUE; };
  bool blackjack() const { return value == BLACKJACK_VALUE && can_double; };
  bool soft_17() const { return useable_ace && value == 17; };
  bool ace_pair() const { return can_split && value == 12; };
};

struct Hand {
  std::array<int, MAX_CARDS> cards{};
  int bet = 1;
  size_t card_size = 0;

  Hand() = default;
  Hand(int card) { add_card(card); };

  static int get_card_value(int card) { return CARD_VALUES[card - 1]; }
  void add_card(int card);
  int pop_card();
  HandInfo get_info() const;
  void reset();
};

struct HandStack {
  std::array<Hand, MAX_HANDS> hands; // Stack to store hands
  size_t hand_size;

  Hand &get_hand() { return hands[hand_size - 1]; };
  void new_hand(int card);
  void split_hand();
  void pop_hand();
  void reset();
};