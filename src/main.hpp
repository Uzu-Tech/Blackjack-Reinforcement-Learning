#pragma once
#include "hand.hpp"
#include "random"
#include <pybind11/pybind11.h>

constexpr int MAX_VALUE = 21;
constexpr int MIN_VALUE = 4;
constexpr int NUM_HAND_VALUES = (MAX_VALUE - MIN_VALUE) + 1;
constexpr int NUM_UPCARDS = 10;

constexpr int HIT = 0;
constexpr int STAND = 1;
constexpr int DOUBLE_DOWN = 2;
constexpr int SPLIT = 3;

using State = int;

struct Result {
  float reward;
  State next_state;
  State split_state; // -1 if no extra hand is made
  bool terminated;
};

class BlackjackEnv {
public:
  BlackjackEnv(int seed) : rng(seed), dist(1, int(CARD_VALUES.size())) {};
  void new_game();
  State get_state() { return get_hand_state(hands.get_hand()); }
  // Action given by policy in python script
  Result play_hand(int action);

private:
  void deal_hand(Hand &hand) { hand.add_card(dist(rng)); }
  Result play_split_hand(Hand &hand);
  void play_dealer_hand();
  State get_hand_state(const Hand& hand);
  float calculate_reward(const HandInfo &hand_info);

  HandStack hands; // Stack to store hands
  Hand dealer_hand;

  std::mt19937 rng;
  std::uniform_int_distribution<int> dist;
};