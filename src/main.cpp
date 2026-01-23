#include "main.hpp"
#include "hand.hpp"
#include "pybind11/pybind11.h"

// TODO Replace all constants 

namespace py = pybind11;

void BlackjackEnv::new_game() {
  hands.reset();
  dealer_hand.reset();

  for (size_t i = 0; i < 2; i++) {
    deal_hand(hands.get_hand());
    deal_hand(dealer_hand);
  }

  play_dealer_hand();
}

State BlackjackEnv::get_hand_state(const Hand &hand) {
  auto [_, hand_value, useable_ace, can_double, can_split] = hand.get_info();
  int upcard = Hand::get_card_value(dealer_hand.cards[0]);

  State idx = 0;
  idx = (hand_value - 4); // Player Sum: 4-21 (18 values), Offset by min value
  idx = idx * 10 + (upcard - 2);    // Dealer Card: 2-11 (10 values)
  idx = idx * 2 + (int)useable_ace; // Usable Ace: 0-1 (2 values)
  idx = idx * 2 + (int)can_double;  // Can Double: 0-1 (2 values)
  idx = idx * 2 + (int)can_split;   // Can Split: 0-1 (2 values)
  return idx;
}

float BlackjackEnv::calculate_reward(const HandInfo &hand_info) {
  const HandInfo &dealer_info = dealer_hand.get_info();
  float bet = static_cast<float>(hand_info.bet);

  if (hands.hand_size == 1 && hand_info.blackjack()) {
    if (dealer_info.blackjack())
      return 0.0f;
    return 1.5f;
  }

  if (hand_info.bust())
    return -bet;
  if (dealer_info.bust())
    return bet;
  if (hand_info.value > dealer_info.value)
    return bet;
  if (hand_info.value < dealer_info.value)
    return -bet;
  return 0.0f;
}

void BlackjackEnv::play_dealer_hand() {
  HandInfo info = dealer_hand.get_info();
  while ((!info.bust() && info.value < 17) || info.soft_17()) {
    deal_hand(dealer_hand);
    info = dealer_hand.get_info();
  }
}

Result BlackjackEnv::play_split_hand(Hand& hand) {
  bool ace_pair = hand.get_info().ace_pair();
  hands.split_hand();
  Hand &hand2 = hands.get_hand();

  deal_hand(hand);
  deal_hand(hand2);

  // If ace pair you can only hit one card to each hand
  if (ace_pair) {
    float reward = calculate_reward(hand.get_info()) +
                    calculate_reward(hand2.get_info());
    return {
        reward,
        -1,
        -1,
        true,
    };
  }

  return {
      0,
      get_hand_state(hand2),
      get_hand_state(hand),
      false,
  };
}

Result BlackjackEnv::play_hand(int action) {
  Hand &hand = hands.get_hand();

  if (action == SPLIT) {
    return play_split_hand(hand);
  }

  if (action == DOUBLE_DOWN) {
    deal_hand(hand);
    hand.bet *= 2;
  }

  if (action == HIT) {
    deal_hand(hand);
    HandInfo hand_info = hand.get_info();
    if (!hand_info.bust()) {
      return {
          0,
          get_hand_state(hand),
          -1,
          false,
      };
    }
  }

  // Player stands or has gone bust
  float reward = calculate_reward(hand.get_info());
  // Remove hand since hand is finished
  hands.pop_hand();
  return {
      reward,
      -1,
      -1,
      (hands.hand_size == 0),
  };
}

PYBIND11_MODULE(blackjack_env, m) {
  m.doc() = "Blackjack engine optimized with C++";
  // 1. Bind the Result struct so Python can access .reward, .state, etc.
  py::class_<Result>(m, "Result")
      .def_readonly("reward", &Result::reward)
      .def_readonly("next_state", &Result::next_state)
      .def_readonly("split_state", &Result::split_state)
      .def_readonly("terminated", &Result::terminated);

  // 2. Bind the main Environment
  py::class_<BlackjackEnv>(m, "BlackjackEnv")
      .def(py::init<int>(), py::arg("seed"))
      .def("new_game", &BlackjackEnv::new_game)
      // Ensure get_state is defined in your header!
      .def("get_state", &BlackjackEnv::get_state)
      .def("play_hand", &BlackjackEnv::play_hand, py::arg("action"));
}