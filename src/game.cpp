#include "game.hpp"



HandInfo get_hand_info(const Hand &hand)
{
    int ace_count = 0;
    int hand_value = 0;

    for (int card : hand)
    {
        int val = get_card_value(card);
        hand_value += val;
        if (val == ACE_VALUE)
            ace_count++;
    }

    while (hand_value > BLACKJACK_VALUE && ace_count > 0)
    {
        hand_value -= ACE_VALUE - 1;
        ace_count--;
    }

    return {hand_value, ace_count > 0};
}

int get_hand_type(const Hand &hand)
{
    if (hand.size() == 2 && hand[0] == hand[1])
        return 2;                               // PAIR
    return get_hand_info(hand).is_soft ? 1 : 0; // SOFT or HARD
}

bool is_soft_17(const Hand &hand)
{
    auto [value, is_soft] = get_hand_info(hand);
    return (value == 17 && is_soft);
}

bool has_blackjack(const Hand &hand) {
    if (hand.size() != 2) return false;
    return (get_card_value(hand[0]) + get_card_value(hand[1]) == BLACKJACK_VALUE);
};

PYBIND11_MODULE(game, m) {
  m.doc() = "Blackjack engine optimized with C++";
  // Export your functions
  m.def("get_card_value", &get_card_value,
        "Get the blackjack value of a card rank");
  m.def("get_hand_value", &get_hand_value, "Get total value of a hand");
  m.def("get_hand_type", &get_hand_type, "0: Hard, 1: Soft, 2: Pair");
  m.def("is_bust", &is_bust, "Check if hand is over 21");
  m.def("has_blackjack", &has_blackjack, "Check if initial hand is 21");
  m.def("is_soft_17", &is_soft_17, "Check if hand is a soft 17 (Dealer rule)");
  m.def("get_random_card", &get_random_card, "Deals a random card to a hand");
}