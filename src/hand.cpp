#include "hand.hpp"
#include <stdexcept>

void Hand::add_card(int card) {
  if (card_size >= MAX_CARDS) {
    throw std::out_of_range("Hand Overflow, too many cards added");
  }
  cards[card_size++] = card;
}

int Hand::pop_card() {
  if (card_size == 0)
    throw std::out_of_range("Can't pop, hand is empty");

  int card = cards[card_size - 1];
  cards[--card_size] = 0;
  return card;
}

HandInfo Hand::get_info() const {
  int ace_count = 0;
  int hand_value = 0;

  for (int i = 0; i < card_size; i++) {
    int val = get_card_value(cards[i]);
    hand_value += val;
    if (val == ACE_VALUE)
      ace_count++;
  }

  while (hand_value > BLACKJACK_VALUE && ace_count > 0) {
    hand_value -= ACE_VALUE - 1;
    ace_count--;
  }

  return {bet, hand_value, ace_count > 0, card_size == 2,
          card_size == 2 && cards[0] == cards[1]};
}

void Hand::reset() {
  for (size_t i = 0; i < MAX_CARDS; i++) {
    cards[i] = 0;
  }
  card_size = 0;
  bet = 1;
}

void HandStack::new_hand(int card) {
  if (hand_size >= MAX_HANDS)
    throw std::out_of_range("Stack Full");

  hands[hand_size].reset();
  hands[hand_size++].add_card(card);
}

void HandStack::split_hand() {
  if (hand_size >= MAX_HANDS)
    throw std::out_of_range("Max Splits Reached");

  Hand &active_hand = hands[hand_size - 1];
  if (active_hand.card_size != 2)
    throw std::runtime_error("Need 2 cards to split");

  // Every hand split gets a bet of 1
  new_hand(active_hand.pop_card());
}

void HandStack::pop_hand() {
  if (hand_size == 0)
    throw std::runtime_error("Can't pop empty hand stack");
  hands[--hand_size].reset();
}

void HandStack::reset() {
  for (size_t i = 0; i < MAX_HANDS; i++) {
    hands[i].reset();
  }
  hand_size = 1;
}