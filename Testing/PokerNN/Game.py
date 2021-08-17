import poker as pkr # Import the poker api
import numpy as np # Import numpy

PlayerNames = 'ThingOne, ThingTwo'.replace(' ', '') # input("Player Names (Seprated By Comma's): ").replace(' ', '')
NumOfPlayers = PlayerNames.count(',') + 1
CardForGame = np.array([str(i) for i in np.random.choice(list(pkr.Card), ((NumOfPlayers * 2) + 5), False)]) # We draw 2 cards for each player and 5 for the table
PlayerInfo = np.concatenate((np.array(PlayerNames.split(','))[:, np.newaxis], np.array(CardForGame[:(NumOfPlayers * 2)].reshape(NumOfPlayers, 2))), axis=1)
PlayerMoney = np.concatenate((np.array([[200, 200]], dtype=int).T, np.zeros((2, 1), dtype=int)), axis=1)
Flop = CardForGame[(NumOfPlayers * 2):]
Pot = 0
Call = 0

# Preflop
while np.all(PlayerMoney[:, 1] != PlayerMoney[0, 1]):
  for i in range(NumOfPlayers):
    if input(str(PlayerInfo[i, 0]) + ' Press Enter Start Turn') == '':
      print(' Player Bets:')
      for j in range(NumOfPlayers):
        print('  ' + PlayerInfo[j, 0] + "'s Bet: " + str(PlayerMoney[j, 1]))

      print(' Cards: ' + str(PlayerInfo[i, 1:3]))
      print(' Money: ' + str(PlayerMoney[0, 0]))
      print(' Pot: ' + str(Pot))

      a = input(' Raise, Call, Check, Or Fold: ')
      if a == 'Raise':
        b = int(input(' How much: '))
        PlayerMoney[0, 0] -= b # Subtract from the player's money
        PlayerMoney[0, 1] += b # Add to the player's bet
      elif a == 'Call':
        PlayerMoney[0, 1] = Call
      # elif a == 'Fold':

      else:
        print('Error 0')

      Call = PlayerMoney[0, 1]
      print('\n')

'''
import poker as pkr # Import the poker api
import numpy as np # Import numpy

class Poker():
  def __init__(self, NumOfPlayers, Blind, StartingCash):
    self.NumOfPlayers, self.Blind, self.StartingCash = NumOfPlayers, Blind, StartingCash

  def StartGame(self):
    self.CardForGame = np.array([str(i) for i in np.random.choice(list(pkr.Card), ((NumOfPlayers * 2) + 5), False)])
    self.PlayerCards = np.array(CardForGame[:(NumOfPlayers * 2)].reshape(NumOfPlayers, 2))
    self.Flop = CardForGame[(NumOfPlayers * 2):]
    return self.PlayerCards

  def PlayerCards(self):
'''
