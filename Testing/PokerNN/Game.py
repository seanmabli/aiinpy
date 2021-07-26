import poker as pkr # Import the poker api
import numpy as np # Import numpy

PlayerNames = 'ThingOne, ThingTwo'.replace(' ', '') # input("Player Names (Seprated By Comma's): ").replace(' ', '')
NumOfPlayers = PlayerNames.count(',') + 1
CardForGame = np.array([str(i) for i in np.random.choice(list(pkr.Card), ((NumOfPlayers * 2) + 5), False)]) # We draw 2 cards for each player and 5 for the table
PlayerInfo = np.concatenate((np.array(PlayerNames.split(','))[:, np.newaxis], np.array(CardForGame[:(NumOfPlayers * 2)].reshape(NumOfPlayers, 2))), axis=1)
PlayerMoney = np.concatenate((np.array([[200, 200]]).T, np.zeros((2, 1))), axis=1)
Flop = CardForGame[(NumOfPlayers * 2):]
Pot = 0
Call = 0

# Preflop
if input(str(PlayerInfo[0, 0]) + ' Press Enter Start Turn') == '':
  print('Cards: ' + str(PlayerInfo[0, 1:3]) + ' Money: ' + str(PlayerMoney[0, 0]) + ' Pot: ' + str(Pot) + ' Bet: ' + str(PlayerMoney[0, 1]))
  print('1. Raise $100')
  print('2. Raise $50')
  print('3. Raise $25')
  print('4. Raise $10')
  print('5. Raise $5')
  print('6. Raise $2')
  print('7. Call $' + str(Call))
  YourChoice = int(input('Your Choice: '))
  if YourChoice in range(0, 7):
    PlayerMoney[0, 1] += np.array([100, 50, 25, 10, 5, 2])[YourChoice]
    PlayerMoney[0, 0] -= np.array([100, 50, 25, 10, 5, 2])[YourChoice]
    Call = PlayerMoney[0, 1]
  if YourChoice == 7:
    PlayerMoney[0, 1] += Call
    PlayerMoney[0, 0] -= Call
    Call = PlayerMoney[0, 1]
  print('Total Bet: ' + str(PlayerMoney[0, 1]))

# print(PlayerInfo)
# print(Flop)

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
