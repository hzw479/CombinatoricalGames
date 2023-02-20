import itertools
import numpy as np
import pickle
from ast import literal_eval as make_tuple
import random

removed_list = []  # for keeping track of which pegs has been removed
file_name = 'PresentationData'
pol1 = 'policy_p1PresentationData'
pol2 = 'policy_p2e1a9g9'

height = 5
width = 5


class sequence_game:
    def __init__(self, p1, p2):
        self.board = []
        self.isEnd = False
        #self.boardHash = None  # for creating a list of board states
        self.p1 = p1
        self.p2 = p2
        self.player = 1  # player 1 starts

    def getHash(self):
        """
        function for getting board state
        :return: returns a list-like-object representing the current board state
        """
        self.boardHash = str(self.board.reshape(self.height * self.width))
        return self.boardHash

    def winner(self):
        """
        :return: returns 1 if player one wins returns -1 if player two wins
        """
        peg_positions = np.argwhere(self.board == 1)
        if len(peg_positions) == 0:  # if no pegs is left
            self.isEnd = True  # game is over
            if self.player == 1:
                return 1  # player 1:is winner
            else:
                return -1  # player 2:is winner
        else:
            self.isEnd = False
            return None


    def change_player(self):
        """
        function for changing player
        :return: returns the number of current player
        """
        if self.player == 1:
            return 2
        else:
            return 1

    def updateState(self, move):
        """
        function to perform the move chosen by the players
        :param move: which move has been chosen to play by the player
        :return: None
        """
        for i in move:
            row = i[0]
            column = i[1]
            self.board[row][column] = 0

    def giveReward(self):
        """
        This function is only used at the end of the game. It tells which player should get a reward for this game.
        :return: returns None
        """
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)

    def reset(self):
        """
        Resets the game
        :return: returns none
        """
        self.board = []
        self.boardHash = None
        self.isEnd = False
        self.player = 1

    def play(self, rounds=100):
        """
        This is the actual game which is used for training player 1.
        :param rounds: how many rounds to play. default is 100.
        :return: returns None
        """
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.available_moves()
                p1_action = self.p1.chooseAction(positions, self.board)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.available_moves()
                    p2_action = self.p2.chooseAction(positions, self.board)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # ended with p2 either win or draw
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

    def play2(self):
        """
        This function plays one game - player 1 vs human agent. Is supposed to be used after training player 1.
        :return:
        """
        while not self.isEnd:
            # Player 1
            positions = self.available_moves()
            p1_action = self.p1.chooseSmartAction(positions, self.board)
            print('Player', self.player, 'takes action', p1_action)
            # take action and upate board state
            self.updateState(p1_action)
            self.showBoard()
            # check board status if it is end
            win = self.winner()
            self.player = self.change_player()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                elif win == -1:
                    print(self.p2.name, "wins!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.available_moves()
                p2_action = self.p2.chooseAction(positions)
                print('Player', self.player, 'takes action', p2_action)

                self.updateState(p2_action)
                self.showBoard()
                win = self.winner()
                self.player = self.change_player()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    elif win == 1:
                        print(self.p1.name, "wins!")
                    else:
                        print("tie!", win)
                    self.reset()
                    break

    def play3(self):
        """
        This function plays the game with the smart player 1 vs a random player 2. This is supposed to be run after training
        and saves data for all games.
        :return:
        """
        while not self.isEnd:
            # Player 1
            positions = self.available_moves()
            p1_action = self.p1.chooseSmartAction(positions, self.board)
            with open(file_name, 'a') as f:  # saves which move has been made
                f.write(str(p1_action) + ', ')
            # take action and upate board state
            self.updateState(p1_action)
            # check board status if it is end
            win = self.winner()
            self.player = self.change_player()
            if win is not None:
                if win == 1:
                    with open(file_name, 'a') as f:  # saves winner
                        f.write('1, \n')
                elif win == -1:
                    with open(file_name, 'a') as f:  # saves winner
                        f.write('2, ')
                self.reset()
                break

            else:
                # Player 2
                positions = self.available_moves()
                p2_action = self.p2.chooseAction(positions)
                with open(file_name, 'a') as f:  # saves which move has been made
                    f.write(str(p2_action) + ', ')
                self.updateState(p2_action)
                win = self.winner()
                self.player = self.change_player()
                if win is not None:
                    if win == -1:
                        with open(file_name, 'a') as f:  # saves winner
                            f.write('2, \n')
                    elif win == 1:
                        with open(file_name, 'a') as f:  # saves winner
                            f.write('1, \n')
                    self.reset()
                    break

    def showBoard(self):
        """
        function for displaying the board when playing computer vs human
        :return: returns None
        """
        print('##########################')
        for i in range(0, self.height):
            print('----------------------------')
            out = '| '
            for j in range(0, self.width):
                if self.board[i, j] == 1:
                    token = '1'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('----------------------------')


class Player:
    """THIS IS THE PLAYER TO TRAIN AND TO BE SMART"""

    def __init__(self, name, exp_rate=0.1):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.3  # should decrease as it continues to gain a larger knowledge base.
        self.exp_rate = exp_rate  # chance of exploring environment
        self.decay_gamma = 0.9  # big gamma means thinking long term
        self.states_value = {}  # dict for storing {boardhash, weight}

    def getHash(self, board):
        """
        function for getting board state
        :return: returns a list-like-object representing the current board state
        """
        boardHash = str(board.reshape(height * width))
        return boardHash

    def chooseAction(self, positions, current_board):
        """
        function for choosing which action to make. This is used for training
        :param positions: available moves to make
        :param current_board: board state
        :return: returns a chosen action
        """
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            act = positions[idx]
        else:  # take smart action
            value_max = -999
            for p in positions:  # all possible moves
                next_board = current_board.copy()
                for i in p:
                    next_board[i] = 0  # performs a move to a 'fake' board
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(
                    next_boardHash)  # checks the value for fake move
                if value >= value_max:
                    value_max = value
                    act = p  # chooses action with highest state_value
        return act

    def chooseSmartAction(self, positions, current_board):
        """
        function for choosing an action. This is for player 1 already trained.
            :param positions: available moves to make
            :param current_board: board state
            :return: returns action to make
         """
        value_max = -999
        for p in positions:
            next_board = current_board.copy()
            for i in p:
                next_board[i] = 0
            next_boardHash = self.getHash(next_board)
            value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
            if value >= value_max:
                value_max = value
                act = p
        return act

    def addState(self, state):
        """
        function for appending the current board state.
        :param state: board state
        :return: returns None
        """
        self.states.append(state)

    def feedReward(self, reward):
        """
        function for giving rewards to all board states from the finished game.
        :param reward: reward to give. 1 or 0.
        :return: returns None
        """
        for st in reversed(self.states):  # goes through all saved board states of this game
            if self.states_value.get(
                    st) is None:  # if it's not already in the dictionary (of board states of ALL games)
                self.states_value[st] = 0  # initialise a value for the state
            self.states_value[st] += self.lr * (
                        self.decay_gamma * reward - self.states_value[st])  # update weight for each board state
            reward = self.states_value[st]

    def reset(self):
        """resetting states"""
        self.states = []

    def savePolicy(self):
        """
        function for saving states and weights for use for the trained agent
        :return: returns None
        """
        fw = open('policy_' + str(self.name) + file_name, 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        """
        function for loading states and weights for the trained agent.
        :param file:
        :return:
        """
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    """TO BE PLAYED BY HUMAN VS THE TRAINED AGENT"""

    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        """
        function for letting human player choose action
        :param positions: list of possible moves to make
        :return: returns None
        """
        while True:
            print(positions)
            action = input("Input your list of pegs to remove(without hardbrackets):")
            "next 4 lines is to parse what human player gives as input"
            if len(action) < 7:
                action = [make_tuple(action)]
            else:
                action = list(make_tuple(action))
            if action in positions:
                return action
            else:
                print('no such input')

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass


class RandomPlayer:
    """RANDOM PLAYER TO PLAY AGAINST SMART AGENT"""

    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        """
        functin to choose random move
        :param positions: possible moves to make
        :return: returns a random move
        """
        while True:
            return random.choice(positions)

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass


"""BELOW IS WHERE THE GAMES ARE ACTUALLY PLAYED"""
if __name__ == "__main__":
    # TRAINING
    """
    p1 = Player("p1")
    p2 = Player("p2")

    st = Nim_squared(p1, p2) # initialize
    print("training...")
    st.play(100000) #play 500000 training games
    p1.savePolicy() #saves policy for player 1
    p2.savePolicy() #saves policy for player 2
    # PLAY AGAINST HUMAN.

    p1 = Player("computer", exp_rate=0) # Exp_rate is 0 as the agent is trained and should therefore not explore 
    p1.loadPolicy(pol1) # loading board states and weights for player 1
    p2 = HumanPlayer("human")
    st = Nim_squared(p1, p2)# initialize
    st.play2() #PLAY
    
    # PLAY AGAINST RANDOM PLAYER.
    p1 = Player("computer", exp_rate=0)  # Exp_rate is 0 as the agent is trained and should therefore not explore
    p1.loadPolicy(pol1)  # loading board states and weights for player 1
    p2 = HumanPlayer('Human')
    st = Nim_squared(p1, p2)  # initialize
    st.play2()
    # for i in range(100000): # Number of games to play against random player
    #    if i%1000==0:
    #        print('game number:', i)
    #    st.play3()
"""