import random
import pickle
import numpy as np

""" always training player 1"""
experience_rate=0.2
learning_rate=0.1 #should decrease as it continues to gain a larger knowledge base.
decay_gamma=0.9 #big gamma means thinking long term
starting_player=1
n=8

file_name = 'test'
policy_player1='pol'+'test'
list_of_winners=[]
class seq_game:
    def __init__(self, n):
        self.n=n
        self.IsEnd=False
        self.Player=starting_player
        self.board=''
        self.exp_rate=experience_rate
        self.p1_states=[]
        self.lr = learning_rate  # should decrease as it continues to gain a larger knowledge base
        self.decay_gamma = decay_gamma  # big gamma means thinking long term
        self.states_value = {}  # dict for storing {board, weight}

    def check_for_losing_move(self, board):
        """
        Checks whether adding 0 or 1 are losing moves
        :return returns a list of losing moves, i.e either [], [0], [1], [0,1]:
        """
        current_board=board
        list_of_losers=[]
        for i in [0,1]:
            possible_loser=current_board+str(i)
            substring_to_test=possible_loser[(-self.n):]
            if substring_to_test in current_board:
                list_of_losers.append(i)
        return list_of_losers
    def check_for_winning_move(self):
        """
        Assumes there are two non-losing moves. Checks whether one of these moves forces next player to lose
        :return:
        """
        winning_move=[]
        for i in [0,1]:
            temp_board=self.board+str(i)
            if len(self.check_for_losing_move(temp_board))==2:
                winning_move.append(i)
        return winning_move
    def change_player(self):
        """
        Function for changing current player
        :return: returns none
        """
        if self.Player==1:
            self.Player=2
        else:
            self.Player=1
    def update_board(self, move):
        """
        updates board
        :param move is a string:
        :return: returns none
        """
        self.board+=move
    def reset(self):
        """
        Resets the game
        :return: returns none
        """
        self.board = '0'
        self.IsEnd = False
        self.Player = starting_player
        self.p1_states=[]
    def savePolicy(self):
        """
        function for saving states and weights for use for the trained agent
        :return: returns None
        """
        fw = open('pol' + file_name, 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()
    def feedReward(self, reward):
        """
        function for giving rewards to all board states from the finished game.
        :param reward: reward to give. 1 or 0.
        :return: returns None
        """
        for st in reversed(self.p1_states):  # goes through all saved board states of this game
            if self.states_value.get(
                    st) is None:  # if it's not already in the dictionary (of board states of ALL games)
                self.states_value[st] = 0  # initialise a value for the state
            self.states_value[st] += self.lr * (
                        self.decay_gamma * reward - self.states_value[st])  # update weight for each board state
            reward = self.states_value[st]
    def loadPolicy(self, file):
        """
        function for loading states and weights for the trained agent.
        :param file:
        :return:
        """
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()
    def smartmove(self, current_board):
        """
        function for choosing an action. This is for player 1 already trained.
            :param positions: available moves to make
            :param current_board: board state
            :return: returns action to make
         """
        value_max = -999
        #print(self.board, current_board, 'board')
        for p in [0,1]:
            next_board = current_board+str(p)
            value = 0 if self.states_value.get(next_board) is None else self.states_value.get(next_board)
            if value >= value_max:
                value_max = value
                act = p
        return str(act)
    def normalmove(self):
        """
        :return: returns move
        """
        losing_moves = self.check_for_losing_move(self.board)
        if len(losing_moves) == 1:  # if only one non-losing move
            non_losing_move = list({0, 1} - set(losing_moves))
            move = str(non_losing_move[0])
        elif len(losing_moves) == 0:
            winning_moves = self.check_for_winning_move()
            if len(winning_moves) > 0:
                move = str(winning_moves[0])
            else:
                move = str(random.randint(0, 1))
        else:
            move= None
        return move
    def training_game(self, rounds=10):
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds {}".format(i))
            while not self.IsEnd:
                if self.Player==1:
                    if np.random.uniform(0, 1) <= self.exp_rate:
                        move=self.normalmove()
                    else:
                        move=self.smartmove(self.board)
                elif self.Player==2:
                    move=self.normalmove()
                if move is not None:
                    self.update_board(move)
                else:
                    self.IsEnd = True
                    if self.Player==2:#if player 1 has won
                        self.feedReward(5)
                        self.savePolicy()
                        list_of_winners.append(1)
                    else:
                        list_of_winners.append(2)
                        self.feedReward(-0.1)
                        self.savePolicy()
                if self.Player == 1:
                    self.p1_states.append(self.board)
                self.change_player()
            self.reset()

play=seq_game(n)
play.training_game(10000)