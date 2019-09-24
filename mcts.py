    
# Monte Carlo Tree Search 

import numpy as np
import math
import random
import copy
import utils
from numpy.random import choice
import statistics
import time

# tflearn
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression


class MCTSAgent():
    
    run_time = {}

    def __init__ (self, board):
        self.visits = {} # the number of visits to each board node
        self.score_aggregate = {} # the aggregate final score of all playouts from each board node
        #self.C = 1.4 # the amount of exploration (higher number = more exploration)
        #self.C = 1.0 # the amount of exploration (higher number = more exploration)
        self.C = 0.5 # the amount of exploration (higher number = more exploration)
        self.heuristic_value_dict = {}
        
        tf.reset_default_graph() # clear out any old models from memory - needed to load a new model
        self.model = self.network_model(len(board.players), board.width, board.height)
        
        self.training_data_inputs = []
        self.training_data_targets = []
    
    def network_model(self, channels, board_width, board_height):
        
        network = input_data(shape = [None, channels, board_width, board_height], name = 'input') # input layer

        network = conv_2d(network, 64, filter_size = (3, 3), padding = 'same', activation = 'relu') # convolutional layer
        network = conv_2d(network, 128, filter_size = (3, 3), padding = 'same', activation = 'relu') # convolutional layer

        network = max_pool_2d(network, kernel_size = (2, 2)) # max pooling layer

        network = conv_2d(network, 128, filter_size = (3, 3), padding = 'same', activation = 'relu') # convolutional layer

        network = max_pool_2d(network, kernel_size = (2, 2)) # max pooling layer

        network = conv_2d(network, 128, filter_size = (3, 3), padding = 'same', activation = 'relu') # convolutional layer

        network = max_pool_2d(network, kernel_size = (4, 4)) # max pooling layer        

        network = fully_connected(network, 256, activation = 'relu') # fully connected layer
        network = dropout(network, 0.8)
        
        network = fully_connected(network, 1, activation = 'sigmoid') # output layer - use sigmoid, not softmax
        
        #network = regression(network, optimizer = 'adam', learning_rate = 1e-3,
        #                    loss = 'categorical_crossentropy', name = 'target')
        network = regression(network, optimizer = 'adam', learning_rate = 1e-3,
                            loss = 'binary_crossentropy', name = 'target')

        model = tflearn.DNN(network)

        return model

    def record(self, board, score):
        # record training data
        self.training_data_inputs.append(board.get_state())
        self.training_data_targets.append(score)
        
        # update number of visits
        board_hashable = board.get_hashable()

        self.visits.update({board_hashable : (self.visits.get(board_hashable, 0) + 1)}) # increment number of visits to this position
        self.visits.update({'total': (self.visits.get('total', 1) + 1)}) # increment total number of visits
        
        # aggregate score for the position
        self.score_aggregate.update({board_hashable : (self.score_aggregate.get(board_hashable, 0) + score)})

    def clear_record(self):
        self.visits = {} # the number of visits to each board node
        self.score_aggregate = {} # the aggregate final score of all playouts from each board node
        self.training_data_inputs = []
        self.training_data_targets = []

    def train(self):
        # train on the training data

        # reshape the data
        self.training_data_inputs = np.array(self.training_data_inputs)

        self.training_data_targets = np.array(self.training_data_targets)
        self.training_data_targets = self.training_data_targets.reshape(len(self.training_data_targets), -1)
        
        self.model.fit(self.training_data_inputs, self.training_data_targets, n_epoch = 5)

        # clear training data
        self.training_data_inputs = []
        self.training_data_targets = []

        # clear heuristic value dictionary
        self.heuristic_value_dict = {}

    def playout_value(self, board, count = None, end_time = None):
        # play a heuristic-guided random game from the current board to the end
        # update the number of visits and aggregate score for this board as 
        # well as for all children

        if end_time != None:
            cur_time = time.time()
            if cur_time >= end_time:
                return self.heuristic_value(board, board.get_hashable()) # time up, break the recursion and exit

        if board.is_game_over():
            return board.result

        move_value_dict = {} # dictionary of moves and their values
        moves = []
        values = []

        file = open('checkup.txt', 'a')
        file.write('count: %s\n' % count)
        file.write(board.board_string())
        file.write('\n')

        # get values of the moves
        for move in board.get_available_moves():

            board.make_move(move)

            board_hashable = board.get_hashable()

            heuristic_value = self.heuristic_value(board, board_hashable)

            exploration_value = math.sqrt( math.log(self.visits.get('total', 1)) / self.visits.get(board_hashable, 1e-5) )

            board.undo_move(move)
            #exploration_value = 0
            #if heuristic_value == board.player_to_move:
                # instant win
            #    exploration_value = 0

            # ISN'T THERE A PROBLEM WITH THIS????
            value = heuristic_value + (board.player_to_move * self.C * exploration_value)

            file.write('move: %s, h_val: %s, e_val: %s, t_val: %s   ' % (move, heuristic_value, exploration_value, value))

            move_value_dict[move] = value
            moves.append(move)
            values.append(value)

        if board.player_to_move == 1:
            move = max(move_value_dict, key=move_value_dict.get)
        else:
            move = min(move_value_dict, key=move_value_dict.get)

        file.write('chosen move: %s\n' % move)
        file.close()

        f = open('move history.txt', 'a')
        f.write(str(move) + '-')
        f.close()

        # make the move
        board.make_move(move)

        # get move value
        value = self.playout_value(board)

        # record the score
        self.record(board, value)        

        # undo move
        board.undo_move(move)


        return value

    def monte_carlo_value(self, board, num_playouts, max_time = None):
        if max_time == None:
            end_time = None
        else:
            cur_time = time.time()
            end_time = cur_time + max_time

        scores = [self.playout_value(copy.deepcopy(board), count = i, end_time = end_time) for i in range(num_playouts)]

        return np.mean(scores)

    def heuristic_value(self, board, board_hash):
        # return the heuristic value of the board position

        # check hashtable
        value = self.heuristic_value_dict.get(board_hash, None)

        if value == None:
            if board.is_game_over():
                value = board.result
            else:
                value = self.model.predict([board.get_state()])[0][0]
            self.heuristic_value_dict.update({board_hash : value})
        return value
    

    def get_best_move(self, board, console_output = False, num_playouts = 100, max_time = None):
        # the agent returns the best move
        
        action_dict = {}

        self.visits = {} # clear visits

        # time limit
        if max_time != None:
            num_moves = len(board.get_available_moves())
            max_time_each_move = max_time / num_moves
        else:
            max_time_each_move = None

        for move in board.get_available_moves():

            board.make_move(move)

            action_dict[move] = self.monte_carlo_value(board, num_playouts, max_time = max_time_each_move)

            board.undo_move(move)

        if console_output:
            print(action_dict)

        if board.player_to_move == 1:
            return max(action_dict, key = action_dict.get)
        else:
            return min(action_dict, key = action_dict.get)

    # test
    def get_predicted_move(self, board):
        action_dict = {}

        for move in board.get_available_moves():
            board.make_move(move)

            action_dict[move] = self.model.predict([board.get_state()])[0][0]

            board.undo_move(move)

        print(action_dict)

