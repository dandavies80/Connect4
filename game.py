import numpy as np
import math
import random
from mcts import MCTSAgent

class Board():
	# board for the game

	run_time = {}

	def __init__(self, width = 7, height = 6):

		self.WIN_LINE_SIZE = 4
		self.width = int(width)
		self.height = int(height)
		self.players = [1, -1] # first player and second player
		self.player_marks = {1:'X',-1:'O'}
		self.player_to_move = self.players[0] # player in index [0] of self.players starts
		self.piece_locations = {} # dictionary of integer locations and the piece in that location
		self.column_heights = [0] * self.width
		self.result = None # if the game is over, the result of the game is 1 if X wins, -1 if O wins, and 0 for a draw	
		self.winning_lines = self.get_winning_lines(self.width, self.height)

	def get_winning_lines(self, width, height):
		# returns lists of integer locations that make up winning lines
		winning_lines_list = []
		directions = [[0,1], [1,1], [1,0], [1,-1]] # [col_dir, row_dir] -> vertical up, diagonal up-right, horizontal right, diagonal down-right
		for col in range(width):
			for row in range(height):
				#print('col %s, row %s' % (col, row))
				for cur_dir in directions:
					#print('cur dir [%s, %s]' % (cur_dir[0], cur_dir[1]))
					winning_line = []
					for i in range(self.WIN_LINE_SIZE):
						#print('i = %s: (%s, %s)' % (i, (col + (cur_dir[0]*i)), row + (cur_dir[1]*i)))
						next_col = col + (cur_dir[0]*i)
						next_row = row + (cur_dir[1]*i)
						if next_col < width and 0 <= next_row < height:
							winning_line.append(self.board_coordinates_to_integer_location(next_col, next_row))
						#print(winning_line)
					if len(winning_line) == self.WIN_LINE_SIZE:
						winning_lines_list.append(winning_line)
					#print(winning_lines_list)
		return winning_lines_list

	def setup_position(self, position_string, player_to_move = 1):
		# set up a position based on an input position string
		# position string is a string of marks 'X', 'O', or '-' 
		""" example input for position_string
		position_str =  '-------'
		position_str += '-------'
		position_str += '--O---X'
		position_str += '-XXX--O'
		position_str += '-OXO--O'
		position_str += 'XOXXOXO'
		"""

		self.piece_locations = {} # clear piece locations dictionary
		char_indx = 0
		for row_indx in reversed(range(self.height)):
			for col_indx in range(self.width):
				position_index = self.board_coordinates_to_integer_location(col_indx, row_indx)
				piece = position_string[char_indx]
				for player, mark in self.player_marks.items():
					if piece == mark:
						self.piece_locations[position_index] = player
						break
				char_indx += 1
				
		# artificially set column heights
		for col_indx in range(self.width):
			column_pieces = []
			for row_indx in range(self.height):
				piece = self.piece_locations.get(self.board_coordinates_to_integer_location(col_indx, row_indx), None)
				if piece != None:
					column_pieces.append(piece)
			self.column_heights[col_indx] = len(column_pieces)
		self.player_to_move = player_to_move

	def play_moves(self, moves):
		# play a list moves on the board
		# example input for moves: [2, 6, 3, 1, 5, 4, 9, 10, 16, 23, 0, 8, 15, 13, 17, 20, 27, 11, 18]

		for move in moves:
			self.make_move(move)

	def get_available_moves(self):
		# return a list of the integer locations available to play

		available_moves = []
		for col_indx in range(self.width):
			if self.column_heights[col_indx] < self.height:
				available_moves.append(self.board_coordinates_to_integer_location(col_indx, self.column_heights[col_indx]))
		return available_moves

	def board_coordinates_to_integer_location(self, col, row):
		# convert board coordinates to an integer location
		"""
		Integer Locations
		  col:  0  1  2  3  4  5  6
		row 6: 43 44 45 46 47 48 49
		row 5: 35 37 38 39 40 41 42
		row 4: 28 29 30 31 32 33 34
		row 3: 21 22 23 24 25 26 27
		row 2: 14 15 16 17 18 19 20
		row 1:  7  8  9 10 11 12 13
		row 0:  0  1  2  3  4  5  6
		"""
		return (row * self.width) + col

	def integer_location_to_board_coordinates(self, integer_location):
		# convert an integer location to board coordinates (row, col)
		row = integer_location // self.width
		col = integer_location % self.width
		return col, row

	def get_state(self):
		# return the board state as bit boards of each player's pieces
		# for input into the neural network
		
		# initialize the board state
		# shape is 2 x width x height		
		state = np.zeros((2, self.width, self.height))

		# the first index is for X, second index for O
		for location in self.piece_locations.keys():
			piece = self.piece_locations.get(location)
			if piece == 1:
				player_index = 0 # player X
			else:
				player_index = 1 # player O
			col, row = self.integer_location_to_board_coordinates(location)
			state[player_index][col][row] = 1.0

		return state

	def get_hashable(self):
		# returns a hashable string of the board for use in a dictionary (hash table) lookup

		# get a hashable string of the board
		# the player to move doesn't matter for connect4 because it's not possible to have
		# the same board state with different players to move

		board_str = ''
		for col in range(self.width):
			for row in range(self.height):				
				board_str += str(self.piece_locations.get(self.board_coordinates_to_integer_location(col, row), 0))

		return board_str

	def make_move(self, integer_location):
		# make the move
		self.piece_locations.update({integer_location : self.player_to_move}) # add the piece to the piece location dictionary
		col, row = self.integer_location_to_board_coordinates(integer_location)
		self.column_heights[col] += 1 # increment column height
		self.player_to_move =  -self.player_to_move # switch players

	def undo_move(self, integer_location):
		# undo the move
		self.piece_locations.pop(integer_location) # remove the piece from the piece location dictionary
		col, row = self.integer_location_to_board_coordinates(integer_location)
		self.column_heights[col] -= 1 # decrement column height
		self.player_to_move =  -self.player_to_move # switch players		

	def is_game_over(self):
		# return a boolean representing if the game is over
		# store the result:
		# 1 = first player ('X') won 
		# 0 = second player ('O') won
		# 0.5 = draw or game not over

		# check for winning line
		for line in self.winning_lines:
			line_sum = 0
			for location in line:
				line_sum += self.piece_locations.get(location, 0)
			if line_sum == self.WIN_LINE_SIZE:
				# X wins
				self.result = 1
				return True
			elif line_sum == -self.WIN_LINE_SIZE:
				self.result = 0
				return True

		# check if the board is full
		if len(self.get_available_moves()) == 0:
			self.result = 0.5
			return True

		return False

	def board_string(self):
		# return a string representing the board graphic
		board_str = ''
		for row in reversed(range(self.height)):
			row_str = '|'
			for col in range(self.width):
				integer_location = self.board_coordinates_to_integer_location(col, row)
				piece = self.piece_locations.get(integer_location)
				piece_str = self.player_marks.get(piece, '-')
				row_str += ' ' + piece_str + ' |'
			board_str += '\n' + row_str + '\n'

		board_str += 'Player to move: ' + self.player_marks.get(self.player_to_move) + '\n'
		return board_str

	def print(self):
		# print out the board
		print(self.board_string())

	def move_str(self, move):
		if self.player_to_move == self.players[0]:
			piece = 'X'
		else:
			piece = 'O'
		return piece + ' moves to location ' + str(move)

	def result_str(self):
		if self.result == None:
			return 'The game has not ended.'
		elif self.result == 0.5:
			return 'Tie'
		else:
			return self.player_marks.get(self.result) + ' Wins.' 
class Game():
	# game server

	move_history = []
	result = None

	#def get_model_filepath(self, generation):
	#	return 'models/gen_' + str(generation) + '/model.tflearn'

	def versus_play(self, agent, num_playouts, max_time):
		# lets a user play a game against an agent
		
		print('Versus play')

		game = Game() # game server

		# determine who goes first
		go_first = input('Would you like to go first?  "y" or "n"')

		board = Board()
		if go_first == 'y':
			player = 1
		else:
			player = -1

		while True:
			board.print()

			if board.is_game_over():
				self.result = board.result
				print('Game Over')
				if board.result == player:
					print('You Win!')
				elif board.result == 0.5:
					print('Tie!')
				else:
					print('You Lose')
				return

			if board.player_to_move == player:
				while True:
					move = int(input('Your move.  Enter move (integer location)'))

					# column location

					if move in board.get_available_moves():
						break
					else:
						print('Invalid entry')
			else:
				print('Agent thinking...')
				move = agent.get_best_move(board, num_playouts = 100, max_time = max_time)
				print('Agent move: %s' % move)

			board.make_move(move)

	def self_play(self, agent, num_playouts = 100, console_output = False):    	
    	# self-play a single game, return the moves

		board = Board()
		self.move_history = []

		while True:
			if console_output:
				print('Self-playing game: %s moves played. \r' % (len(self.move_history)), end='', flush=True)

			if board.is_game_over():
				self.result = board.result
				if console_output:
					print('\n')
				return
			else:
				# get best move
				best_move = agent.get_best_move(board, num_playouts = num_playouts)

				# make the move
				board.make_move(best_move)

				self.move_history.append(best_move)


	def play_game(self, gen1, gen2):
		# play game with agent1 first to move
		# return the result: 1 = agent1 wins, -1 = agent2 wins, 0 = draw

		# create agent
		agent = MCTSAgent(Board())	

		# create board
		board = Board()

		# create file manager
		fm = FileManager()

		while True:
			if board.is_game_over():
				self.move_history = board.move_history
				self.result = board.result
				return board.result
	
			if board.player_to_move == 1:

				#agent.model.load(self.get_model_filepath(gen1))
				agent.model.load(fm.get_model_filepath(gen1))
			else:

				#agent.model.load(self.get_model_filepath(gen2))
				agent.model.load(fm.get_model_filepath(gen2))
			best_move = agent.get_best_move(board)

			board.make_move(best_move)

	def play_random_game(self):
		# play a random game for testing

		# create board
		board = Board()
		self.move_history = []
		while True:
			if board.is_game_over():
				return board.result
	
			random_move = random.choice(board.get_available_moves())

			board.make_move(random_move)
			self.move_history.append(random_move)

	def get_game_string(self):
		# create game string to write to file

		# dump all moves in a line at top (for a computer to read)
		moves_str = ''
		for move_indx in range(len(self.move_history)):
			moves_str += str(self.move_history[move_indx]) + ','

		# write moves in columns (for a human to read)
		moves_str_columns = ''
		for move_indx in range(len(self.move_history)):
			move_num = 1 + (move_indx // 2)
			if move_indx % 2 == 0:
				moves_str_columns += str(move_num) + '. ' + str(self.move_history[move_indx])
			else:
				moves_str_columns += ' ' + str(self.move_history[move_indx]) + '\n'

		# result
		if self.result == 1:
			result_str = '1 - 0'
		elif self.result == 0:
			result_str = '0 - 1'
		elif self.result == 0.5:
			result_str = '1/2 - 1/2'
		else:
			result_str = 'No Result'

		# initial board
		board = Board()
		board_positions_str = board.board_string()

		# each board position
		ply_count = 1
		for move in self.move_history:
			move_num = (ply_count+1) // 2

			board_positions_str += 'Move %s: %s moves to %s\n' % (move_num, board.player_marks.get(board.player_to_move, None), move)

			board.make_move(move)

			board_positions_str += board.board_string()
			ply_count += 1
		if board.is_game_over():
			board_positions_str += '\n\nGame Over. Result: %s\n' % board.result

		return moves_str + '\n\n' + moves_str_columns + '\n' + result_str + '\n' + board_positions_str

	

