# cd documents/python/alphazero/connect4
# implementation of AlphaZero for Connect Four

from game import Board, Game
from mcts import MCTSAgent
import random
import os, copy
import numpy as np
import datetime
import filecmp

class FileManager():

	def write_best_generation(self, generation):
		file = open('models/best_generation', 'w+')
		file.write(str(generation))
		file.close()

	def read_best_generation(self):
		file = open('models/best_generation', 'r')
		lines = file.readlines()
		if len(lines) > 0:
			return int(lines[0])
		return None # there is no best generation

	def get_model_filepath(self, generation):
		return 'models/gen_' + str(generation) + '/model.tflearn'

	def get_game_filepath(self, filename):
		return 'games/' + filename + '.txt'

	def write_game(self, game_str, game_filename):
		file = open(game_filename, 'w')
		file.write(game_str)
		file.close()

	def get_latest_generation(self):
		# get the name of the latest model
		gen = 1
		game = Game()
		while True:
			file_str = self.get_model_filepath(gen)
			gen_exists = os.path.exists(file_str + '.index')
			if not gen_exists:
				if gen == 1:
					return None
				else:
					return gen - 1
			gen += 1

def play_game():
	agent = MCTSAgent(Board())

	game = Game() # game server

	fm = FileManager() # file manager

	# get latest model generation
	latest_gen = fm.get_latest_generation()	

	# get current generation and load latest generation model, if there is one
	if latest_gen == None:
		gen = 1 # current generation
	else:
		agent.model.load(fm.get_model_filepath(latest_gen))
		gen = latest_gen # current generation

	game.versus_play(agent)

def train():

	agent = MCTSAgent(Board())

	game = Game() # game server

	fm = FileManager() # file manager

	# get latest model generation
	latest_gen = fm.get_latest_generation()

	# get current generation and load latest generation model, if there is one
	if latest_gen == None:
		gen = 1 # current generation
	else:
		agent.model.load(fm.get_model_filepath(latest_gen))
		gen = latest_gen + 1 # next generation to train
		
	while True:

		# collect training data through self-play
		training_games = 3
		mcts_num_playouts = 100
		
		print('Collecting training data through self-play')

		for game_num in range(1, training_games+1):
			print('Game %s' % game_num)
			game.self_play(agent, num_playouts = mcts_num_playouts, console_output = True)

		# train
		print('Training model - generation %s' % gen)
		agent.train()

		# save model
		agent.model.save(fm.get_model_filepath(gen))
		
		# run a self-play game of the current generation and write it to the self-play file
		print('Running self-play game for records (gen %s)' % gen)
		game.self_play(agent, num_playouts = mcts_num_playouts, console_output = False)
		game_str = game.get_game_string()

		fm.write_game(game_str, fm.get_game_filepath('self_play_gen_' + str(gen)))
		agent.clear_record()

		# if the exact same self-play game was played, training has converged and we're done
		if gen > 1 and filecmp.cmp(fm.get_game_filepath('self_play_gen_' + str(gen)), fm.get_game_filepath('self_play_gen_' + str(gen-1))):
			print('Same self-play game was played.  Training complete. Previous generation: %s' % (gen-1))
			break

		gen += 1

def performance_check():
	
	board = Board()

	num_playouts = 50

	agent = MCTSAgent(Board())
	start_time = datetime.datetime.now()
	#best_move = agent.get_best_move(board, num_playouts = num_playouts)
	best_move = agent.get_best_move(board, num_playouts = num_playouts, max_time = 5)
	finish_time = datetime.datetime.now()
	diff_time = finish_time - start_time
	
	print('total time: %s' % diff_time.total_seconds())

def play_latest_gen():
	agent = MCTSAgent(Board())

	game = Game() # game server

	fm = FileManager() # file manager

	# get latest model generation
	latest_gen = fm.get_latest_generation()	

	# get current generation and load latest generation model, if there is one
	if latest_gen == None:
		gen = 1 # current generation
	else:
		agent.model.load(fm.get_model_filepath(latest_gen))
		gen = latest_gen # current generation
	
	game.versus_play(agent, num_playouts = 25, max_time = 10)



def test_position():
	# to analyze: Gen 3 move 9

	fm = FileManager()

	"""
	# gen 3
	position_str =  '-------'
	position_str += '--X----'
	position_str += '--O-XO-'
	position_str += '--O-XX-'
	position_str += '--XXXO-'
	position_str += 'O-XXOOO'
	player_to_move = -1
	gen = 3
	"""

	"""
	# gen 10
	position_str =  '-------'
	position_str += '---X---'
	position_str += '--OX---'
	position_str += '--OX---'
	position_str += '--XO---'
	position_str += 'O-OXX--'
	player_to_move = -1
	gen = 10
	"""

	"""
	position_str =  '--OX-O-'
	position_str += '--OO-X-'
	position_str += '--XXXO-'
	position_str += '--XOOX-'
	position_str += '-OOXXO-'
	position_str += '-XOXXOX'
	player_to_move = -1
	gen = 28
	"""
	"""
	position_str =  '--XO-X-'
	position_str += '--XX-O-'
	position_str += '--OOOX-'
	position_str += '--OXXO-'
	position_str += '-XXOOX-'
	position_str += '-OXOOXO'
	player_to_move = 1
	gen = 28
	"""

	"""
	position_str =  ''
	position_str += ''
	position_str += ''
	position_str += ''
	position_str += ''
	position_str += ''
	player_to_move = 
	gen = 
	"""

	position_str =  '-------'
	position_str += '-------'
	position_str += '-------'
	position_str += '-------'
	position_str += '-------'
	position_str += '-------'
	player_to_move = 1 
	gen = 7

	agent = MCTSAgent(Board())

	agent.model.load(fm.get_model_filepath(gen))

	board = Board()
	board.setup_position(position_str, player_to_move)

	board.print()

	value = agent.heuristic_value(board, board.get_hash())
	print('value: ', value)
	#best_move = agent.get_best_move(board, console_output = True, num_playouts = 100)
	#print('best move: %s' % best_move)


#performance_check()
#play_latest_gen()
train()
