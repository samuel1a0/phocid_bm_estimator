import torch
import os
import numpy as np

class Seq_Net( object ):
	_instance = None

	def __new__( class_, pre_trained=True, path=None ):
		""" Single Instance Creation
		#	pre_trained: Boolean, default=True. wether to 
		#	return an already trained instance or a dummy one.
		#	if pre_trained, a path to the weights to be loaded
		#	must be provided.
		#	path: String, path to the file of weights to be loaded """

		if class_._instance == None:
			class_._instance = class_.__create__( )
		if pre_trained:
			class_.__pre_train__( path )
		return class_._instance

	@classmethod
	def __create__( class_ ):
		"""	Net Creation in case it is not already created. """
		net = torch.nn.Sequential(
		torch.nn.Linear(3, 50),
		torch.nn.Linear(50, 50),
		torch.nn.LeakyReLU(),
		torch.nn.Linear(50, 1))
		return net

	@classmethod
	def __pre_train__( class_, path ):
		"""	If required, a pre-trained instance of the net is returned
		#	in order to be able to make body mass estimation right away. """
		if os.path.exists( path ):
			if not torch.cuda.is_available():
				class_._instance.load_state_dict( torch.load( path, map_location=torch.device('cpu') ))
			else:
				class_._instance.load_state_dict( torch.load( path ))
			class_._instance.eval()
		else:
			print( " Error!!! Path doesn't exist!\n a dummy net will be returned, please verify the path and load the weights again")
		return