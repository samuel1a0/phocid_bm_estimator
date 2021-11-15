import numpy as np
import trimesh as tr
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d, Axes3D
from utils.utiles import *
from net.segmentation_net import *
import torch

WEIGHT_PATHS = "weights/model_radial_e3000.pt"


class Segmenter( object ):
	def __init__( self, weights_path, points=1024, device="cpu", r_ch=64):
		self.n_points = points
		self.device = device
		self.model = self.load_net( weights_path, n_points=self.n_points, device=self.device, r_ch=64 )
		return

	def load_net( self, weights_path, n_points=1024, device="cpu", r_ch=64 ):
		if not weights_path:
			weights_path = WEIGHT_PATHS
		model = Seg_Asb_net(n_points=n_points, do=.2, device=device, r_ch=r_ch).to(device)
		if not torch.cuda.is_available():
			model.load_state_dict(torch.load( weights_path, map_location=torch.device("cpu") ))
		else:
			model.load_state_dict(torch.load( weights_path ))
		model.eval()
		return model

	def segment( self, mesh ):
		points, normals, fidx = points_from_model( mesh, self.n_points, return_normals=True )
		points = torch.from_numpy( points )
		points = points.view( 1, -1, 3)
		output = self.model( points.to( self.device ) )
		_, preds = torch.max( output, 2)
		print("Segmentation results: {}/{} points labeled as valid.\n".format(
										np.sum( preds.detach().numpy() ),
										self.n_points ) )

		preds = np.array(preds[:]==0, dtype=np.bool)
		# print(preds)
		valid_fidx = fidx[ preds[0] ]
		# return valid_fidx
		return points.detach().numpy().reshape(-1, 3), normals, preds[0]

	def clean_mesh( self, mesh, valid_fidx):
		mesh = delete_vertex_from_faces( mesh, valid_fidx )
		mesh.show()
		# mesh = isolate_reunite ( mesh, .1 )
		return mesh

	def process( self, mesh, scale=1 ):
		final_points, final_normals = [], []
		for i in range(scale):
			points, normals, preds = self.segment( mesh )
			final_points.extend( points[preds] )
			final_normals.extend( normals[preds] )

		n_mesh = tr.Trimesh(vertices=final_points, vertex_normals=final_normals)

		# mesh.show()
		# mesh = self.clean_mesh( mesh, fidx)
		return n_mesh
