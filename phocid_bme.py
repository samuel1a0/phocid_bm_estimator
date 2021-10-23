import argparse
import open3d as o3d
import trimesh as tr
import numpy as np
import os
import itertools
import pandas as pd
from torch.autograd import Variable
from torch import tensor
from utils.cleaner_utils import *
from utils.utiles import *
from utils.segmentation import *
from utils.mantle import *
from net.net import Seq_Net
from net.segmentation_net import *


""" Constants: Values to be used in case the user don't specified them """
DEFAULT_VALUES = {
	"in_path" : "example_data/",
	"rec_path" : "example_data/reconstructions",
	"pre_trained": 1,
	"weights_path" : "net/weights/model_radial_e1400.pt",
	# "weights_path" : "net/weights/default.pt",
	"degree" : 2,
	"precompute_n" : 1,
	"step" : .05,
	"elasticity" : .95,
	"subdivisions" : 6,
	"max_iterations":500,
	}

MAX_BODY_COUNT = 500



def parse_args():
	"""Parameters configuration. Initialization step for all the process """
	parser = argparse.ArgumentParser()
	parser.add_argument( "--in_path", help="Folder containing the meshes to be processed", type=str )
	parser.add_argument( "--rec_path", help="Folder where the reconstructions will be stored", type=str )
	parser.add_argument( "--pre_trained", help="Wether to use a trained(1) or a dummy(0) net to estimate Body Mass", type=int )
	parser.add_argument( "--weights_path", help="Path to the file of weights to be loaded in the model", type=str )
	parser.add_argument( "--degree", help="Mantle Configuration: degrees of separation for neighbor computation", type=int )
	parser.add_argument( "--precompute_n", help="Mantle Configuration: wether to precompute the neighborhood or compute it at demand", type=int )
	parser.add_argument( "--step", help="Mantle Configuration: deformation step value in each update", type=float )
	parser.add_argument( "--elasticity", help="Mantle Configuration: Restrain to be applyed to neighbors of already fixed vertices in the mantle (range 0-1)", type=float )
	parser.add_argument( "--subdivisions", help="Mantle Configuration: Number of vertices in the Mantle (2^subdivisions)", type=float )
	parser.add_argument( "--max_iterations", help="Mantle Configuration: Max number of iterations to the floor of a mesh", type=float )
	args = parser.parse_args( )
	
	# Isolating and loading default values as needed.
	non_existing = []
	for k in args.__dict__:
		if args.__dict__[k] == None:
			args.__dict__[k] = DEFAULT_VALUES[ k ]
			non_existing.append( k )
	if len(non_existing) > 0:
		print("Some parameters weren't provided.\n Using default values for:{}".format( non_existing ))
	return args



def preprocess(mesh):
	""" Basic mesh processing, preparation step.
	#	This funtion just removes isolated/duplicated
	#	vertices, makes face normals wind-consistent,
	#	ensures all vertices' values are above zero in
	#	all the axes and fixs basics holes (just if it
	#	can be fixed adding a single face to the mesh).
	#	parameters:
	#		mesh:	Mesh. Object to be processed.
	#	return:
	#		mesh:	Mesh. Processed mesh.
	"""
	mesh.process()
	mesh.rezero()
	tr.repair.fix_winding(mesh)
	return mesh


def floor_extraction( mesh, args ):
	""" Creates a mantle-like pattern to estimate and fit
	#	existing floor, and creates a new one.
	#	parameters:
	#		mesh:	Mesh. Object to be processed.
	#		args:	NameSpace. Object containing the 
	#				configuration values degree, precompute_n,
	#				step, elasticity and subdivisions. For
	#				more information about these parameters 
	#				please refere to the 'parse_args' function.
	#	return:
	#		mesh:	Mesh. A raw cleaner version of the mesh received.
	#				(mesh without the scaned floor).
	"""
	c = Cleaner()
	mesh = isolate_reunite( mesh, trigger=.2 )
	d = mesh.scale/100
	m = Mantle( degree=args.degree, precompute_n=args.precompute_n, step=args.step,
				elasticity=args.elasticity, subdivisions=args.subdivisions )
	m.set_mesh( mesh )
	m.fit( max_iterations=700, log=False )
	# (mesh+m.manta).export( "{}/pre-{}.ply".format( args.rec_path, np.random.randint(1000) ))  ## quitar

	mesh = remove_floor( mesh, m.manta, distance=d, isol_tol=.2 )

	mesh = c.add_plane( mesh, reduction_steps=1, edge_size=.04, offset=.1 )#edge_size=.04 )
	return mesh


# def mesh_reconstruction( mesh, args ):
def mesh_reconstruction( points, normals, args ):
	"""	Reconstruction using Poisson Surface Reconstruction algorithm
	#	implemented in the Open3d library. Args expected for this  
	#	method belongs to the Poisson Reconstruction.
	#	depth: int. Maximum depth of the tree that will be used for
	#		surface reconstruction.
	#	width: int. Specifies the target width of the finest level octree
	#		cells (ignored if the --depth is also specified).
	#	scale: float. Specifies the ratio between the diameter of the
	#		cube used for reconstruction and the diameter of the
	#		samples' bounding cube
	#
	#	return:
	#		mesh:	Mesh. Reconstructed mesh.
	#
	#	For more information about the parameters please refer to the 
	#	Open3d documentation
	#	http://www.open3d.org/html/tutorial/Advanced/surface_reconstruction.html#Poisson-surface-reconstruction
	"""

	pcd = o3d.geometry.PointCloud()
	# pcd.points = o3d.utility.Vector3dVector( np.array(mesh.vertices) )
	# pcd.normals = o3d.utility.Vector3dVector( np.array(mesh.vertex_normals) )
	pcd.points = o3d.utility.Vector3dVector( points )
	pcd.normals = o3d.utility.Vector3dVector( normals )
	pmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6, width=0,
	                                                                  scale=1.2, linear_fit=False)[0]
	mesh = tr.Trimesh(vertices=np.array(pmesh.vertices), faces=np.array(pmesh.triangles))
	return mesh


def procces( args, files ):

	""" Processing Pipeline. Raw processing of the meshes provided.
	# 	expected parameters:
	#	args: Array. Composed by all the parameters
	#		requested in the Initialization Step.
	#		(see 'parse_params' function for a list
	#		and information of the options required).
	#	files: array. List of files to be processed
	 """


	# initializing temporal variablies 
	# net = Seq_Net( args.pre_trained, args.weights_path )
	net = Seq_Net( args.pre_trained, "net/weights/default.pt" )
	# seg = Segmenter( args.weights_path, args.n_points, args.device, args.r_ch)
	seg = Segmenter( "net/weights/model_radial_e1400.pt", 1024, "cpu", 64)
	volumenes = []
	cfs = []
	axis = []
	bm_count = []
	box_side = 10

	# Creating a box to cut extremely big reconstructions
	# The size of each side can be controlled by de 
	# "box_side" variable (default=10, size in meters ) 
	box = tr.creation.box(extents=[box_side, box_side, box_side])
	box.apply_translation([0, 0 , -np.min(box.vertices)])
	offset = 0
	for i, (f) in enumerate( files ):
		name = f.split(".")[0]
		v = 0
		cf = 0
		ax = 0
		bm = 0

		if os.path.exists("{}/{}".format( args.in_path, f )):
			mesh = tr.load( "{}/{}".format( args.in_path, f ) )

			# The higer the body counts is, the slowest to process it. 
			# This trigger can be rised if high computing capacity
			# is available.
			if mesh.body_count <MAX_BODY_COUNT:
				mesh = preprocess(mesh)

				# mesh = seg.process( mesh )
				# mesh.export( "{}/segmentation_{}".format( args.rec_path, f ))
				# mesh = mesh_reconstruction( mesh, args )
				points, normals, valid_idx = seg.segment( mesh )
				mesh = mesh_reconstruction( points[valid_idx], normals[valid_idx], args )
				# mesh.fix_normals()

				mesh.export( "{}/{}".format( args.rec_path, f ))
				mm = mesh.copy()

				# looking for watertight objects suceptible to be analized
				mesh = mesh.split(only_watertight=True)
				mesh = sorted(mesh, key=lambda x:len(x.vertices), reverse=True)
				if len(mesh)>0:
					mesh = mesh[0]
					mesh.export( "{}/{}".format( args.rec_path, f ))
					v = mesh.volume
					cf, slices = circ_mayor( mesh, just_value=False )
					if slices:
						skeleton = get_cuasi_medial_axis( slices )
						ax = get_total_length( skeleton )
						bm = net( Variable( tensor( [[v, ax, cf]] ) ).float() ).detach().numpy().flatten()[0]
				else:
					print("Object {} couldn't be processed (reconstruction is not watertight).".format( f ))
					mm.export( "{}/ERROR_{}-{}".format( args.rec_path, "pre", f) )
	    
		axis.append( ax )
		cfs.append( cf )
		volumenes.append( v )
		bm_count.append( bm )

		print("Object '{}' results: EV:{:.2f} \t EMG: {:.2f}\t ESL:{:.2f}, Body Mass:{:.2f}".format(name, v, ax, cf, bm))
	return




def main( args ):
	""" This method makes general validations and begins the
	#	the process of the meshes found. """

	files = os.listdir( args.in_path )
	files = [x for x in files if ".obj" in x or ".ply" in x]

	# Checking existing files to process
	if len( files )>0:
		print("Number of meshes found: {}".format( len(files) ))
		procces( args, files )

	else:
		print( "Data folder doesn't contain any '.ply' nor '.obj' files to process...")
	return 0



# ESTE COMENTARIO SACALO QUE ES SOLO PARA VOS. LO DE ABAJO
# ES COMO SE HACE PARA QUE PYTHON SEPA CUAL ES EL METHODO
# QUE TIENE QUE EJECUTAR CUANDO EMPIEZA EL PROGRAMA
if __name__ == "__main__":
	args = parse_args(  )
	main( args )
