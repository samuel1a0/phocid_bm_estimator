import trimesh as tr
import numpy as np

class Mantle(object):
	""" Mantle. Algorithm that tryes to fit a surface simulating the behavior
	#	of an actual mantle trown over an object. The deformation process follows
	#	an ARAP (As Rigid As Posible) aproximation taking in account the constrains
	#	given for the user.
	"""

	def __init__(self, subdivisions=5, elasticity=.5, step=.1, degree=1, precompute_n=False, contact_color=[1,0,0,1]):
		""" Initialization step. setting up all the configuration for the 'mantle' 
		#	object to work properly.
		#	params:
		#		subdivisions: 	Int. Number of vertices in the Mantle (2^subdivisions).
		#		elasticity : 	Float (0~1). Restrain to be applyed to neighbors of already
		#						fixed vertices in the mantle.
		#		step: 			Float. Deformation step value in each update
		#		degree : 		Int. Degrees of separation for neighbor computation
		#		precompute_n: 	Bool. wether to precompute the neighborhood or compute it 
		#						at demand.
		#		contact_color:	Array. if provided, must be a 1x4 array of float (0~1). Used
		#						in order to easely detect vertices wich made contact with
		#						the surface (to be ).
		"""
		self.elasticity = elasticity
		self.degree = degree
		self.color = np.array(contact_color)*255

		self.subdivisions = subdivisions
		self.manta = None
		self.ceil = []
		self.freedom = []
		self.distance = []
		self.neighbors = {}

		self.step = step
		self.punishment = (1-elasticity)*self.step #*self.subdivisions

		self.direction = [0, 0, 0]
		self.displacement = []
		self.template = None


	def compute_neighbors( self ):
		""" Computes a look-up table for easily find neighboring
		#	vertex within the mantle.
		#	return:
		#		dict_n:		Dict. Dictionary containing for each
		#					vertex (key), an array of all the 
		#					neighboring vertices (value).
		"""
		dict_n = {}
		G = self.manta.vertex_adjacency_graph
		for idx in range(self.manta.vertices.shape[0]):
			nbgs = np.array(list(tr.graph.nx.single_target_shortest_path_length(G, idx, self.degree)))[1:,0]
			dict_n[idx] = nbgs
		return dict_n

	def get_neighors( self, idx ):
		""" Given a vertex index, retrieves all the neighboring vertex
		# 	to be updated for the modifications of the former one.
		#	parameters:
		#		idx: 	Int. Index of the initial vertex.
		#	return:
		#		nbgs:	Array. Array containing the indices of all the
		#				vertices within the distance degree of the initial.
		"""
		nbgs = []
		try:
			nbgs = self.neighbors[idx]
		except:
			G = self.manta.vertex_adjacency_graph
			nbgs = np.array(list(tr.graph.nx.single_target_shortest_path_length(G, idx, self.degree)))[1:,0]
			self.neighbors[idx] = nbgs
		return nbgs

	@property
	def n_points( self ):
		""" return the amount of vertices in the 'mantle' object. """
		return self.manta.vertices.shape[0]

	def generate_mantle( self, subdivisions, mesh, template=None ):
		""" Generate a plane-like object according to the specified
		#	parameters. If a template is provided, the generated
		#	object will follow the same orientation in all three axis
		#	parameters:
		#		subdivisions:
		#		mesh:			Mesh. Object to be analized in order
		#						to fit the mantle.
		#		template:		Mesh. If provided, object to be
		#						subdivided in order to met the required
		#						distance between vertices.
		#	return:
		#		manta:			Mesh. Plane-like mesh (or template-like
		#						if provided).
		"""
		divisions = 2**subdivisions
		if template==None:
			max_lengt = np.mean(mesh.extents)/divisions
			p = [ [0,0,0], [1,0,0], [0,0,1], [1,0,1] ]
			f = [ [ 2,1,0], [2,3,1]]
			p, f = tr.remesh.subdivide_to_size(p, f, max_edge=max_lengt, max_iter=20)
		else:
			max_lengt = np.mean(mesh.extents)/divisions

			p, f = tr.remesh.subdivide_to_size(template.vertices, template.faces, max_edge=max_lengt, max_iter=20)
		manta = tr.Trimesh(vertices=p, faces=f)
		return manta


	def set_mesh( self, mesh ):
		""" Initialization of all the mantle's properties to
		#	be used for the fitting process.
		#	parameters:
		#		mesh:	Mesh. Object to be analyzed.
		"""
		mesh.rezero()
		template, origin, normal = self.get_plane( mesh, return_plane=True)
		self.manta = self.generate_mantle( self.subdivisions, mesh, template)
		self.neighbors = self.compute_neighbors( )

		self.ceil = np.zeros_like( self.manta.vertices )
		self.freedom = np.ones( self.n_points )
		self.distance = np.full(self.manta.vertices.shape[0], np.inf)
		self.displacement = np.zeros_like( self.manta.vertices )

		v = tr.util.unitize(self.manta.facets_normal[0])
		v1 = v / np.linalg.norm( v )
		v2 = np.mean(mesh.face_normals, axis=0) / np.linalg.norm( np.mean(mesh.face_normals, axis=0) )

		angle = np.arccos( np.clip( np.dot(v1, v2), -1.0, 1.0) )
		if np.abs(angle) < np.pi:
			self.direction = -v
			self.manta.fix_normals()
		else:
			self.direction = v
			self.manta.fix_normals()

		self.set_distance( mesh )
		self.displacement[:] = self.step #*self.displacement
		return
    
    
	def get_plane( self, mesh, return_plane=False ):
		""" Given a mesh, takes it's oriented bounding box and
		#	retrieves the plane with the higest chance to be the
		#	closest to the floor of the mesh scanned.
		#	parameters:
		#		mesh:			Mesh. Object to be analyzed.
		#		return_plane:	Bool. Wheter to return or not
		#						a mesh of the plane.
		#	return:
		#		box:			Mesh (optional). Mesh of the
		#						best suited plane found.
		#		origin, normal:	Float. Plane defined as Origin
		#						and Normal mathematical notation.
		"""
		box = tr.Trimesh(vertices=mesh.bounding_box_oriented.vertices,
                         faces=mesh.bounding_box_oriented.faces)
		generic_normal = mesh.face_normals.mean(axis=0)
		distance = box.facets_normal - generic_normal
		idx = np.argmax( np.linalg.norm(distance, axis=1) )
		f = np.unique(box.faces[box.facets[idx]].flatten())
		origin = np.mean( box.vertices[ f ], axis=0)
		normal = np.array(-box.facets_normal[idx])
		if return_plane:
			vs = np.array([0]*box.vertices.shape[0], dtype=np.bool)
			vs[f] = True
			box.update_vertices(vs)
			v = box.faces.copy()
			box.faces[:,[1,2]] = box.faces[:,[0,1]]
			box.faces[:,0] = v[:,2]

			return box, origin, normal
    
		return origin, normal



	def set_distance( self, mesh ):
		""" Computes the distance from the mantle's vertices to the
		#	mesh. This distance is used to prevent the mantle to
		#	be deformed more than necesary.
		"""
		self.ceil[:] = np.inf
		ray = tr.ray.ray_triangle.RayMeshIntersector( mesh )
		normals = np.array([self.direction]*self.n_points )
		coord, idx, _ = ray.intersects_location( self.manta.vertices, normals )

		if len(coord)==0:
			self.direction = -self.direction
			normals = np.array([self.direction]*self.n_points )
			coord, idx, _ = ray.intersects_location( self.manta.vertices, normals )
		self.ceil[ idx ] = coord
		self.distance = np.linalg.norm( self.ceil-self.manta.vertices, axis=1 )
		self.manta.fix_normals()
		return

	def fit( self, max_iterations=100, log=True):
		""" Given a number of iterations, triggers the mantle's
		#	deformation step till 'max_iterations' is reached, or
		#	the mantle reachs an stationary state ( no modifications
		#	are detected in the mantle ).
		"""
		i = 0
		while (not self.check()) and (i<max_iterations):
			self.update()
			if log:
				print("step {}...".format(i+1))
			i +=1
		self.manta.fix_normals()

		print(" Finished... ")
		return

	def update( self ):
		""" Modifies the values of each vertex according to the
		#	actual state
		"""
		actual_displacement = (self.displacement * self.freedom[:, np.newaxis])
		actual_displacement = actual_displacement * self.direction
		self.manta.vertices += actual_displacement
		self.distance -= np.linalg.norm(actual_displacement, axis=1)
		return

    
	def check( self ):
		""" Verifyes the condition of every vertex in the mantle
		#	(distance covered and elasticity restrictions according
		#	neighboring vertex states ) and applys restrictions
		#	as needed
		"""
		idx = np.where(self.distance<=0)[0]
		idx_2 = np.where(self.freedom<=0)[0]
		idx = np.unique( np.concatenate( [idx, idx_2] ) )
		if len(idx) >0:
			nbgs = []
			nbgs = np.array([ y for x in idx.flatten() for y in self.get_neighors( x )])

			nbgs, counts = np.unique(nbgs.flatten(), return_counts=True)
			punish = np.array([self.punishment]*len(nbgs))*counts
			self.freedom[nbgs] -= punish

		self.freedom = np.clip(self.freedom, 0, np.inf)
		return ( np.alltrue( self.freedom == 0) )