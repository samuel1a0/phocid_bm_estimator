# from utiles import *
# from pointcloud_morphology import *
import numpy as np
import trimesh as tr
# import icp
import os
import string, random
import subprocess as spc

## utiles

# class Utiles(Object):
""" Operaciones que no tienen que ver con la limpieza en sí, pero facilitan ciertos procedimientos """

FLOOR_VECTOR = [0, 1, 0]

class Cleaner:
	def __init__(self):
		self.__previous__ = None

	def restore( self ):
		if self.__previous__ != None:
			return self.__previous__.copy()
		else:
			print( " No model previously saved! " )
		return

	def __save_prev__( self, mesh ):
		del self.__previous__
		self.__previous__ = mesh.copy()
		return

	def delete_plane(self, mesh, plane_o, plane_n, tol=.05):
		values = tr.points.point_plane_distance( mesh.vertices, plane_origin=plane_o, plane_normal=plane_n)
		mask = values < tol

		normals = [plane_n] * len(mesh.vertex_normals)
		n_mask = np.isclose( mesh.vertex_normals, normals, atol=.4)
		n_mask = [ np.alltrue( x ) for x in n_mask ]

# 		f_mask = [x and y for x, y in zip( mask, n_mask ) ]
# 		to_keep = [not x for x in f_mask ]
		to_keep = [not(x and y) for x, y in zip( mask, n_mask ) ]
		n_mesh = self.delete_vertices( mesh, mask=to_keep, color=False )
		return n_mesh

    
	def get_points_in_plane(self, mesh, plane_o, plane_n, tol=.05):
		values = tr.points.point_plane_distance( mesh.vertices, plane_origin=plane_o, plane_normal=plane_n)
		mask = values < tol

		normals = [plane_n] * len(mesh.vertex_normals)
		n_mask = np.isclose( mesh.vertex_normals, normals, atol=.4)
		n_mask = [ np.alltrue( x )and(y) for x,y in zip( n_mask, mask ) ]
		return n_mask
    


	def find_plane(self, mesh, offset=0.1):
		"""	Obtiene el plano que más se asemeja a representar el piso, y retorna la matriz transformación que posicionaría al mesh de
			manera que el 'piso' quede coplanar a los ejes 'XY'. Tambien devuelve el plano (piso) como punto origen y normal
			return:
					trf: matriz transformación.
					p_origen : Punto Origen del plano
					p_normal : Normal del plano """
		c, p_o, p_n = 0, 0, 0
		box = mesh.bounding_box_oriented
		extent = min(box.extents)*offset
		mean_normal = np.mean( mesh.vertex_normals, axis=0 )
# 		for i in range( len( box.faces ) ):
# 			o = box.vertices[ box.faces[ i ] ]
# 			o = np.mean(o, axis=0)
# 			n = -box.face_normals[ i ]
# 			if np.arccos(np.clip(np.dot(n, mean_normal), -1.0, 1.0))>(np.pi/2):   # Si no anda, sacar esto!
		for i in range( len( box.facets ) ):
			idx = np.unique(box.faces[box.facets[i]])
			o = box.vertices[ idx ]
			o = np.mean(o, axis=0)
			n = -box.facets_normal[ i ]
			if np.arccos(np.clip(np.dot(n, mean_normal), -1.0, 1.0))>(np.pi/2):   # Si no anda, sacar esto!
				continue
			o = o + n*extent
	#         o = o + n*offset
			sec = mesh.section( plane_origin=o, plane_normal=n)
			if sec !=None and c< sec.vertices.shape[0]:
				c = sec.vertices.shape[0]
				p_o, p_n = o, n
		if type(p_n) == "int":
	#         return self._align_plane()
			return self.find_plane(mesh, offset=offset*1.1)
		trf = tr.geometry.plane_transform(origin=p_o, normal=p_n)
		return trf, p_o, p_n  


	def rezero(self, mesh):
		mesh.vertices = mesh.vertices-np.min(mesh.vertices, axis=0)
		return mesh


	def __align_plane__( self, mesh, offset=.05 ):
		""" Intenta determinar cual es el lado correspondiente al 'piso' en la figura, y colocarlo coplanar a los ejes 'XY'
				'offset': distancia del borde a partir de la cual se toman muestras para
				determinar que lado es candidato a 'piso'. """
		trf, o_piso, n_piso = self.find_plane( mesh, offset )
		trf = tr.geometry.align_vectors( n_piso, FLOOR_VECTOR )
		n_mesh = mesh.copy()
		n_mesh.apply_transform( trf )
		n_mesh = self.rezero( n_mesh )
		return n_mesh

	def align_plane( self, mesh, offset=.05 ):
		""" Intenta determinar cual es el lado correspondiente al 'piso' en la figura, y colocarlo coplanar a los ejes 'XY'
				'offset': distancia del borde a partir de la cual se toman muestras para
				determinar que lado es candidato a 'piso'. """
        
		mesh = self.__align_plane__(mesh, offset)
		mesh = self.__align_plane__(mesh, offset)
		return mesh

	def delete_floor_2( self, mesh, offset=.1, tol=.05 ):
		""" Intenta determinar y eliminar la sección correspondiente al piso en el mesh.
				'offset': distancia del borde a partir de la cual se toman muestras para
				determinar que lado es candidato a 'piso'.
				'tol': distancia desde el plano designado como 'piso'
				 a la cual los elementos siguen considerandose como pertenecientes al mismo """
		initial_value = 0 # mesh.vertices.shape[0]
		last = np.inf
		# self.__save_prev__( mesh )
		while initial_value != last:
			initial_value = mesh.vertices.shape[0]
			# trf, o_piso, n_piso = self.find_plane( mesh, offset )
			mesh = self.align_plane( mesh, offset )
			_, o_piso, n_piso = self.find_plane( mesh, offset )
			mesh = self.delete_plane( mesh, o_piso, n_piso, tol)
			last = mesh.vertices.shape[0]
			tol = tol*.95
		return mesh

	def delete_floor( self, mesh, offset=.1, tol=.05 ):
		""" Intenta determinar y eliminar la sección correspondiente al piso en el mesh.
				'offset': distancia del borde a partir de la cual se toman muestras para
				determinar que lado es candidato a 'piso'.
				'tol': distancia desde el plano designado como 'piso'
				 a la cual los elementos siguen considerandose como pertenecientes al mismo """
		initial_value = 0 # mesh.vertices.shape[0]
		last = np.inf
		# self.__save_prev__( mesh )
		while initial_value != last:
			initial_value = mesh.vertices.shape[0]
			trf, o_piso, n_piso = self.find_plane( mesh, offset )
			mesh = self.delete_plane( mesh, o_piso, n_piso, tol)
			last = mesh.vertices.shape[0]
			tol = tol*.95
		return mesh


	def delete_vertices( self, mesh, mask, isolate=True, color=True ):
		""" Elimina del mesh los vértices indicados según las mascara 'mask' """
		# n_mesh = mesh.copy()
		idx = np.where( mask )
		new_coords = mesh.vertices[ idx[0] ]
		if color:
			if isinstance(mesh.visual, tr.visual.color.ColorVisuals):
				new_cols = mesh.visual.vertex_colors[ idx[0] ].copy()
			else:
				colors = mesh.visual.to_color()
				new_cols = colors.vertex_colors[ idx[0] ].copy()


		old2new = -( np.ones( len( mesh.vertices ) ) )
		old2new[ idx[0] ] = np.arange( len( idx[0] ) )

		new_faces = old2new[ np.asarray( mesh.faces ) ]
		new_faces = new_faces[ ( new_faces != -1 ).all( axis = 1 ) ]

		# n_mesh.vertices = new_coords
		# n_mesh.faces = new_faces
		# n_mesh.visual.vertex_colors = new_cols
		n_mesh = tr.Trimesh( vertices=new_coords, faces=new_faces )
		if color:
			n_mesh.visual.vertex_colors = new_cols

		if isolate:
			n_mesh = self.isolate_mesh( n_mesh, .5)
		return n_mesh


	# Limpieza, Alineación y Smooth
	def isolate_mesh( self, mesh, tol=.5 ):
		""" Separa el mesh en componentes conectados, cuidando que la cantidad
			de vertices restantes sea superior al indice de tolerancia 'tol' """
		# self.__save_prev__( mesh )
		# print(mesh.vertices.shape)
		trigger = (mesh.vertices.shape[0])*tol
		array_mesh = mesh.split( only_watertight=False )
		array_mesh = sorted(array_mesh, key=lambda x:x.vertices.shape[0], reverse=True)
		# n_mesh = tr.boolean.union(array_mesh[:n])
		# print(mesh.vertices, array_mesh)
		actual = array_mesh[0].vertices.shape[0]
		i = 0
		while actual < trigger:
			i+=1
			actual += array_mesh[i].vertices.shape[0]
		off = min(i+1, len(array_mesh))
# 		n_mesh = tr.boolean.union(array_mesh[:off])
		n_mesh = np.sum( array_mesh[:off] )

		return n_mesh


	def align_mesh( self, modelo ):
		""" Intenta alinear el 'mesh' con la figura 'modelo' """
		mesh_box = self.mesh.bounding_box_oriented
		# model_box = modelo.vertices
		model_box = modelo.bounding_box_oriented
		trf = tr.points.absolute_orientation( mesh_box.vertices, model_box)
		n_mesh = self.mesh.copy()
		n_mesh.apply_transform(trf)
		# self.__save_prev__( n_mesh )
		return self.mesh


	def smooth( self, iterations=3 ):
		""" Aplica un Suavizado Laplaciano la cantidad de veces indicada por 'iterations' """
		n_mesh = self.mesh.copy()
		for i in range( iterations ):
			g = n_mesh.vertex_adjacency_graph
			adjacents = []
			nb = [ n_mesh.vertices[ g.neighbors( i ) ]  for i in range( n_mesh.vertices.shape[ 0 ] ) ]
			for n in nb:
				tmp = np.sum( n, axis=0 ) / len( n )
				adjacents.append( tmp )
			n_mesh.vertices = np.array( adjacents )
		# self.__save_prev__( n_mesh )
		return self.mesh

## Reconstrucción y Cropeado

	def reconstruccion( self, mesh, path="./", depth=7):
		self.__save_prev__( mesh )
		name = random_generator( 15 ) + ".ply"
		box = mesh.bounding_box_oriented
		store_temp( mesh, path )
		output_f = path+"/"+name
		points = path+"/p.txt"
		normals = path+"/n.txt"
# !!!		! python2 utiles/reconstruccion.py { output_f } { points } { normals } { depth }
		spc.call( "python2 resources/reconstruccion.py {} {} {} {}".format( output_f, points, normals, depth), shell=True )
		n_mesh = tr.load(output_f)
# !!!		! rm { points }    
# !!!		! rm { normals }
# !!!		! rm { output_f }
		spc.call( "rm {} {} {}".format( points, normals, output_f ) )
		return self.mesh

	def cuasi_crop( self, mbox):
		box = mbox.vertices
		n_mesh = self.mesh.copy()
		x, y, z = n_mesh.vertices[:, 0], n_mesh.vertices[:, 1], n_mesh.vertices[:, 2] 
		x[ x < np.min( box[:, 0] ) ] = np.min( box[:, 0] )
		x[ x > np.max( box[:, 0] ) ] = np.max( box[:, 0] )
		y[ y < np.min( box[:, 1] ) ] = np.min( box[:, 1] )
		y[ y > np.max( box[:, 1] ) ] = np.max( box[:, 1] )
		z[ z < np.min( box[:, 2] ) ] = np.min( box[:, 2] )
		z[ z > np.max( box[:, 2] ) ] = np.max( box[:, 2] )    
		n_mesh.vertices[:, 0] = x
		n_mesh.vertices[:, 1] = y
		n_mesh.vertices[:, 2] = z
		
		self.__save_prev__( n_mesh )
		return self.mesh

	def plane_cut( self, mesh, plane, normal, reduction_steps=3, edge_size=.05, offset=.05 ):
		offset *= np.min(mesh.extents)
		v,f = tr.remesh.subdivide_to_size( plane.vertices, plane.faces, edge_size )
		plane = tr.Trimesh(vertices=v, faces=f)
		plane.vertices +=plane.vertex_normals*offset

		r = tr.ray.ray_triangle.RayMeshIntersector( mesh )
		directions = np.array([-normal]*len(plane.vertices))
		mask = r.intersects_any( plane.vertices, directions)

		plane = self.delete_vertices(plane, mask, isolate=False, color=False)

		for i in range(reduction_steps):
			edges = tr.grouping.group_rows(plane.edges_sorted, require_count=1)
			edges = plane.edges_sorted[edges]
			edges = np.unique(edges.flatten())
			mask = np.ones( plane.vertices.shape[0], dtype=np.bool )
			mask[edges] = False
			plane = self.delete_vertices( plane, mask, isolate=False, color=False )
		return plane

	def add_plane( self, mesh, reduction_steps=3, edge_size=.05, template=None, offset=0):
		if template == None:
			trf, o, n = self.find_plane(mesh)
			#         edge = mesh.outline()
			plane = tr.Trimesh( vertices=mesh.bounding_box_oriented.vertices,
                                faces=mesh.bounding_box_oriented.faces,
                                vertex_colors=mesh.bounding_box_oriented.visual.vertex_colors)
			cfaces = []
			for i, fn in enumerate(plane.face_normals ):
				if ( np.allclose(-n, fn, .2) ):
					cfaces.append(i)
			indices = np.unique( plane.faces[cfaces].reshape(-1) )
			mask = np.zeros( plane.vertices.shape[0], dtype=np.bool )
			mask[indices] = True
			plane = self.delete_vertices( plane, mask, isolate=False )
		else:
			plane = template
		plane = self.plane_cut( mesh, plane, -n, reduction_steps, edge_size, offset=offset )
		return mesh+plane
    
    
	# def store_temp(self, mesh, path):
	# 	# print(path)
	# 	with open("{}/p.txt".format(path), 'w') as f:
	# 		for item in mesh.vertices:
	# 			f.write("{} {} {}\n".format(item[0],item[1], item[2]))

	# 	vertex_normals, _ = calculate_normals(mesh)
	# 	with open("{}/n.txt".format(path), 'w') as f:
	# 		for item in vertex_normals:
	# 			f.write("{} {} {}\n".format(item[0],item[1], item[2]))
	# 	return