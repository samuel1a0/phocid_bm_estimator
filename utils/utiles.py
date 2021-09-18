import numpy as np
import os
import trimesh as tr

def remove_floor( mesh, floor, distance=.02, isol_tol=.1 ):
    """ Remove the vertices from 'mesh' closer than 'distance' to
    #   the floor.
    #   params:
    #       mesh:       Trimesh. Model to be processed.
    #       floor:      Trimesh. Model representing the floor
    #                   of the model.
    #       distance:   Float. Maximum distance of a vertex to
    #                   the floor to consider such vertex as part 
    #                   of the real floor.
    #       isol_tol:   Float. Tolerance value to be used in the
    #                   Isolation process.
     """
    pq = tr.proximity.ProximityQuery( floor )
    a = mesh.copy()

    d,idx = pq.vertex( a.vertices )
    #distance = mesh.vertices.shape[0]/floor.vertices.shape[0]

    mask = d>distance
    a = delete_vertices(a, mask=mask)
    b = isolate_reunite( a, trigger=isol_tol )
    return b

def delete_vertices( mesh, mask, color=True ):
    """ Delete the vertices from the mesh according to the indicated mask.
    #   params:
    #       mesh:       Trimesh. Model to be processed.
    #       mask:       Array (bool). Array indicating for each vertex if
    #                   it should remain or be removed from the mesh.
    #   return:
    #       n_mesh:     Trimesh. Model with only the vertices specified
    #                   in mask.
    #
    #   IMPORTANT: THIS PARTICULAR ALGORITHM (delete_vertices) WAS TAKEN DIRECTLY
    #   FROM THE TRIMESH LIBRARY.
    """
    idx = np.where( mask )
    new_coords = mesh.vertices[ idx[0] ]

    old2new = -( np.ones( len( mesh.vertices ) ) )
    old2new[ idx[0] ] = np.arange( len( idx[0] ) )

    new_faces = old2new[ np.asarray( mesh.faces ) ]
    new_faces = new_faces[ ( new_faces != -1 ).all( axis = 1 ) ]

    n_mesh = tr.Trimesh( vertices=new_coords, faces=new_faces )
    return n_mesh

def isolate( mesh, trigger=.05, stop=50 ):
    """ Filter the mesh to delete small isolated areas (bodies)
    #   params:
    #       mesh:       Trimesh. Model to be processed.
    #       trigger:    Float (0~1). How much of the total
    #                   count of faces should have a body to
    #                   not be excluded.
    #       stop:       Int. Maximum amount of bodies acceptableto
    #                   process the model (the more amount ofbodies
    #                   in the mesh, the slower the process become)
    #   return:
    #       mesh:       Trimesh. Filtered model containing all the
    #                   bodies that fulfil the criteria. """

    if mesh.body_count>stop:
        print("too much bodyes!! skipping isolation...")
        return mesh
    trigger = np.floor( trigger * mesh.vertices.shape[0])
    mesh_v = mesh.split( only_watertight=False )
    arreglo = []
    for m in mesh_v:
        if m.vertices.shape[0] > trigger:
            arreglo.append(m)
    if len(arreglo)>0:
        return (np.sum( arreglo ))
    return mesh

def isolate_reunite( mesh, trigger, min_len=5 ):
    """ Filter the mesh to delete isolated areas (bodies) smaller
    #   than specified values.
    #   params:
    #       mesh:       Trimesh. Model to be processed.
    #       trigger:    Float (0~1). How much of the total
    #                   count of faces should have a body to
    #                   not be excluded.
    #       min_len:    Int. minimum amount of conected components
    #                   that should have a body to not be excluded.
    #   return:
    #       mesh:       Trimesh. Filtered model containing all the
    #                   bodies that fulfil the criteria. """

    t = trigger
    adjacency = mesh.face_adjacency

    components = tr.graph.connected_components(
        edges=adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=min_len,
        engine="scipy")

    keep = []
    level = np.ceil(1/trigger)-1
    i = 0
    trigger = (trigger * mesh.faces.shape[0])

    for c in components:
        if c.shape[0] > trigger:
            keep.append( c )
            i +=1
            if i==level:
                break
    if len(keep)>0:
        lista = mesh.submesh(keep, only_watertight=False)
        return np.sum( lista )
    print("No bodyes are suited to complete {}% of the mesh...".format( t*100 ))
    return mesh



def circ_mayor( mesh, just_value=True ):
    """ Find a plausible maximum girth by slicing the reconstruction
    #   of the mesh all allong its major axis at determined intervals.
    #   params:
    #       mesh:       Trimesh. Model to be sliced in order to found
    #                   a plausible maximum girth.
    #       just_value: Bool. Whether to return or not just the lenght
    #                   of the major slice, or such lenght and the 
    #                   corresponding slice.
    #   return:
    #       lenght:     Float. Length of the major slice found.
    #       diametro:   Trimesh.Path2D. Major slice representation."""
    f = mesh.principal_inertia_vectors

    imin, imax = np.min(mesh.vertices),np.max(mesh.vertices)
    e = (imax)-(imin)
    h = e/30
    r = np.arange(imin, imax+.5, h)
    s = mesh.section_multiplane(np.min(mesh.vertices, axis=0), f[0], r)
    s1 = mesh.section_multiplane(np.min(mesh.vertices, axis=0), -f[0], r)
    s.extend(s1)
    s = [x for x in s if x!=None]
    diametro = sorted(s, key=lambda x:x.length, reverse=True)
    if len( diametro )==0:
        if just_value:
            return 0
        else:
            return 0, None
    if just_value:
        return diametro[0].length
    return diametro[0].length, diametro

def get_cuasi_medial_axis( slices, return_3DView=False ):
    """ Reconstructs a plausible medial axis based on the slices of
    #   obtained from the mesh:
    #   params:
    #       slices:             array of Trimesh.path2D. All the slices obtained
    #                           from the reconstructed mesh.
    #       return_3DView:      whether to return or not a 3D view of the obtained
    #                           axis.
    #   return:
    #       skeleton:           array of float. Coordinates composing the internal
    #                           axis.
    #       view_3D(optional):  Trimesh. 3D mesh of the slices used to compute the axis.
    """
    skeleton = []
    view_3D = []
    for s in slices:
        s = s.to_3D()
        view_3D.append( s )
        skeleton.append( s.centroid )
    skeleton = np.array( skeleton )

    eje_mayor = np.argmax(np.max(skeleton, axis=0)-np.min(skeleton, axis=0))
    skeleton = np.array(sorted( skeleton, key=lambda x:x[eje_mayor]))
    
    if return_3DView:
        return skeleton, view_3D
    return skeleton

def get_total_length( skeleton ):
    """ Length computation based on the coordinates.
    #   params:
    #       skeleton:   array. Coordinates composing the axis.
    #   return:
    #       float:      Total lengh of the internal axis.
    """    
    dist = skeleton[:-1] - skeleton[1:]
    dist = np.linalg.norm(dist, axis=1)
    return dist.sum()