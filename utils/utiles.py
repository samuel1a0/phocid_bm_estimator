import numpy as np
import os
import trimesh as tr

def remove_floor( mesh, floor, distance=.02, isol_tol=.1 ):
    pq = tr.proximity.ProximityQuery( floor )
    a = mesh.copy()

    d,idx = pq.vertex( a.vertices )
    mask = d>distance
    a = delete_vertices(a, mask=mask)
#     mask = d<distance
#     a.update_vertices(mask)

#     b = isolate( a, trigger=isol_tol)
    b = isolate_reunite( a, trigger=isol_tol )
    return b

def delete_vertices( mesh, mask, color=True ):
    """ Elimina del mesh los vértices indicados según las mascara 'mask' """
    idx = np.where( mask )
    new_coords = mesh.vertices[ idx[0] ]
#     if color:
#         if isinstance(mesh.visual, tr.visual.color.ColorVisuals):
#             new_cols = mesh.visual.vertex_colors[ idx[0] ].copy()
#         else:
#             colors = mesh.visual.to_color()
#             new_cols = colors.vertex_colors[ idx[0] ].copy()


    old2new = -( np.ones( len( mesh.vertices ) ) )
    old2new[ idx[0] ] = np.arange( len( idx[0] ) )

    new_faces = old2new[ np.asarray( mesh.faces ) ]
    new_faces = new_faces[ ( new_faces != -1 ).all( axis = 1 ) ]

    n_mesh = tr.Trimesh( vertices=new_coords, faces=new_faces )
#     if color:
#         n_mesh.visual.vertex_colors = new_cols

    return n_mesh

def isolate( mesh, trigger=.05, stop=50 ):
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
#     trigger = (trigger * mesh.vertices.shape[0])
    trigger = (trigger * mesh.faces.shape[0])

    for c in components:
#         v = np.unique( mesh.faces[ c ].flatten() )
#         if v.shape[0] > trigger:
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
    dist = skeleton[:-1] - skeleton[1:]
    dist = np.linalg.norm(dist, axis=1)
    return dist.sum()