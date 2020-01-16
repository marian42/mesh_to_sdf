def scale_to_unit_sphere(mesh):
    import trimesh

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    origin = mesh.bounding_box.centroid
    vertices = mesh.vertices - origin
    distances = np.linalg.norm(vertices, axis=1)
    size = np.max(distances)
    vertices /= size

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def get_mesh_sdf(mesh, method='scan', bounding_radius=1, scan_resolution=400, sample_point_count=10000000, calculate_normals=True):
    import mesh_sdf

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")
        
    if method == 'scan':
        return mesh_sdf.create_from_scans(mesh, bounding_radius=bounding_radius, scan_resolution=scan_resolution, calculate_normals=calculate_normals)
    elif method == 'sample':
        return mesh_sdf.sample_from_mesh(mesh, sample_point_count=sample_point_count, calculate_normals=calculate_normals)        
    else:
        raise ValueError('Unknown point sampling method: {:s}'.format(method))