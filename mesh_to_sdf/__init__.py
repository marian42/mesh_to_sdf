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


def get_surface_point_cloud(mesh, surface_point_method='scan', bounding_radius=1, scan_resolution=400, sample_point_count=10000000, calculate_normals=True):
    import surface_point_cloud

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")
        
    if surface_point_method == 'scan':
        return surface_point_cloud.create_from_scans(mesh, bounding_radius=bounding_radius, scan_resolution=scan_resolution, calculate_normals=calculate_normals)
    elif surface_point_method == 'sample':
        return surface_point_cloud.sample_from_mesh(mesh, sample_point_count=sample_point_count, calculate_normals=calculate_normals)        
    else:
        raise ValueError('Unknown surface point sampling method: {:s}'.format(surface_point_method))