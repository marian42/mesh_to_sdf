import numpy as np
import mesh_to_sdf.surface_point_cloud
import trimesh


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


def get_surface_point_cloud(mesh, surface_point_method='scan', bounding_radius=1, scan_count=100, scan_resolution=400, sample_point_count=10000000, calculate_normals=True):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")
        
    if surface_point_method == 'scan':
        return surface_point_cloud.create_from_scans(mesh, bounding_radius=bounding_radius, scan_count=scan_count, scan_resolution=scan_resolution, calculate_normals=calculate_normals)
    elif surface_point_method == 'sample':
        return surface_point_cloud.sample_from_mesh(mesh, sample_point_count=sample_point_count, calculate_normals=calculate_normals)        
    else:
        raise ValueError('Unknown surface point sampling method: {:s}'.format(surface_point_method))


def mesh_to_sdf(mesh, query_points, surface_point_method='scan', sign_method='normal', bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11):
    if not isinstance(query_points, np.ndarray):
        raise TypeError('query_points must be a numpy array.')
    if len(query_points.shape) != 2 or query_points.shape[1] != 3:
        raise ValueError('query_points must be of shape N ✕ 3.')

    if bounding_radius is None:
        bounding_radius = np.max(np.linalg.norm(mesh.vertices, axis=1)) * 1.1
    
    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    point_cloud = get_surface_point_cloud(mesh, surface_point_method, bounding_radius, scan_count, scan_resolution, sample_point_count, calculate_normals=sign_method=='normal')

    if sign_method == 'normal':
        return point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=False)
    elif sign_method == 'depth':
        return point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=True, sample_count=sample_point_count)
    else:
        raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))
