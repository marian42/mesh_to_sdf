import numpy as np
import mesh_to_sdf.surface_point_cloud
from mesh_to_sdf.utils import scale_to_unit_cube, scale_to_unit_sphere, get_raster_points, check_voxels
import trimesh

class BadMeshException(Exception):
    pass

def get_surface_point_cloud(mesh, surface_point_method='scan', bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, calculate_normals=True):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")

    if bounding_radius is None:
        bounding_radius = np.max(np.linalg.norm(mesh.vertices, axis=1)) * 1.1
        
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


def mesh_to_voxels(mesh, voxel_resolution=64, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, pad=False, check_result=False):
    mesh = scale_to_unit_cube(mesh)

    points = get_raster_points(voxel_resolution)    
    sdf = mesh_to_sdf(mesh, points, surface_point_method, sign_method, 3**0.5, scan_count, scan_resolution, sample_point_count, normal_sample_count)
    voxels = sdf.reshape((voxel_resolution, voxel_resolution, voxel_resolution))

    if check_result and not check_voxels(voxels):
        raise BadMeshException()

    if pad:
        voxels = np.pad(voxels, 1, mode='constant', constant_values=1)

    return voxels

# Sample some uniform points and some normally distributed around the surface as proposed in the DeepSDF paper
def sample_sdf_near_surface(mesh, number_of_points = 500000, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, min_size=0):
    mesh = scale_to_unit_sphere(mesh)
    
    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, 1, scan_count, scan_resolution, sample_point_count, calculate_normals=sign_method=='normal')

    query_points = []
    surface_sample_count = int(number_of_points * 47 / 50) // 2
    surface_points = surface_point_cloud.get_random_surface_points(surface_sample_count, use_scans=surface_point_method=='scan')
    query_points.append(surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3)))
    query_points.append(surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3)))

    unit_sphere_sample_count = number_of_points - surface_sample_count
    unit_sphere_points = np.random.uniform(-1, 1, size=(unit_sphere_sample_count * 2, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]
    query_points.append(unit_sphere_points[:unit_sphere_sample_count, :])
    query_points = np.concatenate(query_points).astype(np.float32)

    if sign_method == 'normal':
        sdf = surface_point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=False)
    elif sign_method == 'depth':
        sdf = surface_point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=True, sample_count=sample_point_count)
    else:
        raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))
    
    model_size = np.count_nonzero(sdf[-unit_sphere_sample_count:] < 0) / unit_sphere_sample_count
    if model_size < min_size:
        raise BadMeshException()

    return query_points, sdf