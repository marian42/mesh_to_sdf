import trimesh
import numpy as np

def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    origin = mesh.bounding_box.centroid
    vertices = mesh.vertices - origin
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def scale_to_unit_cube(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    origin = mesh.bounding_box.centroid
    vertices = mesh.vertices - origin
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

voxel_points = dict()

def get_raster_points(voxel_resolution):
    if voxel_resolution in voxel_points:
        return voxel_points[voxel_resolution]
    
    points = np.meshgrid(
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)

    voxel_points[voxel_resolution] = points
    return points
