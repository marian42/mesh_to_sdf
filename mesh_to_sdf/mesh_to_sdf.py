import trimesh
import logging
logging.getLogger("trimesh").setLevel(9000)
import numpy as np
from sklearn.neighbors import KDTree
import skimage
import math
from sdf.scan import create_scans
import pyrender
from util import get_voxel_coordinates
import time

class BadMeshException(Exception):
    pass

def scale_to_unit_sphere(mesh, rotation_matrix=None):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    origin = mesh.bounding_box.centroid
    vertices = mesh.vertices - origin
    distances = np.linalg.norm(vertices, axis=1)
    size = np.max(distances)
    vertices /= size

    if rotation_matrix is not None:
        vertices = np.matmul(rotation_matrix, vertices.transpose()).transpose()

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

class MeshSDF:
    def __init__(self, mesh, use_scans=True):
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()
        self.mesh = mesh
        
        if use_scans:
            self.scans = create_scans(mesh)

            self.points = np.concatenate([scan.points for scan in self.scans], axis=0)
            self.normals = np.concatenate([scan.normals for scan in self.scans], axis=0)
        else:
            points, indices = mesh.sample(10000000, return_index=True)
            self.points = points
            self.normals = mesh.face_normals[indices]

        self.kd_tree = KDTree(self.points)

    def get_random_surface_points(self, count, use_scans=True):
        if use_scans:
            indices = np.random.choice(self.points.shape[0], count)
            return self.points[indices, :]
        else:
            return self.mesh.sample(count)

    def get_sdf(self, query_points, use_depth_buffer=False, sample_count=11):
        if use_depth_buffer:
            distances, _ = self.kd_tree.query(query_points)
            distances = distances.astype(np.float32).reshape(-1) * -1
            distances[self.is_outside(query_points)] *= -1
            return distances
        else:
            distances, indices = self.kd_tree.query(query_points, k=sample_count)
            distances = distances.astype(np.float32)

            closest_points = self.points[indices]
            direction_to_surface = query_points[:, np.newaxis, :] - closest_points
            inside = np.einsum('ijk,ijk->ij', direction_to_surface, self.normals[indices]) < 0
            inside = np.sum(inside, axis=1) > sample_count * 0.5
            distances = distances[:, 0]
            distances[inside] *= -1
            return distances

    def get_sdf_in_batches(self, points, batch_size=100000):
        result = np.zeros(points.shape[0])
        for i in range(int(math.ceil(points.shape[0] / batch_size))):
            start = i * batch_size
            end = min(result.shape[0], (i + 1) * batch_size)
            result[start:end] = self.get_sdf(points[start:end, :])
        return result

    def get_voxel_sdf(self, voxel_resolution = 32):
        center = self.mesh.bounding_box.centroid
        size = np.max(self.mesh.bounding_box.extents) / 2
        voxels = self.get_sdf(get_voxel_coordinates(voxel_resolution, size, center))
        voxels = voxels.reshape(voxel_resolution, voxel_resolution, voxel_resolution)
        self.check_voxels(voxels)
        return voxels

    def get_sample_points(self, number_of_points = 200000):
        ''' Use sample points as described in the DeepSDF paper '''
        points = []

        surface_sample_count = int(number_of_points * 0.4)
        surface_points = self.get_random_surface_points(surface_sample_count)
        points.append(surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3)))
        points.append(surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3)))

        unit_sphere_sample_count = int(number_of_points * 0.2)
        unit_sphere_points = np.random.uniform(-1, 1, size=(unit_sphere_sample_count * 2, 3))
        unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]
        points.append(unit_sphere_points[:unit_sphere_sample_count, :])
        points = np.concatenate(points).astype(np.float32)

        sdf = self.get_sdf(points)
        
        model_size = np.count_nonzero(sdf[-unit_sphere_sample_count:] < 0) / unit_sphere_sample_count
        if model_size < 0.015:
            raise BadMeshException()

        return points, sdf

    def get_surface_points_and_normals(self, number_of_points = 50000):
        count = self.points.shape[0]
        if count < number_of_points:
            print("Warning: Less than {:d} points sampled.".format(number_of_points))
        indices = np.arange(count)
        np.random.shuffle(indices)
        indices = indices[:number_of_points]
        return np.concatenate([self.points[indices, :], self.normals[indices, :]], axis=1)

    def check_voxels(self, voxels):
        block = voxels[:-1, :-1, :-1]
        d1 = (block - voxels[1:, :-1, :-1]).reshape(-1)
        d2 = (block - voxels[:-1, 1:, :-1]).reshape(-1)
        d3 = (block - voxels[:-1, :-1, 1:]).reshape(-1)

        max_distance = max(np.max(d1), np.max(d2), np.max(d3))
        if max_distance > 2.0 / voxels.shape[0] * 1.75: # The exact value is sqrt(3), the length of the diagonal of a cube
            raise BadMeshException()
    
    def show_pointcloud(self):
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_points(self.points, normals=self.normals))
        pyrender.Viewer(scene, use_raymond_lighting=True, point_size=8)

    def show_reconstructed_mesh(self, voxel_resolution=64):
        scene = pyrender.Scene()
        voxels = self.get_voxel_sdf(voxel_resolution=voxel_resolution)
        voxels = np.pad(voxels, 1, mode='constant', constant_values=1)
        vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(voxels, level=0, spacing=(voxel_resolution, voxel_resolution, voxel_resolution))
        reconstructed = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        reconstructed_pyrender = pyrender.Mesh.from_trimesh(reconstructed, smooth=False)
        scene.add(reconstructed_pyrender)
        pyrender.Viewer(scene, use_raymond_lighting=True)
        
    def is_outside(self, points):
        result = None
        for scan in self.scans:
            if result is None:
                result = scan.is_visible(points)
            else:
                result = np.logical_or(result, scan.is_visible(points))
        return result
