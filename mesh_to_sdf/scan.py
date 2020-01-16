import numpy as np
import math
from rendering.math import get_rotation_matrix
from sdf.pyrender_wrapper import render_normal_and_depth_buffers
from threading import Lock
import pyrender
import random

CAMERA_DISTANCE = 2
VIEWPORT_SIZE = 400

def get_camera_transform(rotation_y, rotation_x = 0):
    camera_transform = np.identity(4)
    camera_transform[2, 3] = CAMERA_DISTANCE
    camera_transform = np.matmul(get_rotation_matrix(rotation_x, axis='x'), camera_transform)
    camera_transform = np.matmul(get_rotation_matrix(rotation_y, axis='y'), camera_transform)
    return camera_transform

'''
A virtual laser scan of an object from one point in space.
This renders a normal and depth buffer and reprojects it into a point cloud.
The resulting point cloud contains a point for every pixel in the buffer that hit the model.
'''
class Scan():
    def __init__(self, mesh, rotation_y, rotation_x):
        self.camera_transform = get_camera_transform(rotation_y, rotation_x)
        self.camera_direction = np.matmul(self.camera_transform, np.array([0, 0, 1, 0]))[:3]
        self.camera_position = np.matmul(self.camera_transform, np.array([0, 0, 0, 1]))[:3]

        z_near = CAMERA_DISTANCE - 1.0
        z_far = CAMERA_DISTANCE + 1.0
        
        camera = pyrender.PerspectiveCamera(yfov=2 * math.asin(1.0 / CAMERA_DISTANCE), aspectRatio=1.0, znear = z_near, zfar = z_far)
        self.projection_matrix = camera.get_projection_matrix()

        color, depth = render_normal_and_depth_buffers(mesh, camera, self.camera_transform, VIEWPORT_SIZE)
        
        indices = np.argwhere(depth != 0)
        depth[depth == 0] = float('inf')

        # This reverts the processing that pyrender does and calculates the original depth buffer in clipping space
        self.depth = (z_far + z_near - (2.0 * z_near * z_far) / depth) / (z_far - z_near)
        
        points = np.ones((indices.shape[0], 4))
        points[:, [1, 0]] = indices.astype(float) / (VIEWPORT_SIZE -1) * 2 - 1
        points[:, 1] *= -1
        points[:, 2] = self.depth[indices[:, 0], indices[:, 1]]
        
        clipping_to_world = np.matmul(self.camera_transform, np.linalg.inv(self.projection_matrix))

        points = np.matmul(points, clipping_to_world.transpose())
        points /= points[:, 3][:, np.newaxis]
        self.points = points[:, :3]

        normals = color[indices[:, 0], indices[:, 1]] / 255 * 2 - 1
        camera_to_points = self.camera_position - self.points
        normal_orientation = np.einsum('ij,ij->i', camera_to_points, normals)
        normals[normal_orientation < 0] *= -1
        self.normals = normals

    def convert_world_space_to_viewport(self, points):
        half_viewport_size = 0.5 * VIEWPORT_SIZE
        clipping_to_viewport = np.array([
            [half_viewport_size, 0.0, 0.0, half_viewport_size],
            [0.0, -half_viewport_size, 0.0, half_viewport_size],
            [0.0, 0.0, 1.0, 0.0],
            [0, 0, 0.0, 1.0]
        ])

        world_to_clipping = np.matmul(self.projection_matrix, np.linalg.inv(self.camera_transform))
        world_to_viewport = np.matmul(clipping_to_viewport, world_to_clipping)
        
        world_space_points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        viewport_points = np.matmul(world_space_points, world_to_viewport.transpose())
        viewport_points /= viewport_points[:, 3][:, np.newaxis]
        return viewport_points

    def is_visible(self, points):
        viewport_points = self.convert_world_space_to_viewport(points)
        pixels = viewport_points[:, :2].astype(int)
        pixels = np.clip(pixels, 0, VIEWPORT_SIZE - 1)
        return viewport_points[:, 2] < self.depth[pixels[:, 1], pixels[:, 0]]

    def show_pointcloud(self):
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_points(self.points * 100, normals=self.normals))
        pyrender.Viewer(scene, use_raymond_lighting=True, point_size=8)

render_lock = Lock()

def get_camera_angles(count):
    increment = math.pi * (3 - math.sqrt(5))
    for i in range(count):
        theta = math.asin(-1 + 2 * i / (count - 1))
        phi = ((i + 1) * increment) % (2 * math.pi)
        yield phi, theta

def create_scans(mesh, camera_count=100):
    scans = []
    render_lock.acquire()

    for phi, theta in get_camera_angles(camera_count):
        scans.append(Scan(mesh, math.degrees(phi), math.degrees(theta)))

    render_lock.release()
    return scans