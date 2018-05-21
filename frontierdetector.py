import cv2
import numpy as np
import skimage
import skimage.measure
import skimage.morphology
import os.path

class FrontierDetector:

    def __init__(self):
        self.frontiers = []
        self.previous_map = None
        self.current_map = None
        self.frontiers_map = None
        self.diff_map = None

        self.k_x  = np.array(( [ 0, -1], [ 0,  1] ), dtype="int" )
        self.k_y  = np.array(( [ 0,  0], [-1,  1] ), dtype="int" )
        self.k_xy = np.array(( [-1,  0], [ 0,  1] ), dtype="int" )
        self.k_yx = np.array(( [ 0, -1], [ 1,  0] ), dtype="int" )
        self.beta = 2
        pass

    def detect_frontiers(self, map):
        responce = self.get_kernels_responce(map.copy())
        obstacles = self.get_obstacle_map(map.copy())
        responce_obstacles = self.get_kernels_responce(obstacles)
        self.frontiers_map = (responce - self.beta * responce_obstacles) > 0
        labels = skimage.measure.label(self.frontiers_map, background = 0)
        centroids = []
        areas = []
        for i in range(1, np.max(labels) + 1):
            props = skimage.measure.regionprops((labels == i).astype(int))
            centroids.append(props[0].centroid)
            areas.append(props[0].area)
        for i in range(len(centroids)):
            self.frontiers.append({'centroid': centroids[i],
                                   'area': areas[i],
                                   'parent': -1,
                                   'cost': 0})

    def frontiers_naive(self, map):
        self.current_map = map.copy()
        self.diff_map = map.copy().astype(np.float)
        if self.previous_map is not None:
            self.diff_map[map == self.previous_map] = -1.0
        self.detect_frontiers(self.diff_map.copy())
        self.previous_map = map.copy()
        pass

    def frontiers_morph(self, map, k_size):
        self.current_map = map.copy()
        self.diff_map = map.copy().astype(np.float)
        if self.previous_map is not None:
            mask = np.uint8(self.previous_map != -1.0)
            kernel = np.ones((k_size, k_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            self.diff_map[mask == 1] = 1.0
        self.detect_frontiers(self.diff_map.copy())
        self.previous_map = map.copy()

    def get_kernels_responce(self, map):
        diff_x  = cv2.filter2D(map, -1,  self.k_x)
        diff_y  = cv2.filter2D(map, -1,  self.k_y)
        diff_xy = cv2.filter2D(map, -1, self.k_xy)
        diff_yx = cv2.filter2D(map, -1, self.k_yx)
        return np.abs(diff_x + diff_y + diff_xy + diff_yx)

    def get_obstacle_map(self, map):
        return (map == 1.0).astype(np.float)