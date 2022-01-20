from posixpath import dirname
from matplotlib import pyplot as plt
import numpy as np
import uuid
import random
import cv2
import os

from utils import make_gaussian_kernel, save_dict

class vegetation_generator:
    def __init__(self, objects, dir_name):
        self.res = 0.1
        self.objects = objects
        self.dir_name = dir_name
        self.load_dem_and_mask()
        self.generate()
        
       
    def generate(self):
        gen_objs = []
        for obj in self.objects:
            gen_objs += self.spawn_objects(obj)
        print("Generated "+str(len(gen_objs))+" objects.")
        save_dict(gen_objs, os.path.join(self.dir_name,'objects'))
         

    def load_dem_and_mask(self):
        self.mask = np.load(os.path.join(self.dir_name,"mask.npz"))['data']
        self.dem = np.load(os.path.join(self.dir_name,"dem.npz"))['data']

    def generate_spawn_area(self, max_h, min_h):
        spawn_mask = (self.dem*self.res < max_h)*(self.dem*self.res > min_h)
        spawn_area = np.sum(spawn_mask)*(self.res*self.res)
        spawn_positions = np.argwhere(spawn_mask == 1)
        return spawn_positions, spawn_mask, spawn_area

    def spawn_objects(self, obj):
        positions, mask, area = self.generate_spawn_area(obj["max_spawn_alt"],obj["min_spawn_alt"])
        if obj["spawn_mode"] == "poisson_clusters":
            return self.make_poisson_clusters(positions, mask, area, obj)
        elif obj["spawn_mode"] == "uniform":
            return self.make_uniform(positions, area, obj)
        else:
            raise ValueError("Unknown sampling mode")

    def make_uniform(self, positions, area, obj):
        nb_points = int((area*obj["density"])*(1+np.random.rand()*obj["randomness"] - obj["randomness"]/2))
        p = np.ones((positions.shape[0]))
        p = p/np.sum(p)
        points = []
        print("Generating "+str(nb_points)+" "+obj["name"]+".")
        for i in range(nb_points):
            gen = {}
            gen['type'] = obj['name']
            gen['asset'], gen['scale'], gen['orientation'] = random.choice(list(zip(obj['assets'],obj['scale'], obj['orientation'])))
            gen['random_scale'] = obj["random_scale"]
            gen['random_rotation'] = obj["random_rotation"]
            idx = np.random.choice(np.arange(positions.shape[0]), p=p)
            gen['position'] = positions[idx]
            points.append(gen)
        return points

    def make_poisson_clusters(self, positions, mask, area, obj):            
        clusters = self.weighted_sampling(positions, mask, area, obj)
        clusters = self.populate_clusters(clusters, obj, mask)
        return clusters

    def weighted_sampling(self, positions, mask, area, obj):
        nb_points = int((area*obj["density"])*(1+np.random.rand()*obj["randomness"] - obj["randomness"]/2))
        nb_cluster = int((nb_points/obj["obj_per_cluster"]))#*(1+np.random.rand()*obj["randomness"] - obj["randomness"]/2))
        padding = 1 + int((1+obj['randomness'])*obj["cluster_size"]/self.res)
        canvas = np.zeros((mask.shape[0]+padding*2,mask.shape[1]+padding*2))
        clusters = []
        print("Generating approximatively "+str(nb_points)+" "+obj["name"]+", gathered in "+str(nb_cluster)+" clusters.")
        p = np.ones((positions.shape[0]))
        p = p/np.sum(p)
        for i in range(nb_cluster):
            width = int((obj["cluster_size"]*2/self.res)*(1+np.random.rand()*obj["randomness"]-obj["randomness"]/2))
            width += 1.0*((width % 2) == 0)
            cluster_kernel = make_gaussian_kernel(int(width),2)
            idx = np.random.choice(np.arange(positions.shape[0]), p=p)
            point = positions[idx] + padding
            canvas[point[0]-int(width/2):point[0]+int(width/2)+1, point[1]-int(width/2):point[1]+int(width/2)+1] += cluster_kernel
            adjusted_canvas = canvas[padding:-padding,padding:-padding]
            adjusted_canvas = adjusted_canvas * mask
            p = 1/(adjusted_canvas[tuple(positions.T)] + 0.001)
            p = p/np.sum(p)
            clusters.append(point - padding)
        return clusters
    
    def populate_clusters(self, clusters_position, obj, mask):            
        objects = []
        obj_footprint = int(obj["footprint"]/self.res)
        obj_footprint += int(1.0*((obj_footprint % 2) == 0))
        obj_kernel = make_gaussian_kernel(obj_footprint, 4)
        padding = 1 + obj_footprint
        for pos in clusters_position:
            width = int((obj["cluster_size"]*2/self.res)*(1+np.random.rand()*obj["randomness"]-obj["randomness"]/2))
            width += int(1.0*((width % 2) == 0))
            cluster_kernel = make_gaussian_kernel(width,2)
            canvas = np.zeros((width + padding*2, width + padding*2))
            canvas[padding:-padding,padding:-padding] = cluster_kernel
            canvas = canvas*mask[pos[0]-int(width/2)-padding:pos[0]+int(width/2)+1+padding, pos[1]-int(width/2)-padding:pos[1]+int(width/2)+1+padding]
            nb_points = int(np.round(obj["obj_per_cluster"]*(1+np.random.rand()*obj["randomness"] - obj["randomness"])))
            assets = random.choice(obj['assets'])
            assets, scales, orientations = random.choice(list(zip(obj['assets'],obj['scale'], obj['orientation'])))
            for _ in range(nb_points):
                gen = {}
                gen["type"] = obj["name"]
                gen['asset'], gen['scale'], gen['orientation'] = random.choice(list(zip(assets, scales, orientations)))
                gen['scale'] = (np.array(gen['scale']) * (np.random.rand() * (obj['random_scale'][1] + obj['random_scale'][0]) + obj['random_scale'][0])).tolist()
                #gen['random_scale'] = obj["random_scale"]
                gen['random_rotation'] = obj["random_rotation"]
                positions = np.argwhere(canvas != 0)
                p = canvas[tuple(positions.T)]
                p = p/np.sum(p)
                idx = np.random.choice(np.arange(positions.shape[0]), p = p)
                point = positions[idx]
                canvas[point[0]-int(obj_footprint/2):point[0]+int(obj_footprint/2)+1, point[1]-int(obj_footprint/2):point[1]+int(obj_footprint/2)+1] -= obj_kernel
                canvas[canvas<0] = 0
                point = point - padding - width/2 + pos
                gen["position"] = point
                objects.append(gen)
        return objects

if __name__ == "__main__":
    # ADD STUFF TO OBJ DICT ABOUT SETTINGS. REMOVE MIXING ARG REPLACE BY LIST OF ASSETS
    trees = {}
    trees['name'] = 'trees'
    trees['density'] = 0.1
    trees['randomness'] = 0.75
    trees['max_spawn_alt'] = 1.0
    trees['min_spawn_alt'] = -0.25
    trees['spawn_mode'] = "poisson_clusters"
    trees['obj_per_cluster'] = 4
    trees['cluster_size'] = 5.0
    trees['assets'] =  [['oak_1.usd',
                        'oak_2.usd'],
                        ['spruce_1.usd',
                        'spruce_2.usd',
                        'spruce_3.usd',
                        'spruce_small_1.usd',
                        'spruce_small_2.usd']]
    trees['scale'] = [[[0.63,0.63,0.63],
                        [0.83,0.83,0.83]],
                        [[1.14, 1.14,1.14],
                        [0.94,0.94,0.94],
                        [0.72,0.72,0.72],
                        [2.76,2.76,2.76],
                        [2.0,2.0,2.0]]]
    trees['orientation'] = [[[1.57, 0.0, 0.0],
                            [1.57,0.0,0.0]],
                            [[1.57,0.0,0.0],
                            [1.57,0.0,0.0],
                            [1.57,0.0,0.0],
                            [1.57,0.0,0.0],
                            [1.57,0.0,0.0]]]
    trees['random_rotation'] = [False, False, True]
    trees['random_scale'] = [0.75, 1.25]
    trees['footprint'] = 2
    
    bushes = {}
    bushes['name'] = 'bushes'
    bushes['density'] = 0.06
    bushes['randomness'] = 0.75
    bushes['max_spawn_alt'] = 1.0
    bushes['min_spawn_alt'] = -0.25
    bushes['spawn_mode'] = "poisson_clusters"
    bushes['obj_per_cluster'] = 10
    bushes['cluster_size'] = 5.0
    bushes['assets'] =  [['mountain_ash_1.usd',
                     'mountain_ash_2.usd',
                     'mountain_ash_3.usd']]
    bushes['scale'] = [[[1.0,1.0,1.0],
                     [1.0,1.0,1.0],
                     [1.0,1.0,1.0]]]
    bushes['orientation'] = [[[1.57, 0.0, 0.0],
                         [1.57,0.0,0.0],
                         [1.57,0.0,0.0]]]
    bushes['random_rotation'] = [False, False, True]
    bushes['random_scale'] = [0.75, 1.25]
    bushes['footprint'] = 2

    ferns = {}
    ferns['name'] = 'ferns'
    ferns['density'] = 1.5
    ferns['randomness'] = 0.3
    ferns['max_spawn_alt'] = 0.5
    ferns['min_spawn_alt'] = -2.0
    ferns['spawn_mode'] = "poisson_clusters"
    ferns['obj_per_cluster'] = 80
    ferns['cluster_size'] = 9.0
    ferns['assets'] = [['reed.usd'],
                   ['fern_1.usd',
                   'fern_2.usd'],
                   ['grass_3.usd',
                   'grass_4.usd',
                   'grass_6.usd']]
    ferns['scale'] = [[[0.2,0.2,0.2]],
                   [[0.5,0.5,0.5],
                   [0.5,0.5,0.5]],
                   [[1.3,1.3,1.3],
                   [1.3,1.3,1.3],
                   [1.3,1.3,1.3]]]
    ferns['orientation'] = [[[1.57,0.0,0.0]],
                       [[3.14,0.0,0.0],
                       [3.14,0.0,0.0]],
                       [[3.14,0.0,0.0],
                       [3.14,0.0,0.0],
                       [3.14,0.0,0.0]]]
    ferns['random_rotation'] = [False, False, True]
    ferns['random_scale'] = [0.5, 1]
    ferns['footprint'] = 0.5

    rocks = {}
    rocks['name'] = 'rocks'
    rocks['density'] = 0.025
    rocks['randomness'] = 0.2
    rocks['max_spawn_alt'] = 1.0
    rocks['min_spawn_alt'] = -1.0
    rocks['spawn_mode'] = "poisson_clusters"
    rocks['obj_per_cluster'] = 15
    rocks['cluster_size'] = 6.0
    rocks['assets'] =  [['big_rock_2.usd',
                        'big_rock_3.usd',
                        'flat_rock_1.usd',
                        'flat_rock_2.usd',
                        'flat_rock_3.usd',
                        'flat_rock_5.usd',
                        'rock_1.usd',
                        'rock_2.usd',
                        'rock_3.usd',
                        'rock_4.usd']]
    rocks['scale'] = [[[0.75,0.75,0.75],
                        [0.75,0.75,0.75],
                        [1.5,1.5,1.5],
                        [1.5,1.5,1.5],
                        [1.5,1.5,1.5],
                        [1.5,1.5,1.5],
                        [1.5,1.5,1.5],
                        [1.5,1.5,1.5],
                        [1.5,1.5,1.5],
                        [1.5,1.5,1.5]]]
    rocks['orientation'] = [[[0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0]]]
    rocks['random_rotation'] = [True, True, True]
    rocks['random_scale'] = [0.5, 1.5]
    rocks['footprint'] = 0.5

    branches = {}
    branches['name'] = 'branches'
    branches['density'] = 0.001
    branches['randomness'] = 0
    branches['max_spawn_alt'] = -2
    branches['min_spawn_alt'] = -4
    branches['spawn_mode'] = "uniform"
    branches['assets'] = ['fallen_tree_1.usd',
                          'fallen_tree_2.usd',
                          'log.usd']
    branches['orientation'] = [[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]]
    branches['scale'] = [[0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25],
                        [0.125, 0.125, 0.125]]
    branches['random_scale'] = [0.75, 1.25]
    branches['random_rotation'] = [False, False, True]

    objects = [trees, bushes, ferns, rocks, branches]
    RLG = vegetation_generator(objects, 'raw_generation/gen0')
    RLG = vegetation_generator(objects, 'raw_generation/gen1')
    RLG = vegetation_generator(objects, 'raw_generation/gen2')
    RLG = vegetation_generator(objects, 'raw_generation/gen3')
    RLG = vegetation_generator(objects, 'raw_generation/gen4')
    RLG = vegetation_generator(objects, 'raw_generation/gen5')
    RLG = vegetation_generator(objects, 'raw_generation/gen6')
    RLG = vegetation_generator(objects, 'raw_generation/gen7')
    RLG = vegetation_generator(objects, 'raw_generation/gen8')