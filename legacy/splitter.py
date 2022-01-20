import numpy as np
import os

from utils import load_dict, save_list_to_txt
from stl_maker import STL_Maker
from sdf_maker import World_Maker

class Splitter:
    def __init__(self, path, output):
        self.path = path
        self.res = 0.1
        self.chunk_size = 20.0
        self.chunk_pixel_size = 19.9
        self.output = output
        self.parser = []
        self.stl_maker = STL_Maker(reduction=0.95)
        self.world_maker = World_Maker()
        self.stl_output = "Media/dem_chunks"
        self.sdf_output = "models/chunks"
        self.cnt = 0

    def initializeChunks(self):
        self.maps = os.listdir(self.path)
        print(self.maps)
        rows = int(np.sqrt(len(self.maps)))
        cols = int(np.ceil(len(self.maps)/rows))
        self.map_grid = np.arange(rows*cols)
        self.map_grid = self.map_grid.reshape([rows,cols])
        map_shape = np.array(np.load(os.path.join(self.path,self.maps[0],'dem.npz'))['data'].shape)
        self.map_chunks = np.ceil(map_shape*self.res/self.chunk_size).astype(np.int32)
        world_cols = int(cols*self.map_chunks[0])
        world_rows = int(rows*self.map_chunks[1])
        self.world_grid = np.arange(world_rows*world_cols)
        self.world_grid = self.world_grid.reshape([world_rows, world_cols])
        print('Building world with',self.world_grid.shape[0],'x',self.world_grid.shape[1],'chunks.')

    def reformatObjects(self, objects):
        self.assets = np.array([o["asset"] for o in objects])
        self.scales = np.array([o["scale"] for o in objects])
        self.orientations = np.array([o["orientation"] for o in objects])
        self.random_scales = np.array([o["random_scale"] for o in objects])
        self.random_rotations = np.array([o["random_rotation"] for o in objects])
        self.positions = np.array([o["position"] for o in objects])
        self.types = np.array([o["type"] for o in objects])
        #return positions, scales, rotations, assets, types

    def get_objects(self, dem, xs, ys, s, offset):
        xc = (xs * s < self.positions[:,0])*(self.positions[:,0] < (xs+1)*s)
        yc = (ys * s < self.positions[:,1])*(self.positions[:,1] < (ys+1)*s)
        objs = self.positions[xc*yc].astype(np.int32)
        if objs.shape[0] != 0:
            z = np.expand_dims(dem[tuple(objs.T)],axis=-1)
            z = (z > 0.)*z*self.res
            o_types = self.types[xc*yc]
            o_assets = self.assets[xc*yc]
            o_scales = self.scales[xc*yc]
            o_orientations = self.orientations[xc*yc]
            o_random_scales = self.random_scales[xc*yc]
            o_random_rotations = self.random_rotations[xc*yc]
            o_positions = np.concatenate([self.positions[xc*yc]*self.res + offset, z], axis=-1)
            return zip(o_types, o_positions, o_assets, o_scales, o_orientations, o_random_scales, o_random_rotations)
        else:
            return []
        
    def cut(self):
        nb_col = self.map_grid.shape[1]
        s = int(self.chunk_size/self.res)
        for map_id in self.map_grid.flatten():
            dem = np.load(os.path.join(self.path,self.maps[map_id],'dem.npz'))['data']
            #Objects reformating
            objects = load_dict(os.path.join(self.path,self.maps[map_id],'objects.pkl'))
            self.reformatObjects(objects)
            # Grid initialization
            col_id = map_id % self.map_grid.shape[1]
            row_id = map_id//self.map_grid.shape[1]
            start = row_id*self.map_chunks[1]*self.map_chunks[0]*nb_col + col_id*self.map_chunks[1]
            world_x_offset = self.map_chunks[0] * col_id * self.chunk_size
            world_y_offset = self.map_chunks[1] * row_id * self.chunk_size
            offset = np.array([world_x_offset, world_y_offset])
            for map_bin_id in range(self.map_chunks[0]*self.map_chunks[1]):
                row_offset = (map_bin_id // self.map_chunks[1])*self.map_chunks[1]*nb_col
                pos = start + row_offset + (map_bin_id % self.map_chunks[1])
                xs = map_bin_id % self.map_chunks[1]
                ys = map_bin_id//self.map_chunks[1]
                local_dem = dem[xs*s:(xs+1)*s,ys*s:(ys+1)*s]
                # Check if block should be loaded
                to_load = not ((np.min(local_dem) > 50.0) or (np.max(local_dem) < -20))
                if to_load:
                    stl_save_path = os.path.join(self.stl_output,str(self.cnt//1000))
                    os.makedirs(stl_save_path, exist_ok=True)
                    stl_save_path = os.path.join(stl_save_path,str(pos)+'.stl')
                    sdf_save_path = os.path.join(self.sdf_output,str(self.cnt//1000),'block_'+str(pos))
                    os.makedirs(sdf_save_path, exist_ok=True)
                    orig = [pos%self.world_grid.shape[1]*self.chunk_pixel_size,
                            pos//self.world_grid.shape[0]*self.chunk_pixel_size]
                    objs = self.get_objects(dem, xs, ys, s, offset)
                    self.world_maker.make_sdf(str(pos), stl_save_path, orig, objs)
                    self.world_maker.save_model(sdf_save_path)
                    #self.stl_maker.makeMesh(local_dem)
                    #self.stl_maker.saveSTL(stl_save_path)
                    self.parser.append(str(pos)+','+str(1)+',model://block_'+str(pos))
                    self.cnt += 1
                else:
                    self.parser.append(str(pos)+','+str(0)+',none')
        print('Total spawnable blocks:',self.cnt)
        save_list_to_txt('parser.csv', self.parser)

SPL = Splitter('raw_generation','tmp')
SPL.initializeChunks()
SPL.cut()
