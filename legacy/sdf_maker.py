from numpy.core.defchararray import rindex
from utils import load_dict, save_list_to_txt
import numpy as np
import os

class World_Maker:
    def __init__(self):
        self.obj_ctr = 0

    def make_sdf(self, name, stl_path, orig, objs):
        self.sdf = []
        self.make_header()
        self.make_block(name, stl_path, orig)
        self.make_config(name)
        for obj in objs:
            self.make_obj(name, obj)
        self.make_footer()

    def make_config(self, name):
        self.config = ['<?xml version="1.0"?>',
        ' ',
        '<model>',
        '  <name>block_'+name+'</name>',
        '  <version>1.0</version>',
        '  <sdf>model.sdf</sdf>',
        ' ',
        '  <author>',
        '    <name>Antoine Richard</name>',
        '    <email>antoine.richard@gatech.edu</email>',
        '  </author>',
        ' ',
        '  <description>',
        '    This is a computer generated model, it\'s a chunk from a bigger map',
        '  </description>',
        '</model>']

    def make_header(self):
        header = ['<?xml version="1.0" ?>',
        '<sdf version="1.5">']
        self.sdf += header

    def make_block(self, name, stl_path, orig):
        lake = ['   <model name="block_'+name+'">',
        '        <static>1</static>',
        '        <link name="lake_'+name+'">',
        '            <pose>'+str(orig[0])+' '+str(orig[1])+' 0 0 0 0</pose>',
        '            <collision name="collision">',
        '                <geometry>',
        '                    <mesh>',
        '                        <uri>file://'+stl_path+'</uri>',
        '                        <scale>1.0 1.0 1.0</scale>',
        '                    </mesh>',
        '                </geometry>',
        '                <pose>0 0 0 0 0 0</pose>',
        '            </collision>',
        '            <visual name="visual">',
        '                <geometry>',
        '                <mesh>',
        '                    <uri>file://'+stl_path+'</uri>',
        '                    <scale>1.0 1.0 1.0</scale>',
        '                </mesh>',
        '                </geometry>',
        '                <pose>0 0 0 0 0 0</pose>',
        '            </visual>',
        '        </link>',
        '   ']
        self.sdf += lake
    
    def make_obj(self, name, obj):
        rd = (obj[5][0] + np.random.rand()*(obj[5][1] - obj[5][0]))
        sx = obj[3][0] * rd
        sy = obj[3][1] * rd 
        sz = obj[3][2] * rd
        rx = obj[4][0] + obj[6][0]*np.random.rand()*3.14
        ry = obj[4][1] + obj[6][1]*np.random.rand()*3.14
        rz = obj[4][2] + obj[6][2]*np.random.rand()*3.14
        obj = ['        <link name="'+obj[0]+'_obj_'+str(self.obj_ctr)+'">',
        '            <pose>'+str(obj[1][0])+' '+str(obj[1][1])+' '+str(obj[1][2])+' '+str(rx)+' '+str(ry)+' '+str(rz)+'</pose>',
        '            <collision name="collision">',
        '                <geometry>',
        '                    <mesh>',
        '                        <uri>file://'+obj[2]+'</uri>',
        '                        <scale>'+str(sx)+' '+str(sy)+' '+str(sz)+'</scale>',
        '                    </mesh>',
        '                </geometry>',
        '                <pose>0 0 0 0 0 0</pose>',
        '            </collision>',
        '            <visual name="visual">',
        '                <geometry>',
        '                <mesh>',
        '                    <uri>file://'+obj[2]+'</uri>',
        '                    <scale>'+str(sx)+' '+str(sy)+' '+str(sz)+'</scale>',
        '                </mesh>',
        '                </geometry>',
        '                <pose>0 0 0 0 0 0</pose>',
        '            </visual>',
        '        </link>',
        '   ']
        self.obj_ctr += 1
        self.sdf += obj
        

    def make_footer(self):
        footer =['   </model>',
        '</sdf>']
        self.sdf += footer

    def save_model(self, path):
        #print(self.obj_ctr)
        save_list_to_txt(os.path.join(path,'model.sdf'), self.sdf)
        save_list_to_txt(os.path.join(path,'model.config'),self.config)

if __name__ == "__main__":
    parser = load_dict('worlds_bins/parser.pkl')

    output = 'models/bins'
    ctr = 0
    for key in parser.keys():
        print(key)
        if parser[key]['load']:
            path2stl = parser[key]['path_stl']
            print(path2stl)
            world = MakeWorld(mesh = path2stl, name = str(key), orig = parser[key]['orig'], bushes=parser[key]['bushes'])
            save_path = os.path.join(output,str(ctr//1000),'block_'+str(key))
            os.makedirs(os.path.join(save_path),exist_ok=True)
            world.save_model(save_path)
            parser[key]['model_path'] = 'model://block_'+str(key)
            ctr += 1
        else:
            parser[key]['model_path'] = 'none'

