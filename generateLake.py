from random_lake_generator import RandomLakeGenerator
from dem_generator import DemGenerator
from vegetation_generator import VegetationGenerator, objects
from stl_maker import generateSTLs

num_lakes_to_gen = 9

for i in range(num_lakes_to_gen):
    RLG = RandomLakeGenerator(dir_name='RLG/lake'+str(i), save_grids=True)
    RLG.run()
    DEMG = DemGenerator('RLG/lake'+str(i)+'/contours.pkl', 'RLG/lake'+str(i), save_png=True)
    DEMG.run()
    VG = VegetationGenerator(objects, 'RLG/lake'+str(i))
    generateSTLs('RLG/lake'+str(i)) 