import numpy as np
import vtk
import os
import pickle

class STL_Maker:
    def __init__(self, res=0.1, reduction=0.99):
        self.res = res
        self.reduction = reduction

        self.height_map = None
        self.Id = None
        self.points = vtk.vtkPoints()
        self.volume = vtk.vtkPolyData()

    def loadHeightMap(self, hm):
        self.height_map = hm.copy()
        self.Id = np.zeros_like(hm,dtype=np.int64)
        #print('Height Map loaded')

    def buildPoints(self):
        for i in range(self.height_map.shape[0]):
            for j in range(self.height_map.shape[1]):
                self.Id[i,j] = i*self.height_map.shape[1]+j
                self.points.InsertPoint(self.Id[i,j],(i*self.res,j*self.res, self.height_map[i,j]*self.res))
        #print('Points built')

    def buildVolume(self):
        self.volume.SetPoints(self.points)
        cells = vtk.vtkCellArray()
        for i in range(1,self.height_map.shape[0]):
            strip = vtk.vtkTriangleStrip()
            strip.GetPointIds().SetNumberOfIds(2*self.height_map.shape[1])
            lp=list(self.Id[i-1,:])
            lc=list(self.Id[i,:])
            for k,(p,q) in enumerate(zip(lp,lc)):
                strip.GetPointIds().SetId(2*k+0,p)
                strip.GetPointIds().SetId(2*k+1,q)
            cells.InsertNextCell(strip)
        self.volume.SetStrips(cells)
        self.volume.Modified()
        if vtk.VTK_MAJOR_VERSION <= 5:
            self.volume.Update()
        #print('Volume built')

    def simplifyVoume(self):
        triangulation=vtk.vtkTriangleFilter()
        triangulation.SetInputData(self.volume)
        triangulation.Update()
        self.tvolume = triangulation.GetOutput()
        decimate=vtk.vtkDecimatePro()
        decimate.SetInputData(self.tvolume)
        decimate.SetTargetReduction(self.reduction)
        decimate.SetPreserveTopology(True)
        decimate.SetBoundaryVertexDeletion(False)
        decimate.Update()
        self.tvolume = decimate.GetOutput()
    
    def saveSTL(self, path):
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(path)
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(self.tvolume)
        else: 
            writer.SetInputData(self.tvolume)
        writer.Write()

    def makeMesh(self, hm):
        self.loadHeightMap(hm)
        self.buildPoints()
        self.buildVolume()
        self.simplifyVoume()

if __name__ == "__main__":
    output = "Media/assets/dem_bins"
    parser_path = "tmp/parser.pkl"
    dem_path = "tmp/local_dem"
    with open(parser_path, 'rb') as handle:
        parser = pickle.load(handle) 
    ctr = 0
    for key in parser.keys():
        print(key)
        if parser[key]['load']:
            STLM = STL_Maker(reduction=0.975)
            STLM.makeMesh(np.load(parser[key]['path_dem']))
            save_path = os.path.join(output,str(ctr//1000))
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, str(key)+'.stl')
            STLM.saveSTL(save_path)
            parser[key]['stl_path'] = save_path
            ctr += 1
        else:
            parser[key]['stl_path'] = 'none'
    with open(parser_path, 'wb') as handle:
        pickle.dump(parser, handle, protocol=2) 
            
