import numpy as np
from vtk.util import numpy_support
import pickle
import vtk
import cv2
import os

def mkVtkIdList(it):
    """
    Creates a VTK Id list.
    """
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil 

def makeFlatSurface(px,py,z,size):
    """
    Creates flat tiles.
    """
    #Array of vectors containing the coordinates of each point
    nodes = np.array([[px, py, z], [px + size[0]//2, py, z], [px+size[0], py, z],
                      [px+size[0], py+size[1]//2, z], [px+size[0], py+size[1], z],
                      [px+size[0]//2, py+size[1], z], [px, py+size[1], z], [px, py+size[1]//2, z],
                      [px+size[0]//2, py+size[1]//2, z]])
    #Array of tuples containing the nodes correspondent of each element
    elements =[(0, 1, 8, 7), (7, 8, 5, 6), (1, 2, 3, 8), (8, 3, 4, 
                        5)]
    #Make the building blocks of polyData attributes
    Mesh = vtk.vtkPolyData()
    Points = vtk.vtkPoints()
    Cells = vtk.vtkCellArray()  
    #Load the point and cell's attributes
    for i in range(len(nodes)):
        Points.InsertPoint(i, nodes[i])
    for i in range(len(elements)):
        Cells.InsertNextCell(mkVtkIdList(elements[i]))
    #Assign pieces to vtkPolyData
    Mesh.SetPoints(Points)
    Mesh.SetPolys(Cells)
    return Mesh

def decimate(height_map, corrected_absolute_error, scale):
    """
    Transforms the heightmap into a mesh
    """
    # Build VTK Array
    row_size, col_size = height_map.shape
    number_of_elevation_entries = height_map.size
    vectorized_elevations = np.reshape(height_map, (number_of_elevation_entries, 1))
    vtk_array = numpy_support.numpy_to_vtk(vectorized_elevations, deep=True,
                             array_type=vtk.VTK_FLOAT)
    # Make a VTK heightmap
    image = vtk.vtkImageData()
    image.SetDimensions(col_size, row_size, 1)
    image.AllocateScalars(vtk_array.GetDataType(), 4)
    image.GetPointData().GetScalars().DeepCopy(vtk_array)
    # Decimate the heightmap
    deci = vtk.vtkGreedyTerrainDecimation()
    deci.SetInputData(image)
    deci.BoundaryVertexDeletionOn()
    deci.SetErrorMeasureToAbsoluteError()
    deci.SetAbsoluteError(corrected_absolute_error)
    deci.Update()
    tvolume = deci.GetOutput()
    # Set the scale correctly
    transform = vtk.vtkTransform()
    transform.Scale(scale)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(tvolume)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    tvolume = transformFilter.GetOutput()
    # Return scaled object
    return tvolume

def makeVTKPlane(p, normal):
    plane = vtk.vtkPlane()
    plane.SetNormal(normal)
    plane.SetOrigin(p)
    return plane

def applyCut(plane, poly):
    """
    Cut a VTK poly using a plane, and returns the side that aligns with the normal.
    """
    # Create cut tool
    cut = vtk.vtkClipPolyData()
    # Feed tool
    if vtk.VTK_MAJOR_VERSION <= 5:
        cut.SetInput(poly)
    else:
        cut.SetInputData(poly)
    # Cut
    cut.SetClipFunction(plane)
    cut.Update()
    # Get normal side
    return cut.GetOutput()

def clearOffset(poly):
    """
    sets the origin of the mesh to be 0,0,z
    """
    Bounds = poly.GetBounds()
    Transform = vtk.vtkTransform()
    Transform.Translate(-Bounds[0],-Bounds[2],0)
    Filter = vtk.vtkTransformPolyDataFilter()
    Filter.SetTransform(Transform)
    Filter.SetInputData(poly)
    Filter.Update()
    return Filter.GetOutput() 

def cut(poly, path, size=50, poly_max_x=750, poly_max_y=750):
    """
    Cut the mesh into a smaller meshes of a given size.
    The way this works in by creating planes and cutting along these planes
    """
    for i in range(size,poly_max_x,size):
        # Cut and get left side of the mesh 
        plane = makeVTKPlane((0,i,0),(0,-1,0))
        tmp = applyCut(plane, poly)
        # We just got a strip that we are going to cut into cubes.
        for j in range(size,poly_max_y,size):
            plane = makeVTKPlane((j*1.0,0,0),(-1,0,0))
            output = applyCut(plane, tmp)
            plane = makeVTKPlane((j*1.0,0,0),(1,0,0))
            tmp = applyCut(plane, tmp)
            Bounds = output.GetBounds()
            if np.abs(Bounds[-1] - Bounds[-2]) < 0.15:
                output = makeFlatSurface(j-size,i-size,np.mean([Bounds[-1],Bounds[-2]]),(np.min([Bounds[1]-Bounds[0],size]),np.min([Bounds[3]-Bounds[2],size])))
            output = clearOffset(output)
            save(output,path+str(i-size)+'-'+str(j-size)+'.stl')
        # Cut and get right side of the mesh 
        plane = makeVTKPlane((0,i,0),(0,1,0))
        poly = applyCut(plane, poly)

def save(poly, path):
    """
    Saves the VTK poly as an STL.
    """
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(path)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(poly)
    else:
        writer.SetInputData(poly)
    writer.Write()

def generateSTLs(folder, size=50):
    os.makedirs(folder+'/parts',exist_ok=True)
    tmp = cv2.GaussianBlur(np.load(folder+'/dem.npz')['data'],(11,11),0)
    poly = decimate(tmp, 1.0, (0.1,0.1,0.1))
    save(poly, folder+'/dem_full.stl')
    cut(poly, folder+'/parts/part_', size=size)

if __name__ == "__main__":
    output = "Media/assets/dem_bins"
    parser_path = "tmp/parser.pkl"
    dem_path = "tmp/local_dem"
    ctr = 0
    for folder in os.listdir('raw_generation'):
        print(folder)
        os.makedirs('raw_generation/'+folder+'/parts',exist_ok=True)
        tmp = cv2.GaussianBlur(np.load('raw_generation/'+folder+'/dem.npz')['data'],(11,11),0)
        poly = decimate(tmp, 1.0, (0.1,0.1,0.1))
        save(poly, 'raw_generation/'+folder+'/dem_full.stl')
        cut(poly, 'raw_generation/'+folder+'/parts/part_', size=50)
