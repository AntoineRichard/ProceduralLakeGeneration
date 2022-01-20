import bpy
import os

def cleanScene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

def loadMesh(path):
    bpy.ops.import_mesh.stl(filepath=path)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.mode_set(mode="OBJECT")

def addUVMap():
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.unwrap(method="ANGLE_BASED",margin=0.001)
    for context in bpy.context.screen.areas:
        print(context.type)
        if context.type == "VIEW_3D":
            obj_context = context
        if context.type == "IMAGE_EDITOR":
            uv_context = context
    bpy.ops.transform.resize({"area" : uv_context}, value=(5, 5, 1))
    bpy.ops.object.mode_set({"area":obj_context}, mode="OBJECT")

def initContext():
    for context in bpy.context.screen.areas:
        print(context.type)
        if context.type == "VIEW_3D":
            obj_context = context
        if context.type == "IMAGE_EDITOR":
            uv_context = context

def saveMesh(path):
    bpy.ops.export_scene.obj(filepath=path,
                                check_existing=False, axis_forward='-Z',
                                axis_up='Y', filter_glob="*.obj;*.mtl",
                                use_selection=True, use_animation=False,
                                use_mesh_modifiers=False, use_edges=True,
                                use_smooth_groups=False, use_smooth_groups_bitflags=False,
                                use_normals=True, use_uvs=True,
                                use_materials=True, use_triangles=False,
                                use_nurbs=False, use_vertex_groups=False,
                                use_blen_objects=True, group_by_object=False,
                                group_by_material=False, keep_vertex_order=False,
                                global_scale=1, path_mode="AUTO")

if __name__ == "__main__":
    source = "raw_generation"
    inputs = "parts"
    outputs = "parts_uv"
    cleanScene()
    initContext()
    for folder in os.listdir(source):
        print(folder)
        os.makedirs(os.path.join(source,folder,outputs),exist_ok=True)
        for part in os.listdir(os.path.join(source,folder,inputs)):
            loadMesh(os.path.join(source,folder,inputs,part))
            addUVMap()
            saveMesh(os.path.join(source,folder,outputs,part.split('.stl')[0]+'.obj'))
            cleanScene()