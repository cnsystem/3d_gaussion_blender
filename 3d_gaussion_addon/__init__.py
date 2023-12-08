bl_info = {
    "name": "3D Gaussian Splatting",
    "author": "Alex Carlier",
    "version": (0, 0, 1),
    "blender": (3, 4, 0),
    "location": "3D Viewport > Sidebar > 3D Gaussian Splatting",
    "description": "3D Gaussian Splatting tool",
}

import bpy
import mathutils
import numpy as np
import time
import random

from .plyfile import PlyData, PlyElement
from .blender_ops import *


class ImportGaussianSplatting(bpy.types.Operator):
    bl_idname = "object.import_gaussian_splatting"
    bl_label = "Import Gaussian Splatting"
    bl_description = "Import a 3D Gaussian Splatting file into the scene"
    bl_options = {"REGISTER", "UNDO"}
    
    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to the Gaussian Splatting file",
        subtype='FILE_PATH',
    )

    def execute(self, context):
        if not self.filepath:
            self.report({'WARNING'}, "Filepath is not set!")
            return {'CANCELLED'}

        bpy.context.scene.render.engine = 'CYCLES'
        if context.preferences.addons["cycles"].preferences.has_active_device():
            bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.transparent_max_bounces = 20

        RECOMMENDED_MAX_GAUSSIANS = 200_000
        ##############################
        # Load PLY
        ##############################
        start_time = time.time()
        plydata = load_ply(self.filepath)
        obj = build_mesh(plydata)
        print("Data loaded in", time.time() - start_time, "seconds")
        ##############################
        # Materials
        ##############################
        start_time = time.time()
        mat = build_material_graph()
        print("Material created in", time.time() - start_time, "seconds")
        ##############################
        # Geometry Nodes
        ##############################
        start_time = time.time()
        obj = bpy.data.objects["GaussianSplatting"]
        mat = bpy.data.materials['GaussianSplatting_bk']
        build_sphere_geometry_node(obj, mat)
        print("Geometry nodes created in", time.time() - start_time, "seconds")
        return {'FINISHED'}

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = bpy.path.abspath("//point_cloud.ply")
        
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l

class ExportGaussianSplatting(bpy.types.Operator):
    bl_idname = "object.export_gaussian_splatting"
    bl_label = "Export 3D Gaussian Splatting"
    bl_description = "Export a 3D Gaussian Splatting to file"
    
    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to the Gaussian Splatting file",
        subtype="FILE_PATH"
    )

    def execute(self, context):
        if not self.filepath.lower().endswith('.ply'):
            self.filepath += ".ply"

        obj = context.active_object

        if obj is None or "gaussian_splatting" not in obj:
            self.report({'WARNING'}, "No Gaussian Splatting selected")
            return {'CANCELLED'}

        mesh: bpy.types.Mesh = obj.data

        N = len(mesh.vertices)
        
        xyz = np.zeros((N, 3))
        normals = np.zeros((N, 3))
        f_dc = np.zeros((N, 3))
        f_rest = np.zeros((N, 45))
        opacities = np.zeros((N, 1))
        scale = np.zeros((N, 3))
        rotation = np.zeros((N, 4))

        position_attr = mesh.attributes.get("position")
        log_opacity_attr = mesh.attributes.get("log_opacity")
        logscale_attr = mesh.attributes.get("logscale")
        sh0_attr = mesh.attributes.get("sh0")
        sh_attrs = [mesh.attributes.get(f"sh{j+1}") for j in range(15)]
        rot_quatxyz_attr = mesh.attributes.get("quatxyz")
        rot_quatw_attr = mesh.attributes.get("quatw")

        for i, _ in enumerate(mesh.vertices):
            xyz[i] = position_attr.data[i].vector.to_tuple()
            opacities[i] = log_opacity_attr.data[i].value
            scale[i] = logscale_attr.data[i].vector.to_tuple()

            f_dc[i] = sh0_attr.data[i].vector.to_tuple()
            for j in range(15):
                f_rest[i, j:j+45:15] = sh_attrs[j].data[i].vector.to_tuple()
            
            rotxyz_quat = rot_quatxyz_attr.data[i].vector.to_tuple()
            rotw_quat = rot_quatw_attr.data[i].value
            rotation[i] = (*rotxyz_quat, rotw_quat)

            # euler = mathutils.Euler(rot_euler_attr.data[i].vector)
            # quat = euler.to_quaternion()
            # rotation[i] = (quat.x, quat.y, quat.z, quat.w)

        # opacities = np.log(opacities / (1 - opacities))
        # scale = np.log(scale)

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(N, dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(self.filepath)
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = bpy.path.abspath("//point_cloud.ply")

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SwitchRenderType(bpy.types.Operator):
    bl_idname = "object.render_gaussion_switch"
    bl_label = "render_gaussion_switch"
    bl_options = {'REGISTER', 'UNDO'}

    enable_point_render: bpy.props.BoolProperty(name="As points")

    def execute(self, context):
        obj = bpy.data.objects["GaussianSplatting"]
        if self.enable_point_render:
            build_point_geometry_node(obj)
            self.enable_point_render = False
        else:
            mat = bpy.data.materials['GaussianSplatting_bk']
            build_sphere_geometry_node(obj, mat)
            self.enable_point_render = True
        return {'FINISHED'}
    

class GaussianSplattingPanel(bpy.types.Panel):
    
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    bl_idname = "OBJECT_PT_gaussian_splatting"
    bl_category = "3D Gaussian Splatting"
    bl_label = "3D Gaussian Splatting"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        # Import Gaussian Splatting button
        row = layout.row()
        row.operator(ImportGaussianSplatting.bl_idname, text="Import Gaussian Splatting")

        if obj is not None and "gaussian_splatting" in obj:
            row = layout.row()
            prop = row.operator(SwitchRenderType.bl_idname, text="SwitchRender")
            if not prop.enable_point_render:
                row = layout.row()
                row.prop(obj.modifiers["GeometryNodes"].node_group.nodes.get("Random Value").inputs["Probability"], "default_value", text="Display Percentage")

            # Export Gaussian Splatting button
            row = layout.row()
            row.operator(ExportGaussianSplatting.bl_idname, text="Export Gaussian Splatting")

def register():
    bpy.utils.register_class(ImportGaussianSplatting)
    bpy.utils.register_class(GaussianSplattingPanel)
    bpy.utils.register_class(ExportGaussianSplatting)
    bpy.utils.register_class(SwitchRenderType)
    bpy.types.Scene.ply_file_path = bpy.props.StringProperty(name="PLY Filepath", subtype='FILE_PATH')

def unregister():
    bpy.utils.unregister_class(ImportGaussianSplatting)
    bpy.utils.unregister_class(GaussianSplattingPanel)
    bpy.utils.unregister_class(ExportGaussianSplatting)
    bpy.utils.unregister_class(SwitchRenderType)
    del bpy.types.Scene.ply_file_path

if __name__ == "__main__":
    register()
