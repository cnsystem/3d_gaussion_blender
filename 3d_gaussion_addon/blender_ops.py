from .plyfile import PlyData, PlyElement
import bpy
import mathutils
import numpy as np
import time
import random

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def SH2RGB(sh):
    c = sh.reshape(-1, 3) * C0 + 0.5
    return np.concatenate([c, np.ones((c.shape[0],1))], axis=1)


def load_ply(filepath):
    model_ = {}
    
    plydata = PlyData.read(filepath)
    model_['xyz'] = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)

    N = len(model_['xyz'])
    
    log_opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    model_['opacities'] = 1 / (1 + np.exp(-log_opacities))

    features_dc = np.zeros((N, 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    model_["features_dc"] = features_dc

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((N, len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((N, 3, 15))
    model_["features_extra"] = features_extra

    log_scales = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
                        np.asarray(plydata.elements[0]["scale_1"]),
                        np.asarray(plydata.elements[0]["scale_2"])), axis=1)
    model_["scales"] = np.exp(log_scales)
    model_["quats"] = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
                        np.asarray(plydata.elements[0]["rot_1"]),
                        np.asarray(plydata.elements[0]["rot_2"]),
                        np.asarray(plydata.elements[0]["rot_3"])), axis=1)

    rot_euler = np.zeros((N, 3))
    for i in range(N):
        quat = mathutils.Quaternion(model_["quats"][i].tolist())
        euler = quat.to_euler()
        rot_euler[i] = (euler.x, euler.y, euler.z)

    model_["rot_euler"] = rot_euler
    model_["color"] = SH2RGB(features_dc).reshape(N, -1)
    return model_


def build_mesh(model_):
    mesh = bpy.data.meshes.new(name="Mesh")
    mesh.from_pydata(model_['xyz'].tolist(), [], [])
    mesh.update()
    # opacity
    opacity_attr = mesh.attributes.new(name="opacity", type='FLOAT', domain='POINT')
    opacity_attr.data.foreach_set("value", model_["opacities"].flatten())
    # scale
    scale_attr = mesh.attributes.new(name="scale", type='FLOAT_VECTOR', domain='POINT')
    scale_attr.data.foreach_set("vector", model_["scales"].flatten())
    # sh attr
    sh0_attr = mesh.attributes.new(name="sh0", type='FLOAT_VECTOR', domain='POINT')
    sh0_attr.data.foreach_set("vector", model_["features_dc"].flatten())
    for j in range(0, 15):
        sh_attr = mesh.attributes.new(name=f"sh{j+1}", type='FLOAT_VECTOR', domain='POINT')
        sh_attr.data.foreach_set("vector", model_["features_extra"][:, :, j].flatten())
    # quat
    rot_quatw_attr = mesh.attributes.new(name="quat", type='QUATERNION', domain='POINT')
    rot_quatw_attr.data.foreach_set("value", model_["quats"].flatten())
    #euler
    rot_euler_attr = mesh.attributes.new(name="rot_euler", type='FLOAT_VECTOR', domain='POINT')
    rot_euler_attr.data.foreach_set("vector", model_["rot_euler"].flatten())
    #color
    color_attr = mesh.color_attributes.new(name="color", type='FLOAT_COLOR', domain='POINT')
    color_attr.data.foreach_set("color", model_["color"].flatten())
    # obj
    obj = bpy.data.objects.new("GaussianSplatting", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = (-np.pi / 2, 0, 0)
    obj["gaussian_splatting"] = True
    return obj


def build_material_graph():
    mat = bpy.data.materials.new(name="GaussianSplatting_bk")
    mat = bpy.data.materials['GaussianSplatting_bk']
    mat.use_nodes = True
    mat.blend_method = "HASHED"
    mat_tree = mat.node_tree
    # remove nodes
    for node in mat_tree.nodes:
        mat_tree.nodes.remove(node)

    # BSDF
    principled_node = mat_tree.nodes.new('ShaderNodeBsdfPrincipled')
    principled_node.location = (3600, 600)
    principled_node.inputs["Base Color"].default_value = (0, 0, 0, 1)
    principled_node.inputs["Specular IOR Level"].default_value = 0
    principled_node.inputs["Roughness"].default_value = 0

    # sh
    sh_attr_nodes = []
    sh_inst_attr_nodes = []  # ellipsoids
    sh_geom_attr_nodes = []  # point cloud
    for j in range(0, 16):
        # INSTANCER
        sh_inst_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
        sh_inst_attr_node.location = (1800, 400 * j + 200)
        sh_inst_attr_node.attribute_name = f"sh{j}"
        sh_inst_attr_node.attribute_type = 'INSTANCER'
        sh_inst_attr_nodes.append(sh_inst_attr_node)
        # GEOMETRY
        sh_geom_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
        sh_geom_attr_node.location = (1800, 400 * j)
        sh_geom_attr_node.attribute_name = f"sh{j}"
        sh_geom_attr_node.attribute_type = 'GEOMETRY'
        sh_geom_attr_nodes.append(sh_geom_attr_node)
        # link: INSTANCER + GEOMETRY
        vector_math_node = mat_tree.nodes.new('ShaderNodeVectorMath')
        vector_math_node.operation = 'ADD'
        vector_math_node.location = (2000, 400 * j)
        mat_tree.links.new(sh_inst_attr_node.outputs["Vector"], vector_math_node.inputs[0])
        mat_tree.links.new(sh_geom_attr_node.outputs["Vector"], vector_math_node.inputs[1])
        sh_attr_nodes.append(vector_math_node)
    # output
    sh = [sh_attr_node.outputs["Vector"] for sh_attr_node in sh_attr_nodes]
    # position
    position_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
    position_attr_node.attribute_name = "position"
    position_attr_node.attribute_type = 'GEOMETRY'
    position_attr_node.location = (0, 0)
    # opacity GEOMETRY
    opacity_geom_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
    opacity_geom_attr_node.location = (2800, -200)
    opacity_geom_attr_node.attribute_name = "opacity"
    opacity_geom_attr_node.attribute_type = 'GEOMETRY'
    # opacity INSTANCER
    opacity_inst_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
    opacity_inst_attr_node.location = (2800, -400)
    opacity_inst_attr_node.attribute_name = "opacity"
    opacity_inst_attr_node.attribute_type = 'INSTANCER'
    # link: INSTANCER + GEOMETRY
    opacity_attr_node = mat_tree.nodes.new('ShaderNodeMath')
    opacity_attr_node.operation = 'ADD'
    opacity_attr_node.location = (3200, -200)
    mat_tree.links.new(opacity_geom_attr_node.outputs["Fac"], opacity_attr_node.inputs[0])
    mat_tree.links.new(opacity_inst_attr_node.outputs["Fac"], opacity_attr_node.inputs[1])

    # Material output
    output_node = mat_tree.nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (3600, 0)
    # Camera location
    combine_xyz_node = mat_tree.nodes.new('ShaderNodeCombineXYZ')
    combine_xyz_node.location = (-200, 200)
    vector_transform_node = mat_tree.nodes.new('ShaderNodeVectorTransform')
    vector_transform_node.vector_type = 'POINT'
    vector_transform_node.convert_from = 'CAMERA'
    vector_transform_node.convert_to = 'WORLD'
    vector_transform_node.location = (0, 200)
    mat_tree.links.new(combine_xyz_node.outputs["Vector"], vector_transform_node.inputs["Vector"])
    # View direction
    dir_node = mat_tree.nodes.new('ShaderNodeVectorMath')
    dir_node.operation = 'SUBTRACT'
    dir_node.location = (200, 200)
    normalize_node = mat_tree.nodes.new('ShaderNodeVectorMath')
    normalize_node.operation = 'NORMALIZE'
    normalize_node.location = (400, 200)
    mat_tree.links.new(position_attr_node.outputs["Vector"], dir_node.inputs[0])
    mat_tree.links.new(vector_transform_node.outputs["Vector"], dir_node.inputs[1])
    mat_tree.links.new(dir_node.outputs["Vector"], normalize_node.inputs[0])
    # Coordinate system transform (x -> -y, y -> -z, z -> x)  TODO: REMOVE
    separate_xyz_node = mat_tree.nodes.new('ShaderNodeSeparateXYZ')
    separate_xyz_node.location = (600, 200)
    mat_tree.links.new(normalize_node.outputs["Vector"], separate_xyz_node.inputs["Vector"])
    combine_xyz_node = mat_tree.nodes.new('ShaderNodeCombineXYZ')
    combine_xyz_node.location = (800, 200)
    mat_tree.links.new(separate_xyz_node.outputs["X"], combine_xyz_node.inputs["Y"])
    mat_tree.links.new(separate_xyz_node.outputs["Y"], combine_xyz_node.inputs["Z"])
    mat_tree.links.new(separate_xyz_node.outputs["Z"], combine_xyz_node.inputs["X"])
    multiply_node = mat_tree.nodes.new('ShaderNodeVectorMath')
    multiply_node.operation = 'MULTIPLY'
    multiply_node.inputs[1].default_value = (-1, -1, 1)
    multiply_node.location = (1000, 200)
    mat_tree.links.new(combine_xyz_node.outputs["Vector"], multiply_node.inputs[0])

    # seperate
    separate_xyz_node = mat_tree.nodes.new('ShaderNodeSeparateXYZ')
    separate_xyz_node.location = (1200, 200)
    mat_tree.links.new(multiply_node.outputs["Vector"], separate_xyz_node.inputs["Vector"])
    x = separate_xyz_node.outputs["X"]
    y = separate_xyz_node.outputs["Y"]
    z = separate_xyz_node.outputs["Z"]
    # compute xx, yy, zz, xy, xz, yz
    xx_node = mat_tree.nodes.new('ShaderNodeMath')
    xx_node.operation = 'MULTIPLY'
    xx_node.location = (1400, 200)
    mat_tree.links.new(x, xx_node.inputs[0])
    mat_tree.links.new(x, xx_node.inputs[1])
    xx = xx_node.outputs["Value"]
    # yy
    yy_node = mat_tree.nodes.new('ShaderNodeMath')
    yy_node.operation = 'MULTIPLY'
    yy_node.location = (1400, 400)
    mat_tree.links.new(y, yy_node.inputs[0])
    mat_tree.links.new(y, yy_node.inputs[1])
    yy = yy_node.outputs["Value"]
    # zz
    zz_node = mat_tree.nodes.new('ShaderNodeMath')
    zz_node.operation = 'MULTIPLY'
    zz_node.location = (1400, 600)
    mat_tree.links.new(z, zz_node.inputs[0])
    mat_tree.links.new(z, zz_node.inputs[1])
    zz = zz_node.outputs["Value"]
    # y
    xy_node = mat_tree.nodes.new('ShaderNodeMath')
    xy_node.operation = 'MULTIPLY'
    xy_node.location = (1600, 200)
    mat_tree.links.new(x, xy_node.inputs[0])
    mat_tree.links.new(y, xy_node.inputs[1])
    xy = xy_node.outputs["Value"]
    # yz
    yz_node = mat_tree.nodes.new('ShaderNodeMath')
    yz_node.operation = 'MULTIPLY'
    yz_node.location = (1600, 400)
    mat_tree.links.new(x, yz_node.inputs[0])
    mat_tree.links.new(y, yz_node.inputs[1])
    yz = yz_node.outputs["Value"]
    # xz
    xz_node = mat_tree.nodes.new('ShaderNodeMath')
    xz_node.operation = 'MULTIPLY'
    xz_node.location = (1600, 600)
    mat_tree.links.new(x, xz_node.inputs[0])
    mat_tree.links.new(y, xz_node.inputs[1])
    xz = xz_node.outputs["Value"]
    # SH 0
    scale_node_0 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_0.operation = 'SCALE'
    scale_node_0.inputs["Scale"].default_value = C0
    scale_node_0.location = (2400, 200)
    mat_tree.links.new(sh[0], scale_node_0.inputs[0])
    # SH 1
    math_node = mat_tree.nodes.new('ShaderNodeMath')
    math_node.operation = 'MULTIPLY'
    math_node.inputs[1].default_value = -C1
    math_node.location = (2200, 400)
    mat_tree.links.new(y, math_node.inputs[0])
    scale_node_1 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_1.operation = 'SCALE'
    scale_node_1.location = (2400, 400)
    mat_tree.links.new(sh[1], scale_node_1.inputs[0])
    mat_tree.links.new(math_node.outputs["Value"], scale_node_1.inputs["Scale"])
    # SH 2
    math_node = mat_tree.nodes.new('ShaderNodeMath')
    math_node.operation = 'MULTIPLY'
    math_node.inputs[1].default_value = C1
    math_node.location = (2200, 600)
    mat_tree.links.new(z, math_node.inputs[0])
    scale_node_2 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_2.operation = 'SCALE'
    scale_node_2.location = (2400, 600)
    mat_tree.links.new(sh[2], scale_node_2.inputs[0])
    mat_tree.links.new(math_node.outputs["Value"], scale_node_2.inputs["Scale"])
    # SH 3
    math_node = mat_tree.nodes.new('ShaderNodeMath')
    math_node.operation = 'MULTIPLY'
    math_node.inputs[1].default_value = -C1
    math_node.location = (2200, 800)
    mat_tree.links.new(x, math_node.inputs[0])
    scale_node_3 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_3.operation = 'SCALE'
    scale_node_3.location = (2400, 800)
    mat_tree.links.new(sh[3], scale_node_3.inputs[0])
    mat_tree.links.new(math_node.outputs["Value"], scale_node_3.inputs["Scale"])
    # SH 4
    math_node = mat_tree.nodes.new('ShaderNodeMath')
    math_node.operation = 'MULTIPLY'
    math_node.inputs[1].default_value = C2[0]
    math_node.location = (2200, 1000)
    mat_tree.links.new(xy, math_node.inputs[0])
    scale_node_4 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_4.operation = 'SCALE'
    scale_node_4.location = (2400, 1000)
    mat_tree.links.new(sh[4], scale_node_4.inputs[0])
    mat_tree.links.new(math_node.outputs["Value"], scale_node_4.inputs["Scale"])
    # SH 5
    math_node = mat_tree.nodes.new('ShaderNodeMath')
    math_node.operation = 'MULTIPLY'
    math_node.inputs[1].default_value = C2[1]
    math_node.location = (2200, 1200)
    mat_tree.links.new(yz, math_node.inputs[0])
    scale_node_5 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_5.operation = 'SCALE'
    scale_node_5.location = (2400, 1200)
    mat_tree.links.new(sh[5], scale_node_5.inputs[0])
    mat_tree.links.new(math_node.outputs["Value"], scale_node_5.inputs["Scale"])
    # SH 6
    math_node1 = mat_tree.nodes.new('ShaderNodeMath')
    math_node1.operation = 'MULTIPLY'
    math_node1.inputs[1].default_value = C2[2]
    math_node1.location = (2200, 1400)
    mat_tree.links.new(zz, math_node1.inputs[0])
    math_node2 = mat_tree.nodes.new('ShaderNodeMath')
    math_node2.operation = 'SUBTRACT'
    math_node2.location = (2200, 1400)
    mat_tree.links.new(math_node1.outputs["Value"], math_node2.inputs[0])
    mat_tree.links.new(xx, math_node2.inputs[1])
    math_node3 = mat_tree.nodes.new('ShaderNodeMath')
    math_node3.operation = 'SUBTRACT'
    math_node3.location = (2200, 1400)
    mat_tree.links.new(math_node2.outputs["Value"], math_node3.inputs[0])
    mat_tree.links.new(yy, math_node3.inputs[1])
    math_node4 = mat_tree.nodes.new('ShaderNodeMath')
    math_node4.operation = 'MULTIPLY'
    math_node4.inputs[1].default_value = C2[1]
    math_node4.location = (2200, 1400)
    mat_tree.links.new(math_node3.outputs["Value"], math_node4.inputs[0])
    scale_node_6 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_6.operation = 'SCALE'
    scale_node_6.location = (2400, 1400)
    mat_tree.links.new(sh[6], scale_node_6.inputs[0])
    mat_tree.links.new(math_node4.outputs["Value"], scale_node_6.inputs["Scale"])
    # SH 7
    math_node = mat_tree.nodes.new('ShaderNodeMath')
    math_node.operation = 'MULTIPLY'
    math_node.inputs[1].default_value = C2[3]
    math_node.location = (2200, 1600)
    mat_tree.links.new(xz, math_node.inputs[0])
    scale_node_7 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_7.operation = 'SCALE'
    scale_node_7.location = (2400, 1600)
    mat_tree.links.new(sh[7], scale_node_7.inputs[0])
    mat_tree.links.new(math_node.outputs["Value"], scale_node_7.inputs["Scale"])
    # SH 8
    math_node1 = mat_tree.nodes.new('ShaderNodeMath')
    math_node1.operation = 'SUBTRACT'
    math_node1.location = (2200, 1800)
    mat_tree.links.new(xx, math_node1.inputs[0])
    mat_tree.links.new(yy, math_node1.inputs[1])
    math_node2 = mat_tree.nodes.new('ShaderNodeMath')
    math_node2.operation = 'MULTIPLY'
    math_node2.inputs[1].default_value = C2[4]
    math_node2.location = (2200, 1800)
    mat_tree.links.new(math_node1.outputs["Value"], math_node2.inputs[0])
    scale_node_8 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_8.operation = 'SCALE'
    scale_node_8.location = (2400, 1800)
    mat_tree.links.new(sh[8], scale_node_8.inputs[0])
    mat_tree.links.new(math_node2.outputs["Value"], scale_node_8.inputs["Scale"])
    # SH 9
    scale_node_9 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_9.operation = 'SCALE'
    scale_node_9.location = (2400, 2000)
    # SH 10
    scale_node_10 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_10.operation = 'SCALE'
    scale_node_10.location = (2400, 2200)
    # SH 11
    scale_node_11 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_11.operation = 'SCALE'
    scale_node_11.location = (2400, 2400)
    # SH 12
    scale_node_12 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_12.operation = 'SCALE'
    scale_node_12.location = (2400, 2600)
    # SH 13
    scale_node_13 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_13.operation = 'SCALE'
    scale_node_13.location = (2400, 2800)
    # SH 14
    scale_node_14 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_14.operation = 'SCALE'
    scale_node_14.location = (2400, 3000)
    # SH 15
    scale_node_15 = mat_tree.nodes.new('ShaderNodeVectorMath')
    scale_node_15.operation = 'SCALE'
    scale_node_15.location = (2400, 3200)
    # Result
    res_nodes = [
        scale_node_0, scale_node_1, scale_node_2, scale_node_3, scale_node_4, scale_node_5, scale_node_6, scale_node_7,
        scale_node_8, scale_node_9, scale_node_10, scale_node_11, scale_node_12, scale_node_13, scale_node_14, scale_node_15
    ]

    add_node = mat_tree.nodes.new('ShaderNodeVectorMath')
    add_node.operation = 'ADD'
    add_node.location = (2600, 200)
    mat_tree.links.new(res_nodes[0].outputs["Vector"], add_node.inputs[0])
    mat_tree.links.new(res_nodes[1].outputs["Vector"], add_node.inputs[1])

    for i in range(2, 16):
        new_add_node = mat_tree.nodes.new('ShaderNodeVectorMath')
        new_add_node.operation = 'ADD'
        new_add_node.location = (2600, 200 + i * 200)
        mat_tree.links.new(res_nodes[i].outputs["Vector"], new_add_node.inputs[0])
        mat_tree.links.new(add_node.outputs["Vector"], new_add_node.inputs[1])
        add_node = new_add_node

    math_node = mat_tree.nodes.new('ShaderNodeVectorMath')
    math_node.operation = 'ADD'
    math_node.inputs[1].default_value = (0.5, 0.5, 0.5)
    math_node.location = (2800, 200)
    mat_tree.links.new(add_node.outputs["Vector"], math_node.inputs[0])

    gamma_node = mat_tree.nodes.new('ShaderNodeGamma')
    gamma_node.inputs["Gamma"].default_value = 2.2
    gamma_node.location = (3000, 200)
    mat_tree.links.new(math_node.outputs["Vector"], gamma_node.inputs["Color"])
    mat_tree.links.new(gamma_node.outputs["Color"], principled_node.inputs["Emission Color"])

    geometry_node = mat_tree.nodes.new('ShaderNodeNewGeometry')
    geometry_node.location = (2600, 0)
    vector_math_node = mat_tree.nodes.new('ShaderNodeVectorMath')
    vector_math_node.operation = 'DOT_PRODUCT'
    vector_math_node.location = (2800, 0)
    mat_tree.links.new(geometry_node.outputs["Normal"], vector_math_node.inputs[0])
    mat_tree.links.new(geometry_node.outputs["Incoming"], vector_math_node.inputs[1])

    math_node = mat_tree.nodes.new('ShaderNodeMath')
    math_node.operation = 'MULTIPLY'
    math_node.location = (3000, 0)
    mat_tree.links.new(opacity_attr_node.outputs["Value"], math_node.inputs[0])
    mat_tree.links.new(vector_math_node.outputs["Value"], math_node.inputs[1])
    mat_tree.links.new(math_node.outputs["Value"], principled_node.inputs["Alpha"])
    # Output
    mat_tree.links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])
    return mat
    

def build_basic_geometry_node(obj):
    obj.modifiers.clear()
    geo_node_mod = obj.modifiers.new(name="GeometryNodes", type='NODES')
    geo_tree = bpy.data.node_groups.new(name="GaussianSplatting", type='GeometryNodeTree')
    geo_node_mod.node_group = geo_tree
    for node in geo_tree.nodes:
        geo_tree.nodes.remove(node)
    
    geo_tree.interface.new_socket('Geometry', description="", in_out='INPUT', socket_type='NodeSocketGeometry')
    geo_tree.interface.new_socket('Geometry', description="", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    return geo_tree


def build_point_geometry_node(obj):
    geo_tree = build_basic_geometry_node(obj)
    group_input_node = geo_tree.nodes.new('NodeGroupInput')
    group_input_node.location = (0, 0)
    group_output_node = geo_tree.nodes.new('NodeGroupOutput')
    group_output_node.location = (600, 0)
    # mesh to points
    mesh_to_points_node = geo_tree.nodes.new('GeometryNodeMeshToPoints')
    mesh_to_points_node.location = (200, 0)
    mesh_to_points_node.inputs["Radius"].default_value = 0.01
    # percent to vis
    set_point_radius_node = geo_tree.nodes.new('GeometryNodeSetPointRadius')
    set_point_radius_node.location = (400, 0)
    # Scale
    scale_attr = geo_tree.nodes.new('GeometryNodeInputNamedAttribute')
    scale_attr.location = (200, -200)
    scale_attr.data_type = 'FLOAT_VECTOR'
    scale_attr.inputs["Name"].default_value = "scale"

    geo_tree.links.new(group_input_node.outputs["Geometry"], mesh_to_points_node.inputs["Mesh"])
    geo_tree.links.new(mesh_to_points_node.outputs["Points"], set_point_radius_node.inputs["Points"])
    geo_tree.links.new(scale_attr.outputs["Attribute"], set_point_radius_node.inputs["Radius"])
    geo_tree.links.new(set_point_radius_node.outputs["Points"], group_output_node.inputs["Geometry"])
    

def build_sphere_geometry_node(obj, mat):
    geo_tree = build_basic_geometry_node(obj)
    group_input_node = geo_tree.nodes.new('NodeGroupInput')
    group_input_node.location = (0, 400)
    group_output_node = geo_tree.nodes.new('NodeGroupOutput')
    group_output_node.location = (1200, 0)
    # percent
    random_value_node = geo_tree.nodes.new('FunctionNodeRandomValue')
    random_value_node.location = (0, 200)
    random_value_node.data_type = 'BOOLEAN'
    random_value_node.inputs["Probability"].default_value = 0.1
    # Scale
    scale_attr = geo_tree.nodes.new('GeometryNodeInputNamedAttribute')
    scale_attr.location = (200, -400)
    scale_attr.data_type = 'FLOAT_VECTOR'
    scale_attr.inputs["Name"].default_value = "scale"
    # euler
    rot_euler_attr = geo_tree.nodes.new('GeometryNodeInputNamedAttribute')
    rot_euler_attr.location = (200, -200)
    rot_euler_attr.data_type = 'FLOAT_VECTOR'
    rot_euler_attr.inputs["Name"].default_value = "rot_euler"
    # mesh to points
    mesh_to_points_node = geo_tree.nodes.new('GeometryNodeMeshToPoints')
    mesh_to_points_node.location = (200, 200)
    mesh_to_points_node.inputs["Radius"].default_value = 0.01
    # sphere
    ico_node = geo_tree.nodes.new('GeometryNodeMeshIcoSphere')
    ico_node.location = (200, 0)
    ico_node.inputs["Subdivisions"].default_value = 1
    ico_node.inputs["Radius"].default_value = 1
    set_shade_smooth_node = geo_tree.nodes.new('GeometryNodeSetShadeSmooth')
    set_shade_smooth_node.location = (400, 0)
    instance_node = geo_tree.nodes.new('GeometryNodeInstanceOnPoints')
    instance_node.location = (600, 0)
    # material
    set_material_node = geo_tree.nodes.new('GeometryNodeSetMaterial')
    set_material_node.location = (800, 0)
    set_material_node.inputs["Material"].default_value = mat
    realize_instances_node = geo_tree.nodes.new('GeometryNodeRealizeInstances')
    realize_instances_node.location = (1000, 0)

    geo_tree.links.new(group_input_node.outputs["Geometry"], mesh_to_points_node.inputs["Mesh"])
    geo_tree.links.new(random_value_node.outputs[3], mesh_to_points_node.inputs["Selection"])
    geo_tree.links.new(mesh_to_points_node.outputs["Points"], instance_node.inputs["Points"])
    geo_tree.links.new(ico_node.outputs["Mesh"], set_shade_smooth_node.inputs["Geometry"])
    geo_tree.links.new(set_shade_smooth_node.outputs["Geometry"], instance_node.inputs["Instance"])
    geo_tree.links.new(rot_euler_attr.outputs["Attribute"], instance_node.inputs["Rotation"])
    geo_tree.links.new(scale_attr.outputs["Attribute"], instance_node.inputs["Scale"])
    geo_tree.links.new(instance_node.outputs["Instances"], set_material_node.inputs["Geometry"])
    geo_tree.links.new(set_material_node.outputs["Geometry"], realize_instances_node.inputs["Geometry"])
    geo_tree.links.new(realize_instances_node.outputs["Geometry"], group_output_node.inputs["Geometry"])

    # quats
    quat_attr = geo_tree.nodes.new('GeometryNodeInputNamedAttribute')
    quat_attr.location = (-200, 0)
    quat_attr.data_type = 'QUATERNION'
    quat_attr.inputs["Name"].default_value = "quats"