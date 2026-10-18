"""Blender-side helpers: version shims, scene construction, materials, render setup.

Tested on Blender 4.5 LTS and 5.2 LTS (macOS arm64, Metal). Version-specific API
points are isolated in the `compat_*` functions below:
  * EEVEE engine id: 'BLENDER_EEVEE_NEXT' (4.2-4.5) vs 'BLENDER_EEVEE' (<4.2 and 5.x).
  * Multilayer EXR: file_format 'OPEN_EXR_MULTILAYER' (<=4.5) vs
    media_type 'MULTI_LAYER_IMAGE' + file_format 'OPEN_EXR' (5.x).
  * Material.use_nodes is deprecated in 5.x (always node based).
  * Smooth shading: Mesh.shade_smooth() (4.1+).
We deliberately avoid the compositor (its Python API changed completely in 5.0)
and Geometry Nodes; all passes come from the multilayer EXR written by the render.
"""
from __future__ import annotations

import json
import os
import sys

import bpy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SEMANTIC = {"background": 0, "leaf": 1, "wood": 2, "pot": 3, "ground": 4, "room": 5}
AOVS = [("leaf_id", "VALUE"), ("semantic", "VALUE"), ("P", "COLOR"), ("N", "COLOR"), ("P_next", "COLOR"), ("P_rest", "COLOR")]


def script_args():
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def log(event, **kw):
    print("[synthtree] " + json.dumps({"event": event, **kw}), flush=True)


# --------------------------------------------------------------------------- compat
def compat_eevee_id():
    items = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    return "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in items else "BLENDER_EEVEE"


def compat_set_exr(image_settings, multilayer: bool, depth: str):
    if hasattr(image_settings, "media_type"):  # Blender 5.x: the format enum depends on media_type
        image_settings.media_type = "MULTI_LAYER_IMAGE" if multilayer else "IMAGE"
        items = {e.identifier for e in image_settings.bl_rna.properties["file_format"].enum_items}
        image_settings.file_format = "OPEN_EXR_MULTILAYER" if multilayer and "OPEN_EXR_MULTILAYER" in items else "OPEN_EXR"
    else:
        image_settings.file_format = "OPEN_EXR_MULTILAYER" if multilayer else "OPEN_EXR"
    image_settings.color_depth = depth  # '32' is mandatory for ids/positions (half breaks ids > 2048)
    image_settings.exr_codec = "ZIP"
    if not multilayer:
        image_settings.color_mode = "RGB"


def compat_new_material(name):
    m = bpy.data.materials.new(name)
    if bpy.app.version < (5, 0, 0):
        m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    return m, nt


def _set_input(node, names, value):
    for nm in names if isinstance(names, (list, tuple)) else [names]:
        if nm in node.inputs:
            node.inputs[nm].default_value = value
            return True
    return False


# --------------------------------------------------------------------------- meshes
def mesh_object(name, verts, faces, collection=None, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts, np.float64).tolist(), [], np.asarray(faces, np.int64).tolist())
    me.update()
    assert len(me.polygons) == len(faces), "face count changed on import"
    if smooth:
        if hasattr(me, "shade_smooth"):
            me.shade_smooth()
        else:
            me.polygons.foreach_set("use_smooth", np.ones(len(faces), bool))
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def set_attr(me, name, kind, domain, values):
    """kind: FLOAT | INT | FLOAT_VECTOR | FLOAT_COLOR."""
    a = me.attributes.get(name) or me.attributes.new(name, kind, domain)
    key = {"FLOAT": "value", "INT": "value", "FLOAT_VECTOR": "vector", "FLOAT_COLOR": "color"}[kind]
    dt = np.int32 if kind == "INT" else np.float32
    a.data.foreach_set(key, np.ascontiguousarray(values, dt).ravel())


def set_verts(me, verts):
    me.vertices.foreach_set("co", np.ascontiguousarray(verts, np.float32).ravel())


# --------------------------------------------------------------------------- materials
def _aov(nt, name, socket, value=None):
    n = nt.nodes.new("ShaderNodeOutputAOV")
    n.aov_name = name
    target = n.inputs["Color"] if socket in ("color", "vector") else n.inputs["Value"]
    if value is not None:
        target.default_value = value
    return n, target


def _common_aovs(nt, semantic, leaf_id_socket=None, p_next=None, p_rest=None):
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    _, s = _aov(nt, "semantic", "value", float(semantic))
    _, s = _aov(nt, "leaf_id", "value", 0.0)
    if leaf_id_socket is not None:
        nt.links.new(leaf_id_socket, s)
    _, s = _aov(nt, "P", "color")
    nt.links.new(geo.outputs["Position"], s)
    _, s = _aov(nt, "N", "color")
    nt.links.new(geo.outputs["Normal"], s)
    _, s = _aov(nt, "P_next", "color")
    nt.links.new(p_next if p_next is not None else geo.outputs["Position"], s)
    _, s = _aov(nt, "P_rest", "color")
    nt.links.new(p_rest if p_rest is not None else geo.outputs["Position"], s)
    return geo


def _attr(nt, name):
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_type = "GEOMETRY"
    a.attribute_name = name
    return a


def leaf_material(translucency=0.25):
    m, nt = compat_new_material("leaf")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    front, back, lid = _attr(nt, "leaf_color_front"), _attr(nt, "leaf_color_back"), _attr(nt, "leaf_id_f")
    pn, pr = _attr(nt, "P_next"), _attr(nt, "P_rest")
    geo = _common_aovs(nt, SEMANTIC["leaf"], lid.outputs["Fac"], pn.outputs["Vector"], pr.outputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    sock = {s.identifier: s for s in mix.inputs}
    nt.links.new(geo.outputs["Backfacing"], sock["Factor_Float"])
    nt.links.new(front.outputs["Color"], sock["A_Color"])
    nt.links.new(back.outputs["Color"], sock["B_Color"])
    col = next(s for s in mix.outputs if s.identifier == "Result_Color")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(col, bsdf.inputs["Base Color"])
    _set_input(bsdf, "Roughness", 0.45)
    _set_input(bsdf, ["Specular IOR Level", "Specular"], 0.35)
    tr = nt.nodes.new("ShaderNodeBsdfTranslucent")
    nt.links.new(col, tr.inputs["Color"])
    ms = nt.nodes.new("ShaderNodeMixShader")
    ms.inputs["Fac"].default_value = translucency
    nt.links.new(bsdf.outputs[0], ms.inputs[1])
    nt.links.new(tr.outputs[0], ms.inputs[2])
    nt.links.new(ms.outputs[0], out.inputs["Surface"])
    return m


def textured_material(name, semantic, rgb, rgb2, scale, roughness=0.8, deforming=False):
    """Principled BSDF with a procedural noise texture.

    On deforming meshes the texture is driven by the rest-position attribute
    P_rest, so it sticks to the bark instead of 'swimming' through it.
    """
    m, nt = compat_new_material(name)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    pn = _attr(nt, "P_next").outputs["Vector"] if deforming else None
    pr = _attr(nt, "P_rest").outputs["Vector"] if deforming else None
    geo = _common_aovs(nt, semantic, None, pn, pr)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    _set_input(noise, "Detail", 6.0)
    nt.links.new(pr if deforming else geo.outputs["Position"], noise.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    sock = {s.identifier: s for s in mix.inputs}
    nt.links.new(noise.outputs["Fac"], sock["Factor_Float"])
    sock["A_Color"].default_value = (*rgb, 1.0)
    sock["B_Color"].default_value = (*rgb2, 1.0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(next(s for s in mix.outputs if s.identifier == "Result_Color"), bsdf.inputs["Base Color"])
    _set_input(bsdf, "Roughness", roughness)
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return m


# --------------------------------------------------------------------------- scene
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    for a in list(scene.view_layers[0].aovs):
        scene.view_layers[0].aovs.remove(a)
    vl = scene.view_layers[0]
    for name, typ in AOVS:
        a = vl.aovs.add()
        a.name, a.type = name, typ
    return scene


class SceneHandles:
    pass


def build_scene(tree: dict, cfg: dict, room_half_size: float = 3.0) -> SceneHandles:
    """Create objects, materials, lights and camera for one tree.

    room_half_size must exceed the camera orbit radius (callers pass radius + margin).
    """
    scene = reset_scene()
    h = SceneHandles()
    rcfg = cfg["render"]

    # wood (deforming)
    h.wood = mesh_object("Wood", tree["wood_verts"], tree["wood_faces"])
    set_attr(h.wood.data, "P_rest", "FLOAT_VECTOR", "POINT", tree["wood_verts"])
    set_attr(h.wood.data, "P_next", "FLOAT_VECTOR", "POINT", tree["wood_verts"])
    set_attr(h.wood.data, "branch_id", "INT", "FACE", tree["wood_face_branch"])
    bark = np.asarray(tree["bark_rgb"], float)
    h.wood.data.materials.append(textured_material("bark", SEMANTIC["wood"], bark, bark * 0.55, 180.0, 0.85, deforming=True))

    # leaves (deforming, rigid per leaf)
    h.leaves = mesh_object("Leaves", tree["leaf_verts"], tree["leaf_faces"])
    me = h.leaves.data
    fl = tree["leaf_face_leaf"]
    set_attr(me, "leaf_id", "INT", "FACE", tree["leaf_id"][fl])
    set_attr(me, "leaf_id_f", "FLOAT", "FACE", tree["leaf_id"][fl].astype(np.float32))  # exact up to 2^24
    rgba = lambda c: np.concatenate([c, np.ones((len(c), 1))], 1)
    set_attr(me, "leaf_color_front", "FLOAT_COLOR", "FACE", rgba(tree["leaf_color_front"][fl]))
    set_attr(me, "leaf_color_back", "FLOAT_COLOR", "FACE", rgba(tree["leaf_color_back"][fl]))
    set_attr(me, "P_rest", "FLOAT_VECTOR", "POINT", tree["leaf_verts"])
    set_attr(me, "P_next", "FLOAT_VECTOR", "POINT", tree["leaf_verts"])
    me.materials.append(leaf_material())

    # pot, soil, ground (static)
    pot = mesh_object("Pot", tree["pot_verts"], tree["pot_faces"], smooth=True)
    pot.data.materials.append(textured_material("terracotta", SEMANTIC["pot"], (0.45, 0.16, 0.07), (0.30, 0.10, 0.05), 40.0, 0.7))
    soil = mesh_object("Soil", tree["soil_verts"], tree["soil_faces"], smooth=False)
    soil.data.materials.append(textured_material("soil", SEMANTIC["pot"], (0.06, 0.04, 0.025), (0.12, 0.08, 0.05), 300.0, 0.95))
    ground = mesh_object("Ground", tree["ground_verts"], tree["ground_faces"], smooth=False)
    ground.data.materials.append(textured_material("ground", SEMANTIC["ground"], (0.32, 0.30, 0.27), (0.18, 0.17, 0.15), 12.0, 0.9))

    bg = rcfg.get("background", {})
    if bg.get("type") == "room":
        from synthtree.meshing import room_mesh
        rv, rf = room_mesh(half_size=max(room_half_size, bg.get("room_half_size_m", 0.0)), height=bg.get("room_height_m", 3.0))
        room = mesh_object("Room", rv, rf, smooth=False)
        room.data.materials.append(textured_material("wall", SEMANTIC["room"], (0.7, 0.68, 0.64), (0.6, 0.58, 0.55), 3.0, 0.9))
        # a closed room blocks sun and sky: light it from the ceiling instead
        half = max(room_half_size, bg.get("room_half_size_m", 0.0))
        ld = bpy.data.lights.new("RoomLight", "AREA")
        ld.shape = "SQUARE"
        ld.size = 1.2 * half
        ld.energy = bg.get("room_light_w", 300.0)
        lo = bpy.data.objects.new("RoomLight", ld)
        lo.location = (0.0, 0.0, bg.get("room_height_m", 3.0) - 0.05)  # area lights emit along -Z
        scene.collection.objects.link(lo)

    # world
    world = bpy.data.worlds.new("World")
    scene.world = world
    if bpy.app.version < (5, 0, 0):
        world.use_nodes = True
    wnt = world.node_tree
    bgn = next(n for n in wnt.nodes if n.type == "BACKGROUND")
    bgn.inputs["Strength"].default_value = bg.get("strength", 1.0)
    if bg.get("type") == "hdri" and bg.get("hdri_path"):
        env = wnt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(bg["hdri_path"])
        wnt.links.new(env.outputs["Color"], bgn.inputs["Color"])
    else:
        bgn.inputs["Color"].default_value = (*bg.get("color", [0.55, 0.6, 0.65]), 1.0)

    # sun
    s = rcfg.get("sun", {})
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = s.get("strength", 3.0)
    sun_data.angle = np.radians(s.get("angle_deg", 3.0))
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    el, az = np.radians(s.get("elevation_deg", 50)), np.radians(s.get("azimuth_deg", 30))
    sun.rotation_mode = "XYZ"
    sun.rotation_euler = (np.pi / 2 - el, 0.0, az + np.pi / 2)

    # camera
    cam_data = bpy.data.cameras.new("Camera")
    h.cam = bpy.data.objects.new("Camera", cam_data)
    h.cam.rotation_mode = "QUATERNION"
    scene.collection.objects.link(h.cam)
    scene.camera = h.cam
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0

    # colour management: Standard view transform, no look, no dither
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = rcfg.get("exposure", 0.0)
    scene.view_settings.gamma = 1.0
    scene.render.dither_intensity = 0.0
    h.scene = scene
    return h


def set_intrinsics(scene, cam_obj, intr):
    from synthtree.intrinsics import to_blender
    p = to_blender(intr)
    cam = cam_obj.data
    cam.type = "PERSP"
    cam.sensor_fit = p["sensor_fit"]
    cam.sensor_width = p["sensor_width"]
    cam.lens_unit = "MILLIMETERS"
    cam.lens = p["lens"]
    cam.shift_x, cam.shift_y = p["shift_x"], p["shift_y"]
    r = scene.render
    r.resolution_x, r.resolution_y = p["resolution_x"], p["resolution_y"]
    r.resolution_percentage = 100
    r.pixel_aspect_x, r.pixel_aspect_y = p["pixel_aspect_x"], p["pixel_aspect_y"]
    return p


def set_camera_pose(cam_obj, c2w_blender):
    from mathutils import Matrix
    cam_obj.matrix_world = Matrix(np.asarray(c2w_blender, float).tolist())


def read_camera_pose(cam_obj):
    bpy.context.view_layer.update()
    return np.array(cam_obj.matrix_world, dtype=np.float64)


# --------------------------------------------------------------------------- render setup
def enable_gpu(scene, prefer="auto"):
    """Select a Cycles GPU backend if available. Returns the device string used."""
    if prefer == "cpu":
        scene.cycles.device = "CPU"
        return "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        scene.cycles.device = "CPU"
        return "CPU"
    for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            return f"GPU:{backend}:" + ",".join(d.name for d in gpus)
    scene.cycles.device = "CPU"
    return "CPU"


def configure_data_pass(scene, device="auto"):
    """Cycles, 1 sample, ~zero-width box filter: every pass value is the exact
    first hit of the ray through the pixel centre (no edge mixing of depth/ids)."""
    r = scene.render
    r.engine = "CYCLES"
    dev = enable_gpu(scene, device)
    c = scene.cycles
    c.samples = 1
    c.use_adaptive_sampling = False
    c.use_denoising = False
    c.pixel_filter_type = "BOX"
    c.filter_width = 0.01
    c.max_bounces = 0
    c.seed = 0
    c.use_animated_seed = False
    r.film_transparent = True
    r.use_motion_blur = False
    r.use_border = False
    vl = scene.view_layers[0]
    vl.use_pass_combined = True
    vl.use_pass_z = True
    vl.use_pass_normal = False
    vl.use_pass_vector = False  # wrong for script-driven deformation; flow comes from P_next
    compat_set_exr(r.image_settings, multilayer=True, depth="32")
    r.use_file_extension = False
    return dev


def configure_beauty_pass(scene, rcfg):
    r = scene.render
    eng = rcfg.get("beauty_engine", "eevee")
    dev = "EEVEE"
    if eng == "cycles":
        r.engine = "CYCLES"
        dev = enable_gpu(scene, rcfg.get("device", "auto"))
        c = scene.cycles
        cc = rcfg.get("cycles", {})
        c.samples = cc.get("samples", 64)
        c.use_adaptive_sampling = True
        c.use_denoising = bool(cc.get("denoise", True))
        c.pixel_filter_type = "BLACKMAN_HARRIS"
        c.filter_width = 1.5
        c.max_bounces = cc.get("max_bounces", 6)
        c.seed = 0
        c.use_animated_seed = False
    else:
        r.engine = compat_eevee_id()
        ec = rcfg.get("eevee", {})
        e = scene.eevee
        e.taa_render_samples = ec.get("samples", 32)
        if hasattr(e, "use_raytracing"):
            e.use_raytracing = bool(ec.get("raytracing", False))
        r.filter_size = 1.5
    r.film_transparent = False
    r.use_motion_blur = False  # motion blur is integrated over sub-frame renders instead
    compat_set_exr(r.image_settings, multilayer=False, depth="16")
    r.use_file_extension = False
    return dev


def render_to(scene, path):
    """Render the current state to `path` atomically (tmp + rename)."""
    tmp = path + ".part"
    scene.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    os.replace(tmp, path)


def set_border_rows(scene, row0, row1, H, pad=2):
    """Restrict rendering to image rows [row0, row1) (top-down), full width."""
    r = scene.render
    if row0 <= 0 and row1 >= H:
        r.use_border = False
        return
    r.use_border = True
    r.use_crop_to_border = False
    r.border_min_x, r.border_max_x = 0.0, 1.0
    r.border_min_y = max(0.0, 1.0 - (row1 + pad) / H)
    r.border_max_y = min(1.0, 1.0 - (row0 - pad) / H)
