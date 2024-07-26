import bpy
import csv
import os
import numpy as np
from math import degrees

# Function to get calibration matrix K from Blender
def get_calibration_matrix_K_from_blender(mode='simple'):
    scene = bpy.context.scene
    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale
    height = scene.render.resolution_y * scale
    camdata = scene.camera.data

    if mode == 'simple':
        aspect_ratio = width / height
        K = np.zeros((3, 3), dtype=np.float32)
        K[0][0] = width / (2 * np.tan(camdata.angle / 2))
        K[1][1] = height / (2 * np.tan(camdata.angle / 2))
        K[0][2] = width / 2
        K[1][2] = height / 2
        K[2][2] = 1
        K = K.transpose()

    return K

# Function to move the camera in a circular motion around the performer
def move_camera_to_frame(frame, camera, radius, height_offset, angle_offset, target=(0, 0, 0)):
    angle = (frame % 360) * (np.pi / 180)
    camera.location.x = radius * np.cos(angle + angle_offset)
    camera.location.y = radius * np.sin(angle + angle_offset)
    camera.location.z = height_offset  
    camera.rotation_euler = camera.location.to_track_quat('Z', 'Y').to_euler()  # Point the camera to the target

# Blender setup
scn = bpy.context.scene

# Create and configure the camera
cam1 = bpy.data.cameras.new("Camera")
Camera = bpy.data.objects.new("Camera", cam1)
scn.camera = Camera

fov = 40.0
pi = 3.14159265

Camera.data.angle = fov * (pi / 180.0)
Camera.rotation_mode = 'XYZ'
scn.collection.objects.link(Camera)

# Adjust the camera position
Camera.location = (0, -3, 1.5)  # Move camera closer to the subject

# Add lamps for lighting
lamp_data_1 = bpy.data.lights.new(name="Lamp1", type='AREA')
lamp_data_1.energy = 1000
lamp_data_1.size = 5.0
lamp_object_1 = bpy.data.objects.new(name="Lamp1", object_data=lamp_data_1)
lamp_object_1.location = (5, -5, 5)
scn.collection.objects.link(lamp_object_1)

lamp_data_2 = bpy.data.lights.new(name="Lamp2", type='AREA')
lamp_data_2.energy = 1000
lamp_data_2.size = 5.0
lamp_object_2 = bpy.data.objects.new(name="Lamp2", object_data=lamp_data_2)
lamp_object_2.location = (-5, 5, 5)
scn.collection.objects.link(lamp_object_2)

# Add additional light for better back lighting
lamp_data_3 = bpy.data.lights.new(name="Lamp3", type='POINT')
lamp_data_3.energy = 800
lamp_object_3 = bpy.data.objects.new(name="Lamp3", object_data=lamp_data_3)
lamp_object_3.location = (0, 0, 5)
scn.collection.objects.link(lamp_object_3)

# Set output folder and other parameters
output_folder = "C:/Users/khotv/rp_aliyah_4d_004_dancing_BLD"
os.makedirs(output_folder, exist_ok=True)
scale = 1
depth_scale = 1.0  
color_depth = '16'
file_format = 'PNG'
resolution = 600
engine = 'BLENDER_EEVEE'

# Set render settings
scene = bpy.context.scene
render = scene.render
render.engine = engine
render.image_settings.color_mode = 'RGBA'
render.image_settings.color_depth = color_depth
render.image_settings.file_format = file_format
render.resolution_x = resolution
render.resolution_y = resolution
render.resolution_percentage = 100
render.film_transparent = True

# Enable nodes for compositing
scene.use_nodes = True
nodes = scene.node_tree.nodes
links = scene.node_tree.links

# Clear existing nodes
for n in nodes:
    nodes.remove(n)

# Create render layers node
render_layers = nodes.new('CompositorNodeRLayers')

# Enable depth and normal passes
scene.view_layers["View Layer"].use_pass_normal = True
scene.view_layers["View Layer"].use_pass_z = True

# Depth output node
depth_file_output = nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
depth_file_output.base_path = os.path.join(output_folder, 'depth')
depth_file_output.file_slots[0].use_node_format = True
depth_file_output.format.file_format = file_format
depth_file_output.format.color_depth = color_depth

map_range = nodes.new(type="CompositorNodeMapRange")
map_range.inputs['From Min'].default_value = 0.0  
map_range.inputs['From Max'].default_value = 10.0
map_range.inputs['To Min'].default_value = 0.0
map_range.inputs['To Max'].default_value = 1.0

# Set clamp through node properties
map_range.use_clamp = True

links.new(render_layers.outputs['Depth'], map_range.inputs[0])
links.new(map_range.outputs[0], depth_file_output.inputs[0])

# Albedo output node
alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")

# Check available outputs in render_layers node
available_outputs = render_layers.outputs.keys()
if 'Image' in available_outputs:
    links.new(render_layers.outputs['Image'], alpha_albedo.inputs['Image'])
else:
    print("Diffuse Color output not found in render layers node. Please check node configuration.")

links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
albedo_file_output.base_path = os.path.join(output_folder, 'albedo')
albedo_file_output.file_slots[0].use_node_format = True
albedo_file_output.format.file_format = file_format
albedo_file_output.format.color_mode = 'RGBA'
albedo_file_output.format.color_depth = color_depth
links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])

# Normal output node
normal_file_output = nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = 'Normal Output'
normal_file_output.base_path = os.path.join(output_folder, 'normal')
normal_file_output.file_slots[0].use_node_format = True
normal_file_output.format.file_format = file_format
normal_file_output.format.color_depth = color_depth
links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

# Initialize CSV writer for camera parameters
csv_filename = 'camera_parameters.csv'
csv_path = os.path.join(output_folder, csv_filename)
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Position_X', 'Position_Y', 'Position_Z', 'Rotation_X', 'Rotation_Y', 'Rotation_Z', 'K'])

# Ensure there is an active camera
if scene.camera is None:
    print("No active camera found. Please set a camera in the scene.")
    bpy.ops.wm.quit_blender()
    csv_file.close()
else:
    # Render 150 frames
    for frame_nr in range(1, 151):
        scene.frame_set(frame_nr)

        # Adjust the camera path
        if frame_nr <= 50:
            radius = 5
            height_offset = 2  # Middle level
            angle_offset = 0
        elif frame_nr <= 100:
            radius = 7
            height_offset = 6  # Top looking down
            angle_offset = np.pi / 4
        else:
            radius = 3
            height_offset = -2  # Bottom looking up
            angle_offset = np.pi / 2

        # Move camera to the frame position
        move_camera_to_frame(frame_nr, scene.camera, radius, height_offset, angle_offset)

        # Update file output paths
        depth_file_output.base_path = os.path.join(output_folder, f'depth/frame_{frame_nr:03d}')
        albedo_file_output.base_path = os.path.join(output_folder, f'albedo/frame_{frame_nr:03d}')
        normal_file_output.base_path = os.path.join(output_folder, f'normal/frame_{frame_nr:03d}')
        
        # Ensure output directories exist
        os.makedirs(depth_file_output.base_path, exist_ok=True)
        os.makedirs(albedo_file_output.base_path, exist_ok=True)
        os.makedirs(normal_file_output.base_path, exist_ok=True)

        # Render image
        render.filepath = os.path.join(output_folder, f"frame_{frame_nr:03d}.png")
        bpy.ops.render.render(write_still=True)

        # Get active camera
        cam = scene.camera

        # Camera transform matrix
        m = cam.matrix_world

        # Compute position and rotation
        pos = m.to_translation()
        rot = m.to_euler('XYZ')

        # Convert rotation to degrees
        rot_deg = [degrees(rot.x), degrees(rot.y), degrees(rot.z)]

        # Get calibration matrix K
        K = get_calibration_matrix_K_from_blender(mode='simple')

        # Print camera parameters to console
        print(f"Frame: {frame_nr}")
        print(f"Position: X={pos.x}, Y={pos.y}, Z={pos.z}")
        print(f"Rotation: X={rot_deg[0]}, Y={rot_deg[1]}, Z={rot_deg[2]}")
        print(f"Calibration Matrix K:\n{K}")

        # Write camera parameters to CSV
        csv_writer.writerow([frame_nr, pos.x, pos.y, pos.z, rot_deg[0], rot_deg[1], rot_deg[2], K.tolist()])

    # Close CSV file
    csv_file.close()
    print("Camera parameters saved to:", csv_path)
    print("Render completed successfully.")


