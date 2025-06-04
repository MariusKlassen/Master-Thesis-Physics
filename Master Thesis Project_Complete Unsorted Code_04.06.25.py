#%% Rotierende Windmühle Stimulus mit dreieckigen Rotorblättern 

import numpy as np
import matplotlib.pyplot as plt

# Frame parameters
zeilen = 1080
spalten = 1920
frames = 16  # Number of frames in the animation

# Windmill parameters
max_diameter = 400              # Maximum diameter the windmill blades can reach
center_x = spalten // 2          # Center position (float for subpixel accuracy)
center_y = zeilen // 2           # Center position (float for subpixel accuracy)
hell = 255                      # White (background) - float scale [0.0, 1.0]
dunkel = 0                      # Black (blade color)
num_blades = 4                  # Number of blades on the windmill
blade_thickness_angle = 1/12 * np.pi  # Angular thickness of each blade in radians

# New parameters for tuning
#start_angle = 1/4 * np.pi        # Initial starting angle of the windmill (in radians)
start_angle = 0
#rotation_speed = 1/8 * np.pi     # Rotation speed per frame (in radians)
#rotation_speed = -1/4 * np.pi
rotation_speed = -1/16 * np.pi
print(f'rotation_speed (vom stimulus) = {rotation_speed/np.pi}π')

# Generate a 3D float array (height, width, frames) filled with 1.0 (white)
matrix = np.full((zeilen, spalten, frames), fill_value=hell, dtype=np.float32)

# Function to draw a single frame of the rotating windmill with radial symmetry
def draw_windmill(matrix, center_x, center_y, max_radius, frame, 
                  num_blades, blade_thickness_angle, start_angle, rotation_speed):
    
    angle_step = 2 * np.pi / num_blades       # Angle between blades
    # Flip the sign of rotation_speed to reverse direction. For + frame * rotation speed: rotation_speed positive -> clockwise rotation and rotation_speed negative -> counter-clockwise rotation. For - frame * rotation speed: rotation_speed positive -> counter-clockwise rotation and rotation_speed negative -> clockwise rotation 
    #rotation_angle = start_angle + frame * rotation_speed  # Current rotation angle for this frame. Clockwise rotation for positive rotation_speed and counter-clockwise rotation for negative rotation_speed
    rotation_angle = start_angle - frame * rotation_speed  # Current rotation angle for this frame. Clockwise rotation for negative rotation_speed and counter-clockwise rotation for positive rotation_speed

    # Create a mask for the frame with white background
    frame_mask = np.full((zeilen, spalten), fill_value=hell, dtype=np.float32)
    
    # Calculate pixel positions relative to the center
    y, x = np.ogrid[:zeilen, :spalten]
    #x = x.astype(float) - center_x
    #y = y.astype(float) - center_y
    x = x - center_x
    y = y - center_y
    
    # Convert cartesian coordinates to polar (angle and radius)
    distance_from_center = np.sqrt(x**2 + y**2)
    angle_from_center = np.arctan2(y, x)
    
    for i in range(num_blades):
        # Calculate the central angle for the current blade
        blade_angle = rotation_angle + i * angle_step

        # Define blade using angle range for symmetric blades
        within_blade = np.abs((angle_from_center - blade_angle + np.pi) % (2 * np.pi) - np.pi) < blade_thickness_angle
        within_radius = distance_from_center <= max_radius
        
        # Apply the blade mask to the frame
        frame_mask[within_blade & within_radius] = dunkel

    # Copy the generated frame into the main matrix
    matrix[:, :, frame] = frame_mask

# Fill the matrix with the rotating windmill for each frame
for k in range(frames):
    draw_windmill(matrix, center_x, center_y, max_diameter / 2, k, 
                  num_blades, blade_thickness_angle, start_angle, rotation_speed)


plt.imshow(matrix[:, :, 0], cmap='gray', interpolation='none')
plt.colorbar()
plt.show()



#%% Rotierende Windmühle Stimulus mit dreieckigen Rotorblättern mit Rotate Function von scipy.ndimage

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# Frame parameters
zeilen = 1080  # Height of each frame
spalten = 1920  # Width of each frame
frames = 16  # Number of frames in the animation

# Windmill parameters
max_diameter = 400  # Maximum diameter of the windmill blades
center_x = spalten // 2  # Center x-coordinate
center_y = zeilen // 2  # Center y-coordinate
hell = 255  # White (background)
dunkel = 0  # Black (blades)
num_blades = 4  # Number of blades
blade_thickness_angle = 1/12 * np.pi  # Angular thickness of each blade in radians

# Rotation parameters
start_angle = 0  # Initial starting angle (in radians)
rotation_speed = -1/16 * np.pi  # Rotation speed per frame (in radians) (negative for clockwise, positive for counter-clockwise)
print(f'rotation_speed (vom stimulus) = {rotation_speed/np.pi}π')

# Generate the initial frame
initial_frame = np.full((zeilen, spalten), fill_value=hell, dtype=np.float32)

# Create windmill blades on the initial frame
y, x = np.ogrid[:zeilen, :spalten]
x = x - center_x
y = y - center_y
distance_from_center = np.sqrt(x**2 + y**2)
angle_from_center = np.arctan2(y, x)

# Draw the blades on the initial frame
angle_step = 2 * np.pi / num_blades  # Angle between blades
for i in range(num_blades):
    blade_angle = start_angle + i * angle_step
    within_blade = np.abs((angle_from_center - blade_angle + np.pi) % (2 * np.pi) - np.pi) < blade_thickness_angle
    within_radius = distance_from_center <= max_diameter / 2
    initial_frame[within_blade & within_radius] = dunkel  # Set blade color

# Initialize the 3D array for storing frames
matrix = np.zeros((zeilen, spalten, frames), dtype=np.float32)

# Generate each frame by rotating the initial frame
for k in range(frames):
    angle = k * np.degrees(rotation_speed)  # Convert radians to degrees because angle parameter needs to be given to the rotate function in degrees
    rotated_frame = rotate(
        initial_frame,
        angle=angle,
        axes=(0, 1),  # Rotate in the (y, x) plane
        reshape=False,
        mode='nearest',  # Cval can be used for mode='constant'. Cval=hell: set the background color to white after rotation
        order=0  # Bilinear interpolation for better quality
    )
    matrix[:, :, k] = rotated_frame

# Display the first and another frame verify
plt.imshow(matrix[:, :, 0], cmap='gray', interpolation='none')
plt.title("First Frame of Rotating Windmill")
plt.colorbar()
plt.show()

# Display a frame to verify
i = 15
plt.imshow(matrix[:, :, i], cmap='gray', interpolation='none')
plt.title(f"Frame {i} of Rotating Windmill")
plt.colorbar()
plt.show()


# Display the first and another frame verify
#fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#axes[0].imshow(matrix[:, :, 0], cmap='gray', interpolation='none')
#axes[0].set_title("First Frame")

#k = 4
#axes[1].imshow(matrix[:, :, k], cmap='gray', interpolation='none')
#axes[1].set_title(f"Frame: {k}")

#plt.show()



#%% Rotierende Windmühle Stimulus mit rechteckigen Rotorblättern
# Breite der Rotorblätter als blade_thickness_pixels

import numpy as np
import matplotlib.pyplot as plt

# Frame parameters
zeilen = 1080
spalten = 1920
frames = 16  # Number of frames in the animation

# Windmill parameters
max_diameter = 400              # Maximum diameter the windmill blades can reach
center_x = spalten // 2          # Center position (float for subpixel accuracy)
center_y = zeilen // 2           # Center position (float for subpixel accuracy)
hell = 255                      # White (background) - float scale [0.0, 1.0]
dunkel = 0                      # Black (blade color)
num_blades = 4                  # Number of blades on the windmill
blade_thickness_pixels = 20.0     # Absolute blade thickness in pixels. The max_radius should be dividable by it

# New parameters for tuning
#start_angle = 1/4 * np.pi        # Initial starting angle of the windmill (in radians)
start_angle = 0
#rotation_speed = 1/8 * np.pi     # Rotation speed per frame (in radians)
#rotation_speed = -1/4 * np.pi
rotation_speed = -1/16 * np.pi
print(f'rotation_speed (vom stimulus) = {rotation_speed/np.pi}π')

# Generate a 3D float array (height, width, frames) filled with white
matrix = np.full((zeilen, spalten, frames), fill_value=hell, dtype=np.float32)

# Function to draw a single frame of the rotating windmill with rectangular blades
def draw_windmill(matrix, center_x, center_y, max_radius, frame, 
                  num_blades, blade_thickness_pixels, start_angle, rotation_speed):
    
    angle_step = 2 * np.pi / num_blades       # Angle between blades
    rotation_angle = start_angle - frame * rotation_speed  # Current rotation angle for this frame

    # Create a mask for the frame with white background
    frame_mask = np.full((zeilen, spalten), fill_value=hell, dtype=np.float32)
    
    # Calculate pixel positions relative to the center
    y, x = np.ogrid[:zeilen, :spalten]
    x = x - center_x
    y = y - center_y
    
    # Distance from center for limiting blade length
    distance_from_center = np.sqrt(x**2 + y**2)

    for i in range(num_blades):
        # Calculate the central angle for the current blade
        blade_angle = rotation_angle + i * angle_step

        # Define rectangular blade geometry using absolute blade thickness in pixels
        # Calculate the perpendicular distance from the blade's axis (to control blade thickness)
        distance = (x * np.sin(blade_angle) - y * np.cos(blade_angle))  # Distance to blade axis
        within_blade = np.abs(distance) < blade_thickness_pixels  # Control blade thickness directly in pixels
        within_radius = distance_from_center <= max_radius  # Limit to circular area

        # Apply the blade mask to the frame
        frame_mask[within_blade & within_radius] = dunkel

    # Copy the generated frame into the main matrix
    matrix[:, :, frame] = frame_mask

# Fill the matrix with the rotating windmill for each frame
for k in range(frames):
    draw_windmill(matrix, center_x, center_y, max_diameter / 2, k, 
                  num_blades, blade_thickness_pixels, start_angle, rotation_speed)

# Display the first frame to verify
plt.imshow(matrix[:, :, 0], cmap='gray', interpolation='none')
plt.colorbar()
plt.show()



#%% Rotierende Windmühle Stimulus mit rechteckigen Rotorblättern mit Rotate Function von scipy.ndimage

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# Frame parameters
zeilen = 1080  # Height of each frame
spalten = 1920  # Width of each frame
frames = 16  # Number of frames in the animation

# Windmill parameters
max_diameter = 400  # Maximum diameter of the windmill blades
center_x = spalten // 2  # Center x-coordinate
center_y = zeilen // 2  # Center y-coordinate
hell = 255  # White (background)
dunkel = 0  # Black (blades)
num_blades = 4  # Number of blades
blade_thickness_pixels = 20  # Blade thickness in pixels

# Rotation parameters
start_angle = 0  # Starting angle in radians
rotation_speed = -1 / 16 * np.pi  # Rotation speed per frame (negative for clockwise)
print(f'rotation_speed (vom stimulus) = {rotation_speed/np.pi}π')

# Generate the initial frame
initial_frame = np.full((zeilen, spalten), fill_value=hell, dtype=np.float32)

# Draw the windmill blades on the initial frame
y, x = np.ogrid[:zeilen, :spalten]
x = x - center_x
y = y - center_y
distance_from_center = np.sqrt(x**2 + y**2)

# Draw each blade
angle_step = 2 * np.pi / num_blades  # Angle between blades
for i in range(num_blades):
    blade_angle = start_angle + i * angle_step
    
    # Calculate perpendicular distance to the blade axis
    distance_to_blade_axis = x * np.sin(blade_angle) - y * np.cos(blade_angle)
    
    # Mask for the blade: rectangular geometry
    within_blade = (np.abs(distance_to_blade_axis) < blade_thickness_pixels)
    within_radius = (distance_from_center <= max_diameter / 2)
    
    # Apply blade color (black) to the initial frame
    initial_frame[within_blade & within_radius] = dunkel

# Initialize the 3D array for storing frames
matrix = np.zeros((zeilen, spalten, frames), dtype=np.float32)

# Generate each frame by rotating the initial frame
for k in range(frames):
    angle = np.degrees(k * rotation_speed)  # Convert radians to degrees because angle parameter needs to be given to the rotate function in degrees
    rotated_frame = rotate(
        initial_frame,
        angle=angle,
        axes=(0, 1), 
        reshape=False,
        mode='nearest',
        order=0  
    )
    matrix[:, :, k] = rotated_frame

# Display one frame to verify
i = 3
plt.imshow(matrix[:, :, i], cmap='gray', interpolation='none')
plt.title(f"Frame {i} of Rotating Windmill")
plt.colorbar()
plt.show()



#%% Statischer Weißer Screen mit bestimmten Wert

import numpy as np
import matplotlib.pyplot as plt

# Frame parameters
zeilen = 32
spalten = 32
frames = 32  # Number of frames in the animation

hell = 1
dunkel = 0

# Generate 3D array
matrix = np.full((zeilen, spalten, frames), fill_value=hell, dtype=np.uint8)

# Display the first frame to verify
plt.imshow(matrix[:, :, 0], cmap='gray', interpolation='none')
plt.colorbar()
plt.show()



#%% Statischer Screen mit bestimmten Wert für 1. und 2. Hälfte (1)

import numpy as np
import matplotlib.pyplot as plt
import os

# Frame parameters
zeilen = 32
spalten = 32  
frames = 32  # Number of frames in the animation

hell = 1  
dunkel = 0  

# Generate 3D array with all elements set to 'hell' (white)
matrix = np.full((zeilen, spalten, frames), fill_value=hell, dtype=np.uint8)

bereichsgrenze = 16
print("Bereichsgrenze (vom stimulus):", bereichsgrenze)

# Update the left half of the array to 'dunkel' (black) for all frames
#matrix[:, :spalten // 2, :] = dunkel
matrix[:, :bereichsgrenze, :] = dunkel

# Display a frame to verify
frame = 0
plt.imshow(matrix[:, :, frame], cmap='gray', interpolation='none')
plt.colorbar()
plt.title(f'Left Half Black (Frame {frame})\n'
    f'Bereichsgrenze: {bereichsgrenze:.0f}')
plt.show()

# Display a frame to verify
frame = 1
plt.imshow(matrix[:, :, frame], cmap='gray', interpolation='none')
plt.colorbar()
plt.title(f'Left Half Black (Frame {frame})\n'
    f'Bereichsgrenze: {bereichsgrenze:.0f}')
plt.show()


#### DAS FOLGENDE FUNKTIONIERT SO NICHT ###
# Save the Plot automatically 
# Specify the base directory where the folder should be created
#base_directory = r'/Users/marius_klassen/Desktop/Masterarbeit - AE Lappe/Coding/Projekt 2/Convolution/Twisted Kernel_3D_Rotation 2D_Rotationswinkel proportional zu Frames_3D Gauß im Gabor (ab 18.11)_kein Fehler mehr bei Convolution_richtigerweise einmal Sinus und einmal Cosinus in Convolution/Actionpoint 49 vom 28.11/Plots/Mit Correlate anstatt Convolve/Nicht-zentrierte Stimuli/Test 2.1.1_1. Hälfte schwarz (Wert 0), 2. Hälfte weiß (Wert = 1) (statisch)/Convolution nur entlang z-Achse für max aktivierten Kernel/Stimulus'  # Change this to your desired path
#save_directory = os.path.join(base_directory, 'Bereichsgrenze = {bereichsgrenze}')  # Add 'plots' subfolder

# Create the directory if it doesn't exist
#if not os.path.exists(save_directory):
#    os.makedirs(save_directory)

# Construct the filename using the 'bereichsgrenze' variable
#filename = f'Bereichsgrenze = {bereichsgrenze}.png'
#filepath = os.path.join(save_directory, filename)  # Full path to save the plot

# Save the plot
#plt.savefig(filepath)
#plt.show()



#%% Statischer Screen mit bestimmten Wert für 1. und 2. Hälfte (2)

import numpy as np
import matplotlib.pyplot as plt

# Frame parameters
zeilen = 32
spalten = 32  
frames = 32  # Number of frames in the animation

hell = 1  
dunkel = 0  

# Generate 3D array with all elements set to 'hell' (white)
matrix = np.full((zeilen, spalten, frames), fill_value=hell, dtype=np.uint8)

# Update the left half of the array to 'dunkel' (black) for all frames
matrix[:, spalten // 2:, :] = dunkel

# Display the first frame to verify
plt.imshow(matrix[:, :, 0], cmap='gray', interpolation='none')
plt.colorbar()
plt.title("First Frame with Right Half Black")
plt.show()



#%% Rotierender Screen mit bestimmten Wert für 1. und 2. Hälfte (1) mit Rotate Function von scipy.ndimage

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from fractions import Fraction

# Frame parameters
zeilen = 32  # Spatial Y dimension
spalten = 32  # Spatial X dimension
frames = 32  # Number of frames in the animation

hell = 1  # White value
dunkel = 0  # Black value

# Generate the initial frame: left half black, right half white
initial_frame = np.full((zeilen, spalten), fill_value=hell, dtype=np.uint8)
initial_frame[:, :spalten // 2] = dunkel  # Left half black

# Initialize the 3D array to store frames
matrix = np.zeros((zeilen, spalten, frames), dtype=np.uint8)

# Rotation parameters
#rotation_speed = -360 / frames  # Rotation speed in Degrees/Frame. Full rotation in one cycle (360 degrees)
#rotation_speed = -22.5  # Rotation speed in Degrees/Frame. Positive values: counter-clockwise rotation. Negative values: clockwise rotation
rotation_speed = 1/16 * np.pi # Rotation speed in Radians/Frame. Positive values: counter-clockwise rotation. Negative values: clockwise rotation
#rotation_speed = -1/32 * np.pi
rotation_speed_fraction = Fraction(rotation_speed/np.pi).limit_denominator()
print(f'rotation_speed (vom stimulus) = {rotation_speed_fraction}π')

# Generate each frame by rotating the initial frame
for i in range(frames):
    rotated_frame = rotate(
        initial_frame, 
        angle=i * np.degrees(rotation_speed),    # Convert radians to degrees because angle parameter needs to be given to the rotate function in degrees
        axes=(0, 1),        # Rotate in the (y, x) plane
        reshape=False,      # Keep the output frame size the same
        mode='nearest',     # Fill edges with nearest values to avoid artifacts
        order=0             # Bilinear interpolation for a balance of speed and quality
    )
    matrix[:, :, i] = rotated_frame
    
# Display the first frame
plt.imshow(matrix[:, :, 0], cmap='gray', interpolation='none')
plt.title("First Frame")
plt.colorbar()
plt.show()

# Display a frame to verify
i = 3
plt.imshow(matrix[:, :, i], cmap='gray', interpolation='none')
plt.title(f"Frame {i}")
plt.colorbar()
plt.show()    


# Display the first and another frame verify
#fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#axes[0].imshow(matrix[:, :, 0], cmap='gray', interpolation='none')
#axes[0].set_title("First Frame")

#k = 2
#axes[1].imshow(matrix[:, :, k], cmap='gray', interpolation='none')
#axes[1].set_title(f"Frame: {k}")

#plt.show()



#%% Rotierender Screen mit bestimmten Wert für 1. und 2. Hälfte (2) mit Rotate Function von scipy.ndimage

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from fractions import Fraction

# Frame parameters
zeilen = 32  # Spatial Y dimension
spalten = 32  # Spatial X dimension
frames = 32  # Number of frames in the animation

hell = 1  # White value
dunkel = 0  # Black value

# Generate the initial frame: left half black, right half white
initial_frame = np.full((zeilen, spalten), fill_value=hell, dtype=np.uint8)
initial_frame[:, spalten // 2:] = dunkel  # Left half black

# Initialize the 3D array to store frames
matrix = np.zeros((zeilen, spalten, frames), dtype=np.uint8)

# Rotation parameters
#rotation_speed = -360 / frames  # Rotation speed in Degrees/Frame. Full rotation in one cycle (360 degrees)
#rotation_speed = -22.5  # Rotation speed in Degrees/Frame. Positive values: counter-clockwise rotation. Negative values: clockwise rotation
rotation_speed = 1/16 * np.pi # Rotation speed in Radians/Frame. Positive values: counter-clockwise rotation. Negative values: clockwise rotation
#rotation_speed = -1/32 * np.pi
rotation_speed_fraction = Fraction(rotation_speed/np.pi).limit_denominator()
print(f'rotation_speed (vom stimulus) = {rotation_speed_fraction}π')

# Generate each frame by rotating the initial frame
for i in range(frames):
    rotated_frame = rotate(
        initial_frame, 
        angle=i * np.degrees(rotation_speed),    # Convert radians to degrees because angle parameter needs to be given to the rotate function in degrees
        axes=(0, 1),        # Rotate in the (y, x) plane
        reshape=False,      # Keep the output frame size the same
        mode='nearest',     # Fill edges with nearest values to avoid artifacts
        order=0             # Bilinear interpolation for a balance of speed and quality
    )
    matrix[:, :, i] = rotated_frame
    
# Display the first frame
plt.imshow(matrix[:, :, 0], cmap='gray', interpolation='none')
plt.title("First Frame")
plt.colorbar()
plt.show()

# Display a frame to verify
i = 3
plt.imshow(matrix[:, :, i], cmap='gray', interpolation='none')
plt.title(f"Frame {i}")
plt.colorbar()
plt.show()    


# Display the first and another frame verify
#fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#axes[0].imshow(matrix[:, :, 0], cmap='gray', interpolation='none')
#axes[0].set_title("First Frame")

#k = 2
#axes[1].imshow(matrix[:, :, k], cmap='gray', interpolation='none')
#axes[1].set_title(f"Frame: {k}")

#plt.show()



#%% Statischer Balken

import numpy as np
import matplotlib.pyplot as plt

# Dimensions of the matrix (image size)
zeilen = 32  # Height (y-axis)
spalten = 32  # Width (x-axis)
frames = 32  # Number of frames (time dimension)

# Bar parameters
breite_des_balkens_x = 4  # Width of the bar in the x-direction
breite_des_balkens_y = 12  # Width of the bar in the y-direction
print("Breite des Balkens in x:", breite_des_balkens_x)
print("Breite des Balkens in y:", breite_des_balkens_y)

# Static position of the bar
startposition_x = 15  # Initial x-position of the bar
startposition_y = 10  # Initial y-position of the bar

# Colors
hell = 1  # White background
dunkel = 0  # Black bar

# Generate 3D array (filled with white pixels)
matrix = np.full((zeilen, spalten, frames), fill_value=hell, dtype=np.uint8)

# Update the matrix to include the static bar in every frame
x_start = startposition_x
x_end =  startposition_x + breite_des_balkens_x
y_start = startposition_y
y_end = startposition_y + breite_des_balkens_y

# Set the bar region to black across all frames
matrix[y_start:y_end, x_start:x_end, :] = dunkel

# Display a specific frame 
frame = 0
plt.imshow(matrix[:, :, frame], cmap='gray', interpolation='none')
plt.colorbar()
plt.title(f'Frame {frame}')
plt.show()



#%% Rotierender Balken mit Rotate Function von scipy.ndimage (für zentrierte Stimuli)
### FUNKTIONIERT NUR FÜR DEN FALL, DASS DER BALKEN ZENTRIERT IST, DENN DIE ROTATE FUNKTION VON SCIPY.NDIMAGE ROTIERT DAS IMAGE UM DEN GANZEN FRAME  ###

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate  
from fractions import Fraction
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Dimensions of the matrix (image size)
#zeilen = 32  # Height (y-axis)
#spalten = 32  # Width (x-axis)
#frames = 32  # Number of frames (time dimension)
#zeilen = 33  # Height (y-axis)
#spalten = 33  # Width (x-axis)
#frames = 33  # To match 33 x 33 x 33 Kernels
#zeilen = 65  # Height (y-axis)
#spalten = 65  # Width (x-axis)
#frames = 65  # To match 65 x 65 x 65 Kernels
#zeilen = 16  # Height (y-axis)
#spalten = 16  # Width (x-axis)
#frames = 16  # To match 16 x 16 x 16 Kernels
zeilen = 1080  # For Full HD Array
spalten = 1920  # For Full HD Array
#frames = 64  # For Full HD Array
frames = 16  # For Full HD Array

# Bar parameters
#breite_des_balkens_x = 4  # Width of the bar in the x-direction
#breite_des_balkens_y = 12  # Width of the bar in the y-direction
#breite_des_balkens_x = 1  # Width of the bar in the x-direction
#breite_des_balkens_y = 12  # Width of the bar in the y-direction
#breite_des_balkens_x = 1  # Width of the bar in the x-direction
#breite_des_balkens_y = 6  # Width of the bar in the y-direction
breite_des_balkens_x = 15  # Width of the bar in the x-direction
breite_des_balkens_y = 200  # Width of the bar in the y-direction
print("Breite des Balkens in x:", breite_des_balkens_x)
print("Breite des Balkens in y:", breite_des_balkens_y)

# Static position of the bar
#startposition_x = 7  # Initial x-position of the bar. For 16 x  16 x 16 Kernels
#startposition_y = 5  # Initial y-position of the bar. For 16 x  16 x 16 Kernels
#startposition_x = 15  # Initial x-position of the bar. For 32 x  32 x 32 Kernels
#startposition_y = 10  # Initial y-position of the bar. For 32 x  32 x 32 Kernels
#startposition_x = 31  # Initial x-position of the bar. For 65 x 65 x 65 Kernels
#startposition_y = 26  # Initial y-position of the bar. For 65 x 65 x 65 Kernels
startposition_x = 500  # Initial x-position of the bar. For Full HD Array
startposition_y = 600  # Initial y-position of the bar. For Full HD Array

# Colors
hell = 1  # White background
dunkel = 0  # Black bar

# Rotation parameters
#rotation_speed = -360 / frames  # Rotation speed in Degrees/Frame. Full rotation in one cycle (360 degrees)
#rotation_speed = -22.5  # Rotation speed in Degrees/Frame. Positive values: counter-clockwise rotation. Negative values: clockwise rotation
rotation_speed = 1/16 * np.pi # Rotation speed in Radians/Frame. Positive values: counter-clockwise rotation. Negative values: clockwise rotation
#rotation_speed = -1/32 * np.pi
rotation_speed_fraction = Fraction(rotation_speed/np.pi).limit_denominator()
print(f'rotation_speed (vom stimulus) = {rotation_speed_fraction}π')

# Update the matrix to include the static bar in every frame
x_start = startposition_x
x_end =  startposition_x + breite_des_balkens_x
y_start = startposition_y
y_end = startposition_y + breite_des_balkens_y

# Set the bar region to black across all frames
initial_frame = np.full((zeilen, spalten), fill_value=hell, dtype=np.uint8)
initial_frame[y_start:y_end, x_start:x_end] = dunkel

# Generate 3D array (filled with white pixels)
matrix = np.full((zeilen, spalten, frames), fill_value=hell, dtype=np.uint8)

# Generate each frame by rotating the initial frame
for i in range(frames):
    rotated_frame = rotate(
        initial_frame, 
        angle=i * np.degrees(rotation_speed),    # Convert radians to degrees because angle parameter needs to be given to the rotate function in degrees
        axes=(0, 1),        # Rotate in the (y, x) plane
        reshape=False,      # Keep the output frame size the same
        mode='nearest',     # Fill edges with nearest values to avoid artifacts
        order=3,            # Bilinear interpolation for a balance of speed and quality
        #order=0 
    )
    matrix[:, :, i] = rotated_frame
    
# Display a specific frame 
frame = 0
plt.imshow(matrix[:, :, frame], cmap='gray', interpolation='none')
plt.colorbar()
plt.title(f'Frame {frame}')
#plt.gca().set_aspect('equal')  # Adjust aspect ratio
plt.show()

# Display another frame
frame = 15
plt.imshow(matrix[:, :, frame], cmap='gray', interpolation='none')
#plt.colorbar()
plt.title(f'Frame {frame}')

# Get current axis
ax = plt.gca()

# Adjust aspect ratio
ax.set_aspect('equal')

# Add a colorbar with size matching the y-axis
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)  # Adjust size and padding. Size=0.2 means 20% of the height of the axes. Also e.g. size="5%" possible i.e. width of the colorbar as 5% of the plot's width
cb = plt.colorbar(ax.images[0], cax=cax)
#cb.set_label('Intensity')  # Add a label to the colorbar

# Show the plot
plt.show()



#%% Rotierender Balken mit Rotate Function von scipy.ndimage (auch für nicht-zentrierte Stimuli)
### FUNKTIONIERT AUCH FÜR DEN FALL, DASS DER BALKEN NICHT ZENTRIERT IST, INDEM DER BALKEN NACH ZENTRIERTER ROTATION DURCH DIE ROTATE FUNKTION VON SCIPY.NDIMAGE ZURÜCK AN DIE URSPRUNGSPOSITION GESHIFTET WIRD ###

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate  
from fractions import Fraction
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Dimensions of the matrix (image size)
#zeilen = 32  # Height (y-axis)
#spalten = 32  # Width (x-axis)
#frames = 32  # Number of frames (time dimension)
#zeilen = 33  # Height (y-axis)
#spalten = 33  # Width (x-axis)
#frames = 33  # To match 33 x 33 x 33 Kernels
#zeilen = 65  # Height (y-axis)
#spalten = 65  # Width (x-axis)
#frames = 65  # To match 65 x 65 x 65 Kernels
#zeilen = 16  # Height (y-axis)
#spalten = 16  # Width (x-axis)
#frames = 16  # To match 16 x 16 x 16 Kernels
zeilen = 1080  # For Full HD Array
spalten = 1920  # For Full HD Array
#frames = 64  
frames = 16  
#frames = 64

# Bar parameters
#breite_des_balkens_x = 4  # Width of the bar in the x-direction
#breite_des_balkens_y = 12  # Width of the bar in the y-direction
#breite_des_balkens_x = 1  
#breite_des_balkens_y = 12 
#breite_des_balkens_x = 1 
#breite_des_balkens_y = 6  
#breite_des_balkens_x = 15  
#breite_des_balkens_y = 200  
breite_des_balkens_x = 4  
breite_des_balkens_y = 200  # Width of the bar in the y-direction
print("Breite des Balkens in x:", breite_des_balkens_x)
print("Breite des Balkens in y:", breite_des_balkens_y)

# Static position of the bar
#startposition_x = 7  # Initial x-position of the bar. For 16 x  16 x 16 Kernels
#startposition_y = 5  # Initial y-position of the bar. For 16 x  16 x 16 Kernels
#startposition_x = 15  # Initial x-position of the bar. For 32 x  32 x 32 Kernels
#startposition_y = 10  # Initial y-position of the bar. For 32 x  32 x 32 Kernels
#startposition_x = 31  # Initial x-position of the bar. For 65 x 65 x 65 Kernels
#startposition_y = 26  # Initial y-position of the bar. For 65 x 65 x 65 Kernels
startposition_x = 500  # Initial x-position of the bar. For Full HD Array
startposition_y = 600  # Initial y-position of the bar. For Full HD Array

# Colors
hell = 1  # White background
dunkel = 0  # Black bar

# Rotation parameters
#rotation_speed = -360 / frames  # Rotation speed in Degrees/Frame. Full rotation in one cycle (360 degrees)
#rotation_speed = -22.5  # Rotation speed in Degrees/Frame. Positive values: counter-clockwise rotation. Negative values: clockwise rotation
#rotation_speed = 1/16 * np.pi  # Rotation speed in Radians/Frame. Positive values: counter-clockwise rotation. Negative values: clockwise rotation
#rotation_speed = -1/32 * np.pi
#rotation_speed = -1/64 * np.pi
#rotation_speed = 1/90 * np.pi
rotation_speed = 1/16 * np.pi  # Test 2.17 und 2.20 und 2.23 und 2.24 und 2.27
#rotation_speed = 1/15 * np.pi  # Test 2.21 und 2.22 (11.04.25) und Test 2.25 und 2.26
#rotation_speed = 1/32 * np.pi  # AP 68 (27.05.25)
#rotation_speed = 1/64 * np.pi  # AP 68 (27.05.25)
rotation_speed_fraction = Fraction(rotation_speed/np.pi).limit_denominator()
print(f'rotation_speed (vom stimulus) = {rotation_speed_fraction}π')

# Update the matrix to include the static bar in every frame
x_start = startposition_x
x_end =  startposition_x + breite_des_balkens_x
y_start = startposition_y
y_end = startposition_y + breite_des_balkens_y

# Generate 3D array (filled with white pixels)
matrix = np.full((zeilen, spalten, frames), fill_value=hell, dtype=np.uint8)

# Compute center of the image
center_y, center_x = zeilen // 2, spalten // 2

# Compute bar offset from the center
offset_x = startposition_x - center_x
offset_y = startposition_y - center_y

# Create an empty frame and place the bar at the center
initial_frame_centered = np.full((zeilen, spalten), fill_value=hell, dtype=np.uint8)
y_start_c = center_y - breite_des_balkens_y // 2
y_end_c = center_y + breite_des_balkens_y // 2
x_start_c = center_x - breite_des_balkens_x // 2
x_end_c = center_x + breite_des_balkens_x // 2
initial_frame_centered[y_start_c:y_end_c, x_start_c:x_end_c] = dunkel

# Generate each frame by rotating the centered bar and shifting it back
for i in range(frames):
    rotated_frame = rotate(
        initial_frame_centered, 
        angle=i * np.degrees(rotation_speed),  
        axes=(0, 1),        
        reshape=False,      
        mode='nearest',     
        order=3
    )
    #rotated_frame = (rotated_frame > 0.5).astype(np.uint8)  # Thresholding
    
    # Shift the rotated frame back to the original position
    matrix[:, :, i] = np.roll(rotated_frame, shift=(offset_y, offset_x), axis=(0, 1))


# Define a region to generate the matrix in (Test 2.21, 09.04. Test 2.23, 15.04)
col_start, col_end = 350, 650
row_start, row_end = 450, 750
matrix = matrix[row_start:row_end, col_start:col_end, :]
    
# Display a specific frame 
frame = 0
plt.imshow(matrix[:, :, frame], cmap='gray', interpolation='none')
plt.colorbar()
plt.title(f'Frame {frame}')
#plt.gca().set_aspect('equal')  # Adjust aspect ratio
plt.show()

# Display another frame
frame = 15
plt.imshow(matrix[:, :, frame], cmap='gray', interpolation='none')
#plt.colorbar()
plt.title(f'Frame {frame}')

# Get current axis
ax = plt.gca()

# Adjust aspect ratio
ax.set_aspect('equal')

# Add a colorbar with size matching the y-axis
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)  # Adjust size and padding. Size=0.2 means 20% of the height of the axes. Also e.g. size="5%" possible i.e. width of the colorbar as 5% of the plot's width
cb = plt.colorbar(ax.images[0], cax=cax)
#cb.set_label('Intensity')  # Add a label to the colorbar

# Show the plot
plt.show()



#%% Rotierender Balken Stimulus als Video ohne Achsen und Colorbar

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Assuming `matrix` is your 3D matrix with shape (height, width, num_frames)
height, width, num_frames = matrix.shape

# Create a video writer 1
#fps = 10  # Frames per second
#fps = 5
fps = 30  # Original video

#fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
#fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Lossless Codec for .avi files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4

# Specify the directory and file name for saving the video
output_directory = "/Users/marius_klassen/Desktop"  # Replace with your desired folder
#output_filename = "output_video.avi"           # Specify the file name
#output_filename = f"output_video_without axes and colorbar_Pixel in y = {zeilen}, Pixel in x = {spalten}, frames = {num_frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, breite_des_balkens_x = {breite_des_balkens_x}, breite_des_balkens_y = {breite_des_balkens_y}, startposition_x = {startposition_x}, startposition_y = {startposition_y}, hell = {hell}, dunkel = {dunkel}, fps = {fps}.avi"   
#output_filename = f"output_video_without axes and colorbar_Pixel in y = {zeilen}, Pixel in x = {spalten}, frames = {num_frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, breite_des_balkens_x = {breite_des_balkens_x}, breite_des_balkens_y = {breite_des_balkens_y}, startposition_x = {startposition_x}, startposition_y = {startposition_y}, hell = {hell}, dunkel = {dunkel}, fps = {fps}.mp4"           
#output_filename = f"output_video_without axes and colorbar_Pixel in y = {height}, Pixel in x = {width}, frames = {num_frames}, rotation_speed = {max_activation_alpha/np.pi:.4f}π, fps = {fps}.mp4"  # Test 2.29 (Actionpoint 64, 24.04.25): Autokorrelation der Kernels        
output_filename = f"output_video_x({col_start}-{col_end})_y({row_start}-{row_end})_Pixel in y = {row_end - row_start}, Pixel in x = {col_end - col_start}, frames = {frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, breite_des_balkens_x = {breite_des_balkens_x}, breite_des_balkens_y = {breite_des_balkens_y}, startposition_x = {startposition_x}, startposition_y = {startposition_y}, hell = {hell}, dunkel = {dunkel}, fps = {fps}.mp4"             
os.makedirs(output_directory, exist_ok=True)   # Ensure the directory exists
output_path = os.path.join(output_directory, output_filename)

# Create a video writer 2
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through the frames and save them to the video
for frame in range(num_frames):
    # Generate the frame using matplotlib
    plt.imshow(matrix[:, :, frame], cmap='gray', interpolation='none')
    plt.axis('off')  # Turn off axes for a cleaner image
    plt.tight_layout(pad=0)
    
    # Save the current frame as an image
    plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()  # Close the figure to save memory

    # Read the saved frame with OpenCV
    frame_image = cv2.imread('temp_frame.png')

    # Resize to match the video dimensions (if necessary)
    frame_image = cv2.resize(frame_image, (width, height))
    
    # Write the frame to the video
    video_writer.write(frame_image)

# Release the video writer
video_writer.release()

# Remove the temporary image file
os.remove('temp_frame.png')

print(f"Video saved at: {output_path}")



#%% Rotierender Balken Stimulus als Video ohne Achsen und Colorbar in bestimmtem Gebiet (Region of Interest = ROI)

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Define Region of Interest (ROI) - Modify these values as needed
#roi_x_start = 500  # Start x-coordinate
#roi_x_end = 516  # End x-coordinate
#roi_y_start = 600  # Start y-coordinate
#roi_y_end = 616  # End y-coordinate  
#roi_x_start = 526  
#roi_x_end = 542   
#roi_y_start = 592  
#roi_y_end = 608    
#roi_x_start = 560  
#roi_x_end = 592   
#roi_y_start = 600  
#roi_y_end = 632  
#roi_x_start = 500 - breite_des_balkens_y // 2 
#roi_x_end = 500 + breite_des_balkens_y // 2
#roi_y_start = 600 - breite_des_balkens_y // 2
#roi_y_end = 600 + breite_des_balkens_y // 2
#roi_x_start = 500 - 16 // 2  # For kernel size = 16
#roi_x_end = 500 + 16 // 2
#roi_y_start = 600 - 16 // 2
#roi_y_end = 600 + 16 // 2
#roi_x_start = 160   # For kernel size = 16 for 300 x 300 Input Array and point in Alpha Map at y=150 and x=160 (Test 2.27). (0,0) in Alpha Map corresponds to (0:16, 0:16, :) in the Stimulus -> (150,160) in Alpha Map corresponds to (150:166, 160:176, :) in the Stimulus
#roi_x_end = 160 + 16
#roi_y_start = 150
#roi_y_end = 150 + 16
#roi_x_start = 180   # For kernel size = 16 for 300 x 300 Input Array and point in Alpha Map at y=150 and x=180 (Test 2.28)
#roi_x_end = 180 + 16
#roi_y_start = 150
#roi_y_end = 150 + 16
roi_x_start = 40   # For kernel size = 16 for 300 x 300 Input Array and point in Alpha Map at y=150 and x=180 (Test 2.28)
roi_x_end = 40 + 16
roi_y_start = 150
roi_y_end = 150 + 16


# Crop the matrix to the selected ROI
matrix_cropped = matrix[roi_y_start:roi_y_end, roi_x_start:roi_x_end, :]

# Assuming `matrix` is your 3D matrix with shape (height, width, num_frames)
height, width, num_frames = matrix_cropped.shape

# Create a video writer 1
#fps = 10  # Frames per second
#fps = 5
fps = 30  # Original video

#fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
#fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Lossless Codec for .avi files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4

# Specify the directory and file name for saving the video
#output_directory = "/Users/marius_klassen/Desktop"  # Replace with your desired folder
#output_filename = "output_video.avi"           # Specify the file name
#output_filename = f"output_video_ROI_x({roi_x_start}-{roi_x_end})_y({roi_y_start}-{roi_y_end})_Pixel in y = {zeilen}, Pixel in x = {spalten}, frames = {frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, breite_des_balkens_x = {breite_des_balkens_x}, breite_des_balkens_y = {breite_des_balkens_y}, startposition_x = {startposition_x}, startposition_y = {startposition_y}, hell = {hell}, dunkel = {dunkel}, fps = {fps}.avi"
#output_filename = f"output_video_without axes and colorbar_ROI_x( {roi_x_start} - {roi_x_end} )_y( {roi_y_start} - {roi_y_end} )_Pixel in y = {zeilen}, Pixel in x = {spalten}, frames = {frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, breite_des_balkens_x = {breite_des_balkens_x}, breite_des_balkens_y = {breite_des_balkens_y}, startposition_x = {startposition_x}, startposition_y = {startposition_y}, hell = {hell}, dunkel = {dunkel}, fps = {fps}.mp4"           
output_filename = f"output_video_ROI_x({roi_x_start}-{roi_x_end})_y({roi_y_start}-{roi_y_end})_Pixel in y = {zeilen}, Pixel in x = {spalten}, frames = {frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, breite_des_balkens_x = {breite_des_balkens_x}, breite_des_balkens_y = {breite_des_balkens_y}, startposition_x = {startposition_x}, startposition_y = {startposition_y}, hell = {hell}, dunkel = {dunkel}, fps = {fps}.mp4"             
os.makedirs(output_directory, exist_ok=True)   # Ensure the directory exists
output_path = os.path.join(output_directory, output_filename)

# Create a video writer
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through the frames and save them to the video
for frame in range(num_frames):
    # Generate the frame using matplotlib
    plt.imshow(matrix_cropped[:, :, frame], cmap='gray', interpolation='none')
    plt.axis('off')  # Turn off axes for a cleaner image
    plt.tight_layout(pad=0)
    
    # Save the current frame as an image
    plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()  # Close the figure to save memory

    # Read the saved frame with OpenCV
    frame_image = cv2.imread('temp_frame.png')

    # Resize to match the video dimensions (if necessary)
    frame_image = cv2.resize(frame_image, (width, height))
    
    # Write the frame to the video
    video_writer.write(frame_image)

# Release the video writer
video_writer.release()

# Remove the temporary image file
os.remove('temp_frame.png')

print(f"Video saved at: {output_path}")



#%% Rotierende Quadrate Stimulus (Uniform) mit unterschiedlichen Farben mit Scaled Coarse Pink Noise (Background) 

import numpy as np
import matplotlib.pyplot as plt
from colorednoise import powerlaw_psd_gaussian
import scipy.ndimage

# Function to create a uniform color square
def create_uniform_square(shape, color):
    return np.full(shape, color, dtype=np.uint8)

# Function to generate pink noise for a 2D array
def generate_pink_noise_2d(shape, exponent=-1):  # Exponent = -1 for Pink Noise
    return powerlaw_psd_gaussian(exponent, shape)

# Function to create scaled coarse pink noise
def create_coarse_pink_noise_scaled(base_shape, coarse_factor, exponent=-1, target_min=0, target_max=255):
    downsampled_shape = (base_shape[0] // coarse_factor, base_shape[1] // coarse_factor)
    coarse_noise = generate_pink_noise_2d(downsampled_shape, exponent)
    coarse_noise = scipy.ndimage.zoom(coarse_noise, coarse_factor, order=0, mode='nearest')

    # Normalize and scale to target range [target_min, target_max]
    coarse_noise_min = coarse_noise.min()
    coarse_noise_max = coarse_noise.max()
    coarse_noise_normalized = (coarse_noise - coarse_noise_min) / (coarse_noise_max - coarse_noise_min)
    coarse_noise_scaled = coarse_noise_normalized * (target_max - target_min) + target_min

    return coarse_noise_scaled

# Define shape and parameters for the background
background_shape = (1080, 1920)
frames = 64
coarse_factor_background = 2  # Higher value -> Larger patches (coarser noise)

# Generate coarse pink noise background
background = create_coarse_pink_noise_scaled(background_shape, coarse_factor_background)
background_3d = np.repeat(background[:, :, np.newaxis], frames, axis=2)

# Number of squares to create a rotation effect
num_squares = 8

# Radius of rotation and center of rotation
radius = 100
center_x = background_shape[1] // 2  # Horizontal center
center_y = background_shape[0] // 2  # Vertical center

# Angular speed (radians per frame)
angular_speed = 2 * np.pi / frames

# Create uniform squares for rotation (colors alternate for better visibility)
square_size = 40
colors = np.linspace(50, 200, num_squares, dtype=np.uint8)  # Different colors for each square

# Generate squares at different angles around the center
squares = [create_uniform_square((square_size, square_size), colors[i]) for i in range(num_squares)]

# Initialize angles for each square
angles = np.linspace(0, 2 * np.pi, num_squares, endpoint=False)

# Update background by moving squares in a rotational pattern
for frame in range(frames):
    for i, square in enumerate(squares):
        # Update angle for this square
        #current_angle = angles[i] + frame * rotation_speed  # Current rotation angle for this frame. Clockwise rotation for positive rotation_speed and counter-clockwise rotation for negative rotation_speed
        current_angle = angles[i] - frame * rotation_speed  # Current rotation angle for this frame. Clockwise rotation for negative rotation_speed and counter-clockwise rotation for positive rotation_speed

        # Calculate new x and y positions based on the angle and radius
        #square_x = int(center_x + radius * np.cos(current_angle)) % (background_shape[1] - square_size)  # Ensures square stays within width
        #square_y = int(center_y + radius * np.sin(current_angle)) % (background_shape[0] - square_size)  # Ensures square stays within height
        square_x = int(center_x + radius * np.cos(current_angle)) 
        square_y = int(center_y + radius * np.sin(current_angle)) 

        # Place the square at the new position
        background_3d[square_y:square_y + square_size, square_x:square_x + square_size, frame] = square

# Plot the background with the rotating squares (Frame 0)
plt.imshow(background_3d[:, :, 0], cmap='gray', interpolation='none')
plt.colorbar()
plt.title('Background with Rotating Squares (Frame 0)')
plt.show()

# Display another frame (e.g., Frame 60)
k = 2
plt.imshow(background_3d[:, :, k], cmap='gray', interpolation='none')
plt.colorbar()
plt.title(f'Background with Rotating Squares (Frame {k})')
plt.show()

# For later convolution, set the moving stimulus as the matrix
matrix = background_3d



#%% Rotierende Quadrate Stimulus (Uniform) mit derselben Farbe mit Scaled Coarse Pink Noise (Background)

import numpy as np
import matplotlib.pyplot as plt
from colorednoise import powerlaw_psd_gaussian
import scipy.ndimage

# Function to create a uniform color square
def create_uniform_square(shape, color):
    return np.full(shape, color, dtype=np.uint8)

# Function to generate pink noise for a 2D array
def generate_pink_noise_2d(shape, exponent=-1):  # Exponent = -1 for Pink Noise
    return powerlaw_psd_gaussian(exponent, shape)

# Function to create scaled coarse pink noise
def create_coarse_pink_noise_scaled(base_shape, coarse_factor, exponent=-1, target_min=0, target_max=255):
    downsampled_shape = (base_shape[0] // coarse_factor, base_shape[1] // coarse_factor)
    coarse_noise = generate_pink_noise_2d(downsampled_shape, exponent)
    coarse_noise = scipy.ndimage.zoom(coarse_noise, coarse_factor, order=0, mode='nearest')

    # Normalize and scale to target range [target_min, target_max]
    coarse_noise_min = coarse_noise.min()
    coarse_noise_max = coarse_noise.max()
    coarse_noise_normalized = (coarse_noise - coarse_noise_min) / (coarse_noise_max - coarse_noise_min)
    coarse_noise_scaled = coarse_noise_normalized * (target_max - target_min) + target_min

    return coarse_noise_scaled

# Define shape and parameters for the background
#background_shape = (1080, 1920)
background_shape = (32, 32)
#frames = 64
frames = 32
coarse_factor_background = 2  # Higher value -> Larger patches (coarser noise)
print("Background Coarse Factor:", coarse_factor_background)

# Generate coarse pink noise background
background = create_coarse_pink_noise_scaled(background_shape, coarse_factor_background)
background_3d = np.repeat(background[:, :, np.newaxis], frames, axis=2)

# Number of squares to create a rotation effect
num_squares = 4
print("Anzahl der Quadrate:", num_squares)

# Radius of rotation and center of rotation
#radius = 100
radius = 8
print("Radius der Rotation:", radius)
center_x = background_shape[1] // 2  # Horizontal center
center_y = background_shape[0] // 2  # Vertical center

# Angular rotation speed (radians per frame)
rotation_speed = 1/16 * np.pi
#rotation_speed = 0
rotation_speed_fraction = Fraction(rotation_speed/np.pi).limit_denominator()
print(f'rotation_speed (vom stimulus) = {rotation_speed_fraction}π')

# Create a single uniform square (all squares will have the same color)
#square_size = 40
square_size = 2
print("Größe der Quadrate:", square_size)

square_color = 200  # Define your desired color intensity (0-255)
print("Farbwert der Quadrate:", square_color)
square = create_uniform_square((square_size, square_size), square_color)

# Generate initial angles for each square around the circle
angles = np.linspace(0, 2 * np.pi, num_squares, endpoint=False)

# Update background by moving squares in a rotational pattern
for frame in range(frames):
    for i in range(num_squares):
        # Update angle for this square
        #current_angle = angles[i] + frame * rotation_speed  # Current rotation angle for this frame. Clockwise rotation for positive rotation_speed and counter-clockwise rotation for negative rotation_speed
        current_angle = angles[i] - frame * rotation_speed  # Current rotation angle for this frame. Clockwise rotation for negative rotation_speed and counter-clockwise rotation for positive rotation_speed

        # Calculate new x and y positions based on the angle and radius
        #square_x = int(center_x + radius * np.cos(current_angle)) % (background_shape[1] - square_size)  # Ensures square stays within width
        #square_y = int(center_y + radius * np.sin(current_angle)) % (background_shape[0] - square_size)  # Ensures square stays within height
        square_x = int(center_x + radius * np.cos(current_angle)) 
        square_y = int(center_y + radius * np.sin(current_angle)) 

        # Place the square at the new position
        background_3d[square_y:square_y + square_size, square_x:square_x + square_size, frame] = square

# Plot the background with the rotating squares (Frame 0)
plt.imshow(background_3d[:, :, 0], cmap='gray', interpolation='none')
plt.colorbar()
plt.title('Background with Rotating Squares (Frame 0)')
plt.show()

# Display another frame (e.g., Frame 60)
k = 2
plt.imshow(background_3d[:, :, k], cmap='gray', interpolation='none')
plt.colorbar()
plt.title(f'Background with Rotating Squares (Frame {k})')
plt.show()

# For later convolution, set the moving stimulus as the matrix
matrix = background_3d



#%% Rotierende Quadrate Stimulus (Uniform) mit derselben Farbe mit Uniformen Background 

import numpy as np
import matplotlib.pyplot as plt
from colorednoise import powerlaw_psd_gaussian
import scipy.ndimage

# Function to create a uniform color square
def create_uniform_square(shape, color):
    return np.full(shape, color, dtype=np.uint8)

# Dimensions of the matrix (image size)
zeilen = 32  # Height (y-axis)
spalten = 32  # Width (x-axis)
frames = 32  # Number of frames (time dimension)

# Colors
hell = 255  # White background
dunkel = 0  # Black bar

# Generate 3D array (filled with white pixels)
background_3d = np.full((zeilen, spalten, frames), fill_value=hell, dtype=np.uint8)
#background_3d = np.full((zeilen, spalten, frames), fill_value=dunkel, dtype=np.uint8)

# Number of squares to create a rotation effect
num_squares = 4
print("Anzahl der Quadrate:", num_squares)

# Radius of rotation and center of rotation
#radius = 100
radius = 8
print("Radius der Rotation:", radius)
center_x = spalten // 2  # Horizontal center
center_y = zeilen // 2  # Vertical center

# Angular rotation speed (radians per frame)
#rotation_speed = 1/32 * np.pi
#rotation_speed = -1/32 * np.pi
rotation_speed = 1/16 * np.pi
#rotation_speed = 0
rotation_speed_fraction = Fraction(rotation_speed/np.pi).limit_denominator()
print(f'rotation_speed (vom stimulus) = {rotation_speed_fraction}π')

# Create a single uniform square (all squares will have the same color)
#square_size = 40
square_size = 4
print("Größe der Quadrate:", square_size, 'x', square_size)

square_color = 0  # Define your desired color intensity (0-255)
#square_color = 255
print("Farbwert der Quadrate:", square_color)
square = create_uniform_square((square_size, square_size), square_color)

# Generate initial angles for each square around the circle
angles = np.linspace(0, 2 * np.pi, num_squares, endpoint=False)

# Update background by moving squares in a rotational pattern
for frame in range(frames):
    for i in range(num_squares):
        # Update angle for this square
        #current_angle = angles[i] + frame * rotation_speed  # Current rotation angle for this frame. Clockwise rotation for positive rotation_speed and counter-clockwise rotation for negative rotation_speed
        current_angle = angles[i] - frame * rotation_speed  # Current rotation angle for this frame. Clockwise rotation for negative rotation_speed and counter-clockwise rotation for positive rotation_speed

        # Calculate new x and y positions based on the angle and radius
        #square_x = int(center_x + radius * np.cos(current_angle)) % (background_shape[1] - square_size)  # Ensures square stays within width
        #square_y = int(center_y + radius * np.sin(current_angle)) % (background_shape[0] - square_size)  # Ensures square stays within height
        square_x = int(center_x + radius * np.cos(current_angle)) 
        square_y = int(center_y + radius * np.sin(current_angle)) 

        # Place the square at the new position
        background_3d[square_y:square_y + square_size, square_x:square_x + square_size, frame] = square

# Plot the background with the rotating squares (Frame 0)
plt.imshow(background_3d[:, :, 0], cmap='gray', interpolation='none')
plt.colorbar()
plt.title('Background with Rotating Squares (Frame 0)')
plt.show()

# Display another frame (e.g., Frame 60)
k = 2
plt.imshow(background_3d[:, :, k], cmap='gray', interpolation='none')
plt.colorbar()
plt.title(f'Background with Rotating Squares (Frame {k})')
plt.show()

# For later convolution, set the moving stimulus as the matrix
matrix = background_3d



#%% Rotierende Perlin Noise Disc mit Rotate Function von scipy.ndimage (auch für nicht-zentrierte Stimuli) mit Sharp Edge

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter
from skimage.draw import disk

#pip install noise

from noise import pnoise2

width, height = 1920, 1080

scale = 100.0             # Coarse enough for visible waves. Larger scale -> coarser noise
octaves = 2              # Multiple detail layers
persistence = 0.4        # Each octave adds less detail
lacunarity = 2.0         # Standard frequency increase
base = 0                 # Optional: use to vary look


perlin_image = np.zeros((height, width))
for y in range(height):
    for x in range(width):
        perlin_image[y, x] = pnoise2(x / scale, y / scale, octaves=octaves, lacunarity=lacunarity, persistence=persistence)  # persistence = 0.6, lacunarity = 2.0

# Normalize to [0, 1]
perlin_image = (perlin_image - perlin_image.min()) / (perlin_image.max() - perlin_image.min())


# --------------------------
# Configuration
# --------------------------
zeilen = 1080
spalten = 1920
frames = 16
rotation_speed = 1/16 * np.pi  # radians/frame
disc_radius = 100
disc_diameter = 2 * disc_radius

# --------------------------
# 1. Generate Static Perlin-style Background
# --------------------------
np.random.seed(0)
#background = gaussian_filter(np.random.rand(zeilen, spalten), sigma=10)
background = perlin_image

# --------------------------
# 2. Extract Static Square Patch for the Disc
# --------------------------
#center_y, center_x = zeilen // 2, spalten // 2
center_x = 500
center_y = 600
patch_top = center_y - disc_radius
patch_left = center_x - disc_radius
patch = background[patch_top:patch_top+disc_diameter, patch_left:patch_left+disc_diameter].copy()

# Create circular mask for disc
yy, xx = np.ogrid[:disc_diameter, :disc_diameter]
circle_mask = (yy - disc_radius)**2 + (xx - disc_radius)**2 <= disc_radius**2

# --------------------------
# 3. Create Stimulus with Static Background and Rotating Disc
# --------------------------
matrix = np.zeros((zeilen, spalten, frames), dtype=np.float32)

for i in range(frames):
    angle_deg = np.degrees(i * rotation_speed)
    rotated_patch = rotate(patch, angle=angle_deg, reshape=False, mode='nearest', order=3)
    
    # Copy background
    frame = np.copy(background)
    
    # Apply rotated disc to frame
    disc_region = frame[patch_top:patch_top+disc_diameter, patch_left:patch_left+disc_diameter]
    disc_region[circle_mask] = rotated_patch[circle_mask]
    
    matrix[:, :, i] = frame
    
# Define a region to generate the matrix in (Test 2.21, 09.04. Test 2.23, 15.04)
#col_start, col_end = 350, 650
#row_start, row_end = 450, 750
#matrix = matrix[row_start:row_end, col_start:col_end, :]


# --------------------------
# 4. Display Frames Separately
# --------------------------
#frame_indices = [0, 15, 31, 47, 63]
frame_indices = [0, 6, 12, 15]
for idx in frame_indices:
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix[:, :, idx], cmap='gray', interpolation='none', vmin=0, vmax=1)
    plt.title(f'Frame {idx}')
    plt.axis('off')
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.show()
    
    
    
#%% Rotierende Perlin Noise Disc Stimulus als Video ohne Achsen und Colorbar   

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Assuming `matrix` is your 3D matrix with shape (height, width, num_frames)
height, width, num_frames = matrix.shape

# Create a video writer 1
#fps = 10  # Frames per second
#fps = 5
fps = 30  # Original video

#fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
#fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Lossless Codec for .avi files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4

# Specify the directory and file name for saving the video
output_directory = "/Users/marius_klassen/Desktop"  # Replace with your desired folder
#output_filename = "output_video.avi"           # Specify the file name
#output_filename = f"output_video_without axes and colorbar_Pixel in y = {zeilen}, Pixel in x = {spalten}, frames = {num_frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, breite_des_balkens_x = {breite_des_balkens_x}, breite_des_balkens_y = {breite_des_balkens_y}, startposition_x = {startposition_x}, startposition_y = {startposition_y}, hell = {hell}, dunkel = {dunkel}, fps = {fps}.avi"   
#output_filename = f"output_video_without axes and colorbar_Pixel in y = {zeilen}, Pixel in x = {spalten}, frames = {num_frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, breite_des_balkens_x = {breite_des_balkens_x}, breite_des_balkens_y = {breite_des_balkens_y}, startposition_x = {startposition_x}, startposition_y = {startposition_y}, hell = {hell}, dunkel = {dunkel}, fps = {fps}.mp4"           
#output_filename = f"output_video_without axes and colorbar_Pixel in y = {height}, Pixel in x = {width}, frames = {num_frames}, rotation_speed = {max_activation_alpha/np.pi:.4f}π, fps = {fps}.mp4"  # Test 2.29 (Actionpoint 64, 24.04.25): Autokorrelation der Kernels        
#output_filename = f"output_video_x({col_start}-{col_end})_y({row_start}-{row_end})_Pixel in y = {row_end - row_start}, Pixel in x = {col_end - col_start}, frames = {frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, breite_des_balkens_x = {breite_des_balkens_x}, breite_des_balkens_y = {breite_des_balkens_y}, startposition_x = {startposition_x}, startposition_y = {startposition_y}, hell = {hell}, dunkel = {dunkel}, fps = {fps}.mp4"             

#output_filename = f"output_video_Pixel in y = {height}, Pixel in x = {width}, frames = {num_frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, disc diameter = {disc_diameter}, center_x = {center_x}, center_y = {center_y}, scale = {scale}, octaves = {octaves}, lacunarity = {lacunarity}, persistence = {persistence}, hell = 1, dunkel = 0, fps = {fps}.mp4"        
output_filename = f"output_x({col_start}-{col_end})_y({row_start}-{row_end})_zeilen = {row_end - row_start}, spalten = {col_end - col_start}, frames = {num_frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, disc diameter = {disc_diameter}, center_x = {center_x}, center_y = {center_y}, scale = {scale}, octaves = {octaves}, lacunarity = {lacunarity}, persistence = {persistence}, hell = 1, dunkel = 0, fps = {fps}.mp4"        

os.makedirs(output_directory, exist_ok=True)   # Ensure the directory exists
output_path = os.path.join(output_directory, output_filename)

# Create a video writer 2
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through the frames and save them to the video
for frame in range(num_frames):
    # Generate the frame using matplotlib
    plt.imshow(matrix[:, :, frame], cmap='gray', interpolation='none')
    plt.axis('off')  # Turn off axes for a cleaner image
    plt.tight_layout(pad=0)
    
    # Save the current frame as an image
    plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()  # Close the figure to save memory

    # Read the saved frame with OpenCV
    frame_image = cv2.imread('temp_frame.png')

    # Resize to match the video dimensions (if necessary)
    frame_image = cv2.resize(frame_image, (width, height))
    
    # Write the frame to the video
    video_writer.write(frame_image)

# Release the video writer
video_writer.release()

# Remove the temporary image file
os.remove('temp_frame.png')

print(f"Video saved at: {output_path}")



#%% Rotierende Black and White (Binary) Noise Disc mit Rotate Function von scipy.ndimage (auch für nicht-zentrierte Stimuli) mit Sharp Edge

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.draw import disk

# --------------------------
# Parameters
# --------------------------
width, height = 1920, 1080
frames = 16
rotation_speed = 1/16 * np.pi  # radians/frame
disc_radius = 100
disc_diameter = 2 * disc_radius
density = 0.02  # Proportion of white pixels (0 to 1)

center_x = 500
center_y = 600
patch_top = center_y - disc_radius
patch_left = center_x - disc_radius

# --------------------------
# Binary Random Background
# --------------------------
np.random.seed(0)
background = (np.random.rand(height, width) < density).astype(np.float64)

# --------------------------
# Extract Square Patch for the Disc
# --------------------------
patch = background[patch_top:patch_top+disc_diameter, patch_left:patch_left+disc_diameter].copy()

# Create circular mask
yy, xx = np.ogrid[:disc_diameter, :disc_diameter]
circle_mask = (yy - disc_radius)**2 + (xx - disc_radius)**2 <= disc_radius**2

# --------------------------
# Create Stimulus with Rotating Binary Disc
# --------------------------
matrix = np.zeros((height, width, frames), dtype=np.float64)

for i in range(frames):
    angle_deg = np.degrees(i * rotation_speed)
    
    # Rotate patch using nearest neighbor to preserve binary values
    rotated_patch = rotate(patch, angle=angle_deg, reshape=False, mode='nearest', order=3)
    
    # Threshold to ensure binary result after rotation
    rotated_patch = (rotated_patch > 0.5).astype(np.float64)
    
    # Copy background and insert rotated disc
    frame = background.copy()
    disc_region = frame[patch_top:patch_top+disc_diameter, patch_left:patch_left+disc_diameter]
    disc_region[circle_mask] = rotated_patch[circle_mask]
    
    matrix[:, :, i] = frame
    
# Define a region to generate the matrix in (Test 2.21, 09.04. Test 2.23, 15.04)
#col_start, col_end = 350, 650
#row_start, row_end = 450, 750
#col_start, col_end = 100, 116
#row_start, row_end = 100, 116
#matrix = matrix[row_start:row_end, col_start:col_end, :]


# --------------------------
# Display Selected Frames
# --------------------------
frame_indices = [0, 6, 12, 15]
for idx in frame_indices:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix[:, :, idx]*255, cmap='gray', interpolation='none', vmin=0, vmax=1)
    ax.set_title(f'Frame {idx}')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Intensity')
    plt.tight_layout()
    plt.show()



#%% Rotierende Black and White (Binary) Noise Disc als Video ohne Achsen und Colorbar 

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Assuming `matrix` is your 3D matrix with shape (height, width, num_frames)
height, width, num_frames = matrix.shape

# Create a video writer 1
#fps = 10  # Frames per second
#fps = 5
fps = 30  # Original video

#fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
#fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Lossless Codec for .avi files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4

# Specify the directory and file name for saving the video
output_directory = "/Users/marius_klassen/Desktop"  # Replace with your desired folder
#output_filename = "output_video.avi"           # Specify the file name
#output_filename = f"output_video_without axes and colorbar_Pixel in y = {zeilen}, Pixel in x = {spalten}, frames = {num_frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, breite_des_balkens_x = {breite_des_balkens_x}, breite_des_balkens_y = {breite_des_balkens_y}, startposition_x = {startposition_x}, startposition_y = {startposition_y}, hell = {hell}, dunkel = {dunkel}, fps = {fps}.avi"   
#output_filename = f"output_video_without axes and colorbar_Pixel in y = {zeilen}, Pixel in x = {spalten}, frames = {num_frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, breite_des_balkens_x = {breite_des_balkens_x}, breite_des_balkens_y = {breite_des_balkens_y}, startposition_x = {startposition_x}, startposition_y = {startposition_y}, hell = {hell}, dunkel = {dunkel}, fps = {fps}.mp4"           
#output_filename = f"output_video_without axes and colorbar_Pixel in y = {height}, Pixel in x = {width}, frames = {num_frames}, rotation_speed = {max_activation_alpha/np.pi:.4f}π, fps = {fps}.mp4"  # Test 2.29 (Actionpoint 64, 24.04.25): Autokorrelation der Kernels        
#output_filename = f"output_video_x({col_start}-{col_end})_y({row_start}-{row_end})_Pixel in y = {row_end - row_start}, Pixel in x = {col_end - col_start}, frames = {frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, breite_des_balkens_x = {breite_des_balkens_x}, breite_des_balkens_y = {breite_des_balkens_y}, startposition_x = {startposition_x}, startposition_y = {startposition_y}, hell = {hell}, dunkel = {dunkel}, fps = {fps}.mp4"             

output_filename = f"output_video_Pixel in y = {height}, Pixel in x = {width}, frames = {num_frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, disc diameter = {disc_diameter}, center_x = {center_x}, center_y = {center_y}, density = {density}, hell = 1, dunkel = 0, fps = {fps}.mp4"        
#output_filename = f"output_x({col_start}-{col_end})_y({row_start}-{row_end})_zeilen = {row_end - row_start}, spalten = {col_end - col_start}, frames = {num_frames}, rotation_speed = {rotation_speed/np.pi:.4f}π, disc diameter = {disc_diameter}, center_x = {center_x}, center_y = {center_y}, density = {density}, hell = 1, dunkel = 0, fps = {fps}.mp4"        

os.makedirs(output_directory, exist_ok=True)   # Ensure the directory exists
output_path = os.path.join(output_directory, output_filename)

# Create a video writer 2
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through the frames and save them to the video
for frame in range(num_frames):
    # Generate the frame using matplotlib
    plt.imshow(matrix[:, :, frame], cmap='gray', interpolation='none')
    plt.axis('off')  # Turn off axes for a cleaner image
    plt.tight_layout(pad=0)
    
    # Save the current frame as an image
    plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()  # Close the figure to save memory

    # Read the saved frame with OpenCV
    frame_image = cv2.imread('temp_frame.png')

    # Resize to match the video dimensions (if necessary)
    frame_image = cv2.resize(frame_image, (width, height))
    
    # Write the frame to the video
    video_writer.write(frame_image)

# Release the video writer
video_writer.release()

# Remove the temporary image file
os.remove('temp_frame.png')

print(f"Video saved at: {output_path}")



#%% Twisted / Rotational Sine Kernel mit 3D Gauß im Gabor. Mit symmetrischem Binning (06.02.2025)  
# Negative alpha -> clockwise rotation, positive alpha -> counter-clockwise rotation

import numpy as np
import matplotlib.pyplot as plt

def generate_rotational_sine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha):
    """
    Generate a 3D Gabor kernel sensitive to 2D rotation in the xy-plane
    over the z-dimension (representing frames).
    
    Parameters:
        size_x (int): Size of the kernel along x-axis.
        size_y (int): Size of the kernel along y-axis.
        size_z (int): Number of frames along z-axis.
        sigma (float): Gaussian envelope spread.
        lambd (float): Wavelength of the sinusoidal factor.
        alpha (float): Twisting parameter controlling rotational sensitivity.
        
    Returns:
        np.array: 3D rotational Gabor kernel.
    """
    # Initialize the 3D kernel
    rotational_sine_kernel = np.zeros((size_x, size_y, size_z))
    
    # Calculate the center of the kernel
    center_x, center_y, center_z = size_x // 2, size_y // 2, size_z // 2
    
    # Iterate over each point in the 3D kernel
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                # Shift coordinates to center the kernel
                x_prime = x - center_x + 0.5
                y_prime = y - center_y + 0.5
                z_prime = z - center_z + 0.5

                # Calculate radial distance and angle in the xy-plane
                r = np.sqrt(x_prime**2 + y_prime**2)
                theta = np.arctan2(y_prime, x_prime)  # Angle in the xy-plane

                # Apply the rotation transformation with z as time
                theta_rotated = theta + alpha * z  # z acts as the temporal dimension. Negative alpha -> clockwise rotation, positive alpha -> counter-clockwise rotation

                # Convert back to transformed Cartesian coordinates in the xy-plane
                x_rotated = r * np.cos(theta_rotated)  # Die v_tangential = w * r Beziehung (Proportionalität von v_tangential zu r) steckt hier drin 
                y_rotated = r * np.sin(theta_rotated)  # Die v_tangential = w * r Beziehung (Proportionalität von v_tangential zu r) steckt hier drin 
                
                # Gabor function calculation
                gaussian = 1 / ((np.sqrt(2 * np.pi) * sigma)**3) * \
                           np.exp(-(x_prime**2 + y_prime**2 + z_prime**2) / (2 * sigma**2))  #  # Ist egal, ob hier "x_prime**2 + y_prime**2 + z_prime**2" steht oder "x_rotated**2 + y_rotated**2 + z_rotated**2", ist dasselbe (nämlich = "r**2")
                
                # Sinusoidal function with the rotation in the xy-plane
                sinusoidal = np.sin(2 * np.pi * x_rotated / lambd)
                gabor_value = gaussian * sinusoidal
                
                # Assign the value to the kernel at each (x, y, z) position
                rotational_sine_kernel[y, x, z] = gabor_value

    # Ensure the overall sum of elements is 0 for proper filtering
    #mean_value = np.mean(rotational_sine_kernel)
    #rotational_sine_kernel -= mean_value
    
    # Normalize the sum of absolute values to 1
    #abs_sum = np.sum(np.abs(rotational_sine_kernel))
    #if abs_sum != 0:
    #    rotational_sine_kernel /= abs_sum
        
    #rotational_sine_kernel *= 1e2  # Scale absolute sum to a value of 100 instead of 1
    
    return rotational_sine_kernel



#%% Twisted / Rotational Sine Kernel mit 3D Gauß im Gabor. Mit symmetrischem Binning (06.02.2025). Mit Phase als Parameter (09.04.2025)
# Negative alpha -> clockwise rotation, positive alpha -> counter-clockwise rotation
# Negative phase -> clockwise rotation, positive phase -> counter-clockwise rotation

import numpy as np
import matplotlib.pyplot as plt

def generate_rotational_sine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha, phase):
    """
    Generate a 3D Gabor kernel with phase, sensitive to 2D rotation in the xy-plane over the z-dimension.

    Parameters:
        size_x (int): Size of the kernel along x-axis.
        size_y (int): Size of the kernel along y-axis.
        size_z (int): Number of frames along z-axis.
        sigma (float): Gaussian envelope spread.
        lambd (float): Wavelength of the sinusoidal factor.
        alpha (float): Twisting parameter controlling rotational sensitivity.
        phase (float): Initial angular offset applied to the rotational orientation in the xy-plane. 

    Returns:
        np.array: 3D rotational Gabor kernel with phase.
    """
    # Initialize the 3D kernel
    rotational_sine_kernel = np.zeros((size_x, size_y, size_z))
    
    # Calculate the center of the kernel
    center_x, center_y, center_z = size_x // 2, size_y // 2, size_z // 2
    
    # Iterate over each point in the 3D kernel
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                # Shift coordinates to center the kernel
                x_prime = x - center_x + 0.5
                y_prime = y - center_y + 0.5
                z_prime = z - center_z + 0.5

                # Calculate radial distance and angle in the xy-plane
                r = np.sqrt(x_prime**2 + y_prime**2)
                theta = np.arctan2(y_prime, x_prime)  # Angle in the xy-plane

                # Apply the rotation transformation with z as time
                theta_rotated = theta + alpha * z + phase  # z acts as the temporal dimension. Negative alpha -> clockwise rotation, positive alpha -> counter-clockwise rotation. phase is the offset angle at the starting point z = 0. Negative phase -> clockwise rotation, positive phase -> counter-clockwise rotation

                # Convert back to transformed Cartesian coordinates in the xy-plane
                x_rotated = r * np.cos(theta_rotated)  # Die v_tangential = w * r Beziehung (Proportionalität von v_tangential zu r) steckt hier drin 
                y_rotated = r * np.sin(theta_rotated)  # Die v_tangential = w * r Beziehung (Proportionalität von v_tangential zu r) steckt hier drin 
                
                # Gabor function calculation
                gaussian = 1 / ((np.sqrt(2 * np.pi) * sigma)**3) * \
                           np.exp(-(x_prime**2 + y_prime**2 + z_prime**2) / (2 * sigma**2))  #  # Ist egal, ob hier "x_prime**2 + y_prime**2 + z_prime**2" steht oder "x_rotated**2 + y_rotated**2 + z_rotated**2", ist dasselbe (nämlich = "r**2")
                
                # Sinusoidal function with the rotation in the xy-plane
                sinusoidal = np.sin(2 * np.pi * x_rotated / lambd )  
                gabor_value = gaussian * sinusoidal
                
                # Assign the value to the kernel at each (x, y, z) position
                rotational_sine_kernel[y, x, z] = gabor_value

    # Ensure the overall sum of elements is 0 for proper filtering
    mean_value = np.mean(rotational_sine_kernel)
    rotational_sine_kernel -= mean_value
    
    # Normalize the sum of absolute values to 1
    abs_sum = np.sum(np.abs(rotational_sine_kernel))
    if abs_sum != 0:
        rotational_sine_kernel /= abs_sum
        
    #rotational_sine_kernel *= 1e2  # Scale absolute sum to a value of 100 instead of 1
    
    return rotational_sine_kernel



#%% Twisted Sine Kernel mit 3D Gauß im Gabor. Mit symmetrischem Binning (06.02.2025). Mit Phase als Parameter (09.04.2025). Mit Translation (28.05.25)
# Negative alpha -> clockwise rotation, positive alpha -> counter-clockwise rotation
# Negative phase -> clockwise rotation, positive phase -> counter-clockwise rotation

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

def generate_rotational_sine_gabor_kernel_with_translation(size_x, size_y, size_z, sigma, lambd, alpha, phase, vx):
    rotational_sine_kernel_translated = np.zeros((size_x, size_y, size_z))
    center_x, center_y, center_z = size_x // 2, size_y // 2, size_z // 2

    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                # Shift coordinates to center the kernel
                x_prime = x - center_x + 0.5
                y_prime = y - center_y + 0.5
                z_prime = z - center_z + 0.5

                # Apply full spatial translation to the coordinates
                z_offset = z - center_z  # So that z=8 → 0
                x_shifted = x_prime - vx * z_offset

                # Compute radial distance and angle in rotated space
                r = np.sqrt(x_shifted**2 + y_prime**2)
                theta = np.arctan2(y_prime, x_shifted)
                theta_rotated = theta + alpha * z + phase

                # Rotate
                x_rotated = r * np.cos(theta_rotated)
                y_rotated = r * np.sin(theta_rotated)

                # Gabor calculation
                gaussian = 1 / ((np.sqrt(2 * np.pi) * sigma)**3) * \
                           np.exp(-(x_shifted**2 + y_prime**2 + z_prime**2) / (2 * sigma**2))
                           
                sinusoidal = np.sin(2 * np.pi * x_rotated / lambd)
                gabor_value = gaussian * sinusoidal

                rotational_sine_kernel_translated[y, x, z] = gabor_value

    # Zero-mean and normalize
    rotational_sine_kernel_translated -= np.mean(rotational_sine_kernel_translated)
    abs_sum = np.sum(np.abs(rotational_sine_kernel_translated))
    if abs_sum != 0:
        rotational_sine_kernel_translated /= abs_sum

    return rotational_sine_kernel_translated



#%% Twisted / Rotational Cosine Kernel mit 3D Gauß im Gabor. Mit symmetrischem Binning (06.02.2025) 
# Negative alpha -> clockwise rotation, positive alpha -> counter-clockwise rotation

import numpy as np
import matplotlib.pyplot as plt

def generate_rotational_cosine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha):
    """
    Generate a 3D Gabor kernel sensitive to 2D rotation in the xy-plane
    over the z-dimension (representing frames).
    
    Parameters:
        size_x (int): Size of the kernel along x-axis.
        size_y (int): Size of the kernel along y-axis.
        size_z (int): Number of frames along z-axis.
        sigma (float): Gaussian envelope spread.
        lambd (float): Wavelength of the sinusoidal factor.
        alpha (float): Twisting parameter controlling rotational sensitivity.
        
    Returns:
        np.array: 3D rotational Gabor kernel.
    """
    # Initialize the 3D kernel
    rotational_cosine_kernel = np.zeros((size_x, size_y, size_z))
    
    # Calculate the center of the kernel
    center_x, center_y, center_z = size_x // 2, size_y // 2, size_z // 2
    
    # Iterate over each point in the 3D kernel
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                # Shift coordinates to center the kernel
                x_prime = x - center_x + 0.5
                y_prime = y - center_y + 0.5
                z_prime = z - center_z + 0.5

                # Calculate radial distance and angle in the xy-plane
                r = np.sqrt(x_prime**2 + y_prime**2)
                theta = np.arctan2(y_prime, x_prime)  # Angle in the xy-plane

                # Apply the rotation transformation with z as time
                theta_rotated = theta + alpha * z  # z acts as the temporal dimension. Negative alpha -> clockwise rotation, positive alpha -> counter-clockwise rotation

                # Convert back to transformed Cartesian coordinates in the xy-plane
                x_rotated = r * np.cos(theta_rotated)  # Die v_tangential = w * r Beziehung (Proportionalität von v_tangential zu r) steckt hier drin 
                y_rotated = r * np.sin(theta_rotated)  # Die v_tangential = w * r Beziehung (Proportionalität von v_tangential zu r) steckt hier drin 

                # Gabor function calculation
                gaussian = 1 / ((np.sqrt(2 * np.pi) * sigma)**3) * \
                           np.exp(-(x_prime**2 + y_prime**2 + z_prime**2) / (2 * sigma**2))  #  # Ist egal, ob hier "x_prime**2 + y_prime**2 + z_prime**2" steht oder "x_rotated**2 + y_rotated**2 + z_rotated**2", ist dasselbe (nämlich = "r**2")
                
                # Cosinusoidal function with the rotation in the xy-plane
                cosinusoidal = np.cos(2 * np.pi * x_rotated / lambd)
                gabor_value = gaussian * cosinusoidal
                
                # Assign the value to the kernel at each (x, y, z) position
                rotational_cosine_kernel[y, x, z] = gabor_value

    # Ensure the overall sum of elements is 0 for proper filtering
    #mean_value = np.mean(rotational_cosine_kernel)
    #rotational_cosine_kernel -= mean_value
    
    # Normalize the sum of absolute values to 1
    #abs_sum = np.sum(np.abs(rotational_cosine_kernel))
    #if abs_sum != 0:
    #    rotational_cosine_kernel /= abs_sum
        
    #rotational_cosine_kernel *= 1e2  # Scale absolute sum to a value of 100 instead of 1
    
    return rotational_cosine_kernel



#%% Twisted / Rotational Cosine Kernel mit 3D Gauß im Gabor. Mit symmetrischem Binning (06.02.2025). Mit Phase als Parameter (09.04.2025)
# Negative alpha -> clockwise rotation, positive alpha -> counter-clockwise rotation
# Negative phase -> clockwise rotation, positive phase -> counter-clockwise rotation

import numpy as np
import matplotlib.pyplot as plt

def generate_rotational_cosine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha, phase):
    """
    Generate a 3D Gabor kernel with phase, sensitive to 2D rotation in the xy-plane over the z-dimension.

    Parameters:
        size_x (int): Size of the kernel along x-axis.
        size_y (int): Size of the kernel along y-axis.
        size_z (int): Number of frames along z-axis.
        sigma (float): Gaussian envelope spread.
        lambd (float): Wavelength of the sinusoidal factor.
        alpha (float): Twisting parameter controlling rotational sensitivity.
        phase (float): Initial angular offset applied to the rotational orientation in the xy-plane. 

    Returns:
        np.array: 3D rotational Gabor kernel with phase.
    """
    # Initialize the 3D kernel
    rotational_cosine_kernel = np.zeros((size_x, size_y, size_z))
    
    # Calculate the center of the kernel
    center_x, center_y, center_z = size_x // 2, size_y // 2, size_z // 2
    
    # Iterate over each point in the 3D kernel
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                # Shift coordinates to center the kernel
                x_prime = x - center_x + 0.5
                y_prime = y - center_y + 0.5
                z_prime = z - center_z + 0.5

                # Calculate radial distance and angle in the xy-plane
                r = np.sqrt(x_prime**2 + y_prime**2)
                theta = np.arctan2(y_prime, x_prime)  # Angle in the xy-plane

                # Apply the rotation transformation with z as time
                theta_rotated = theta + alpha * z + phase  # z acts as the temporal dimension. Negative alpha -> clockwise rotation, positive alpha -> counter-clockwise rotation. Negative phase -> clockwise rotation, positive phase -> counter-clockwise rotation

                # Convert back to transformed Cartesian coordinates in the xy-plane
                x_rotated = r * np.cos(theta_rotated)  # Die v_tangential = w * r Beziehung (Proportionalität von v_tangential zu r) steckt hier drin 
                y_rotated = r * np.sin(theta_rotated)  # Die v_tangential = w * r Beziehung (Proportionalität von v_tangential zu r) steckt hier drin 

                # Gabor function calculation
                gaussian = 1 / ((np.sqrt(2 * np.pi) * sigma)**3) * \
                           np.exp(-(x_prime**2 + y_prime**2 + z_prime**2) / (2 * sigma**2))  #  # Ist egal, ob hier "x_prime**2 + y_prime**2 + z_prime**2" steht oder "x_rotated**2 + y_rotated**2 + z_rotated**2", ist dasselbe (nämlich = "r**2")
                
                # Cosinusoidal function with the rotation in the xy-plane
                cosinusoidal = np.cos(2 * np.pi * x_rotated / lambd) 
                gabor_value = gaussian * cosinusoidal
                
                # Assign the value to the kernel at each (x, y, z) position
                rotational_cosine_kernel[y, x, z] = gabor_value

    # Ensure the overall sum of elements is 0 for proper filtering
    mean_value = np.mean(rotational_cosine_kernel)
    rotational_cosine_kernel -= mean_value
    
    # Normalize the sum of absolute values to 1
    abs_sum = np.sum(np.abs(rotational_cosine_kernel))
    if abs_sum != 0:
        rotational_cosine_kernel /= abs_sum
        
    #rotational_cosine_kernel *= 1e2  # Scale absolute sum to a value of 100 instead of 1
    
    return rotational_cosine_kernel



#%% Twisted Cosine Kernel mit 3D Gauß im Gabor. Mit symmetrischem Binning (06.02.2025). Mit Phase als Parameter (09.04.2025). Mit Translation (28.05.25)
# Negative alpha -> clockwise rotation, positive alpha -> counter-clockwise rotation
# Negative phase -> clockwise rotation, positive phase -> counter-clockwise rotation

import numpy as np

def generate_rotational_cosine_gabor_kernel_with_translation(size_x, size_y, size_z, sigma, lambd, alpha, phase, vx):
    rotational_cosine_kernel_translated = np.zeros((size_x, size_y, size_z))
    center_x, center_y, center_z = size_x // 2, size_y // 2, size_z // 2

    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                # Shift coordinates to center the kernel
                x_prime = x - center_x + 0.5
                y_prime = y - center_y + 0.5
                z_prime = z - center_z + 0.5

                # Apply full spatial translation to the coordinates
                z_offset = z - center_z  # So that z=8 → 0
                x_shifted = x_prime - vx * z_offset

                # Compute radial distance and angle in rotated space
                r = np.sqrt(x_shifted**2 + y_prime**2)
                theta = np.arctan2(y_prime, x_shifted)
                theta_rotated = theta + alpha * z + phase

                # Rotate
                x_rotated = r * np.cos(theta_rotated)
                y_rotated = r * np.sin(theta_rotated)

                # Gabor calculation
                gaussian = 1 / ((np.sqrt(2 * np.pi) * sigma)**3) * \
                           np.exp(-(x_shifted**2 + y_prime**2 + z_prime**2) / (2 * sigma**2))
                           
                cosinusoidal = np.cos(2 * np.pi * x_rotated / lambd)
                gabor_value = gaussian * cosinusoidal

                rotational_cosine_kernel_translated[y, x, z] = gabor_value

    # Zero-mean and normalize
    rotational_cosine_kernel_translated -= np.mean(rotational_cosine_kernel_translated)
    abs_sum = np.sum(np.abs(rotational_cosine_kernel_translated))
    if abs_sum != 0:
        rotational_cosine_kernel_translated /= abs_sum

    return rotational_cosine_kernel_translated



#%% Parameters for the twisted Gabor kernel

from fractions import Fraction

# Parameters for the Gabor kernel
size = 32  # Kernel size in px x px x frames 
size_x = 32  # Kernel size in px in x-direction
size_y = 32  # Kernel size in px in y-direction
size_z = 32  # Kernel size in px in z-direction (Frames i.e. Time)
#size = 16 
#size_x = 16  
#size_y = 16 
#size_z = 16 

# Define possible cubic sizes
#size_values = [16, 32, 64]  # For 16 x 16 x 16, 32 x 32 x 32 and 64 x 64 x 64 Kernels
#size_values = [16]  # For 16 x 16 x 16, 32 x 32 x 32 and 64 x 64 x 64 Kernels

#sigma = 5.0  # Standard Deviation of the Gauß, gives the width of the Gauß
#sigma = 2.5  # Test vom Stand 30.10
sigma = 3.106  # Test vom Stand 25.12 (Actionpoint 38). For 16 x 16 x 16 Kernels

# Define sigma values for different kernel sizes
#sigma_values = {16: 3.106, 32: 6.211, 64: 12.422}

# Choose the lambda (lambd) values 
#lambd_values = [16.0, 32.0]  # Spatial frequency or rather wavelength
#lambd_values = [8.0, 16.0, 32.0]
#lambd_values = [8.0, 16.0]
#lambd_values = [8.0, 16.0, 24.0]
#lambd_values = [8.0, 8.0]
#lambd_values = [4.0, 8.0, 16.0]
#lambd_values = [1.0, 2.0, 4.0, 8.0]
lambd_values = [4.0, 6.4, 8.0]
#print("Lambda Values:", lambd_values)

# Choose the alpha values in multiples of pi. Positive sign: counter-clockwise rotation. Negative sign: clockwise rotation
#alpha_values = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # Rotation speed / Angular velocity in x-y plane
#alpha_values = [0.0, 0.25, 0.5, 0.75]
alpha_values = [0, -1/64, -1/32, -1/16, -1/8, -1/4, -1/3, 1/64, 1/32, 1/16, 1/8, 1/4, 1/3] 
#alpha_values = [0, -1/64, -1/32, -1/16, -1/8, -1/4, 1/64, 1/32, 1/16, 1/8, 1/4] 
#alpha_values = [0, -1/60, -1/30, -1/15, -1/8, -1/4, -1/3, 1/60, 1/30, 1/15, 1/8, 1/4, 1/3]  # Für Rot Geschw des Stimulus = 1/15π 

# Convert each value in alpha_values to a Fraction
alpha_values_fractions = [Fraction(value).limit_denominator() for value in alpha_values]

# Convert each fraction to a decimal in multiples of pi
alpha_values_decimals = [float(value) for value in alpha_values]  

# Choose the actual alpha values not in multiples of pi
alpha_values = [value * np.pi for value in alpha_values] 

# Choose the phase values in multiples of pi. Positive sign: counter-clockwise rotation. Negative sign: clockwise rotation
phase_values = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
#phase_values = [0]
#phase_values = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875]
#phase_values = [0, 4/15, 8/15, 12/15, 16/15, 20/15, 24/15, 28/15]  # Für Rot Geschw des Stimulus = 1/15π 

# Convert each value in phase_values to a Fraction
phase_values_fractions = [Fraction(value).limit_denominator() for value in phase_values]

# Choose the actual phase values not in multiples of pi
phase_values = [value * np.pi for value in phase_values] 

print(f"Length of Lambda Values: {len(lambd_values)}")    # Print the length of lambd_values  
print(f"Length of Alpha Values: {len(alpha_values)}")    # Print the length of alpha_values    
#print(f"Length of Sizes: {len(size_values)}")    # Print the length of size_values  
#print(f"Length of Sigma Values: {len(sigma_values)}")    # Print the length of sigma_values  
print(f"Length of Phase Values: {len(phase_values)}")    # Print the length of phase_values  



#%% Print Parameters for the twisted Gabor Kernel

import numpy as np

# Function to format values in multiples of π
def format_pi_multiples(values):
    return [f"{round(value / np.pi, 3)}π" for value in values]

# Function to print parameters
def print_parameters():
    print("===== Parameters for the Rotational Gabor Kernel =====")
    print(f"Kernel size (size): {size}")
    print(f"Kernel size in x-direction (size_x): {size_x}")
    print(f"Kernel size in y-direction (size_y): {size_y}")
    print(f"Kernel size in z-direction (size_z): {size_z}")
    #print(f"Sigma (sigma): {sigma:.3f}")
    #print(f"Sigma values (sigma_values): {sigma_values}")
    #print(f"Number of Sigma values: {len(sigma_values)}")
    print(f"Lambda values (lambd_values): {lambd_values}")
    print(f"Number of Lambda values: {len(lambd_values)}")
    #print(f"Alpha values (alpha_values in multiples of π): {format_pi_multiples(alpha_values)}")
    print(f"Alpha values (alpha_values in fractions of π): {[f'{frac}π' for frac in alpha_values_fractions]}")
    print(f"Number of Alpha values: {len(alpha_values)}")
    #print(f"Kernel sizes (size_values): {size_values}")
    #print(f"Number of Kernel sizes: {len(size_values)}")
    print(f"Phase values (phase_values in fractions of π): {[f'{frac}π' for frac in phase_values_fractions]}")
    print(f"Number of Phase values: {len(phase_values)}")
    print("===========================================")

# Call the function
print_parameters()



#%% Convolutions with Twisted Sine and Twisted Cosine Kernels on Stimulus

from scipy.signal import convolve


# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 64 und 16 x 16 x 16 Kernels
zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels bei 32 x 32 x 32 Stimulus Array
zeilen = 1    # = n_rows of Input Array - k_rows of Kernel Array + 1
spalten = 1   # = n_columns of Input Array - k_columns of Kernel Array + 1

# Initialize arrays to store convolution results for sine and cosine kernels
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), zeilen, spalten))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), zeilen, spalten))

# Iterate over each kernel and perform convolution for both sine and cosine kernels
for k, lambd in enumerate(lambd_values):
    for j, alpha in enumerate(alpha_values):
        
        sine_kernel = generate_rotational_sine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha)
        cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha)
        sine_convolution = convolve(matrix, sine_kernel, mode='valid', method='fft')  # Flips the kernel in all axes before convolution calculation
        cosine_convolution = convolve(matrix, cosine_kernel, mode='valid', method='fft')  # Flips the kernel in all axes before convolution calculation
            
        # Remove singleton dimension from the result array
        sine_convolution = np.squeeze(sine_convolution)
        cosine_convolution = np.squeeze(cosine_convolution)
            
        sine_convolutions[k, j] = sine_convolution  # Assign the result. sine_convolutions[k, j] = sine_convolutions[k, j, :, :]
        cosine_convolutions[k, j] = cosine_convolution  # Assign the result. cosine_convolutions[k, j] = cosine_convolutions[k, j, :, :]
            
        
        
#%% Convolutions with Twisted Sine and Twisted Cosine Kernels on Stimulus using Correlation instead of Convolution from Scipy.Signal

from scipy.signal import correlate


# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 16 und 16 x 16 x 64 Kernels
zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels
#zeilen = 1049    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1889   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels bei 32 x 32 x 32 Stimulus Array
#zeilen = 1    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1   # = n_columns of Input Array - k_columns of Kernel Array + 1

# Initialize arrays to store convolution results for sine and cosine kernels
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), zeilen, spalten))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), zeilen, spalten))

# Iterate over each kernel and perform convolution for both sine and cosine kernels
for k, lambd in enumerate(lambd_values):
    for j, alpha in enumerate(alpha_values):
        
        sine_kernel = generate_rotational_sine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha)
        cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha)
        sine_convolution = correlate(matrix, sine_kernel, mode='valid', method='fft')  # Does not flip the kernel in all axes before convolution calculation
        cosine_convolution = correlate(matrix, cosine_kernel, mode='valid', method='fft')  # Does not flip the kernel in all axes before convolution calculation
            
        # Remove singleton dimension from the result array
        sine_convolution = np.squeeze(sine_convolution)
        cosine_convolution = np.squeeze(cosine_convolution)
            
        sine_convolutions[k, j] = sine_convolution  # Assign the result. sine_convolutions[k, j] = sine_convolutions[k, j, :, :]
        cosine_convolutions[k, j] = cosine_convolution  # Assign the result. cosine_convolutions[k, j] = cosine_convolutions[k, j, :, :]
            
        
        
#%% Convolutions with Twisted Sine and Twisted Cosine Kernels on Stimulus using Correlation instead of Convolution from Scipy.Signal. With Kernel Phase as separate Parameter Dimension

from scipy.signal import correlate


# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 16 und 16 x 16 x 64 Kernels
#zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels
#zeilen = 1049    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1889   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels bei 32 x 32 x 32 Stimulus Array
#zeilen = 1    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 16 Kernels on 300 x 300 Input Array (not Full HD like all the above)
#zeilen = 285    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 285   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels on 300 x 300 Input Array (not Full HD like all the above)
#zeilen = 269    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 269   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 64 x 64 x 64 Kernels on 300 x 300 Input Array (not Full HD like all the above)
zeilen = 237    # = n_rows of Input Array - k_rows of Kernel Array + 1
spalten = 237   # = n_columns of Input Array - k_columns of Kernel Array + 1

# Initialize arrays to store convolution results for sine and cosine kernels
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(phase_values), zeilen, spalten))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(phase_values), zeilen, spalten))

# Iterate over each kernel and perform convolution for both sine and cosine kernels
for k, lambd in enumerate(lambd_values):
    for j, alpha in enumerate(alpha_values):
        for i, phase in enumerate(phase_values):
            
            sine_kernel = generate_rotational_sine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha, phase)
            cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha, phase)
            
            sine_convolution = correlate(matrix, sine_kernel, mode='valid', method='fft')  # Does not flip the kernel in all axes before convolution calculation
            cosine_convolution = correlate(matrix, cosine_kernel, mode='valid', method='fft')  # Does not flip the kernel in all axes before convolution calculation
            
            # Remove singleton dimension from the result array
            sine_convolution = np.squeeze(sine_convolution)
            cosine_convolution = np.squeeze(cosine_convolution)

            sine_convolutions[k, j, i] = sine_convolution   # Assign the result. sine_convolutions[k, j, i] = sine_convolutions[k, j, i, :, :]
            cosine_convolutions[k, j, i] = cosine_convolution  # Assign the result. cosine_convolutions[k, j] = cosine_convolutions[k, j, i, :, :]
            
        
        
#%% Convolutions with Twisted Sine and Twisted Cosine Kernels on Stimulus using Correlation instead of Convolution from Scipy.Signal. With Kernel Size as separate Parameter Dimension Using Non-qubic Kernels without Including different Sigmas

from scipy.signal import correlate


# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 16 und 16 x 16 x 64 Kernels
zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels bei 32 x 32 x 32 Stimulus Array
#zeilen = 1    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1   # = n_columns of Input Array - k_columns of Kernel Array + 1

# Initialize arrays to store convolution results for different sizes
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), zeilen, spalten))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), zeilen, spalten))

# Iterate over lambda and alpha first
for k, lambd in enumerate(lambd_values):
    for j, alpha in enumerate(alpha_values):
        for s, size in enumerate(size_values):  

            # Generate 3D sine and cosine kernels with cubic size
            sine_kernel = generate_rotational_sine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)
            cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)
            
            # Perform convolution
            sine_convolution = correlate(matrix, sine_kernel, mode='valid', method='fft')
            cosine_convolution = correlate(matrix, cosine_kernel, mode='valid', method='fft')
                
            # Remove singleton dimension from the result array
            sine_convolution = np.squeeze(sine_convolution)
            cosine_convolution = np.squeeze(cosine_convolution)
                
            # Store results in the appropriate index
            sine_convolutions[k, j, s] = sine_convolution  
            cosine_convolutions[k, j, s] = cosine_convolution  
            
            
        
#%% Convolutions with Twisted Sine and Twisted Cosine Kernels on Stimulus using Correlation instead of Convolution from Scipy.Signal. With Kernel Size as separate Parameter Dimension Using Qubic Kernels with Including different Sigmas (1). Without Padding
### DAS SLICING KANN ICH AUCH EINMAL VORHER AUßERHALB DER CONVOLUTION LOOPS MACHEN UND ABSPEICHERN ALS LISTE, ARRAY O.Ä. UND IM LOOP DANN EINFACH DARAUS NEHMEN
### FEHLER BEIM KOMPILIEREN, WEIL PADDING FEHLT ###

import numpy as np
from scipy.signal import correlate


# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 16 und 16 x 16 x 64 Kernels
#zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels bei 32 x 32 x 32 Stimulus Array
#zeilen = 1    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1   # = n_columns of Input Array - k_columns of Kernel Array + 1

# Define the center slice function to extract the central slice from the third dimension
def extract_central_slice(matrix, kernel_depth):
    """ Extracts a central slice from the third dimension of the matrix """
    center_idx = matrix.shape[2] // 2  # Center frame index (64 // 2 = 32)
    start_idx = center_idx - (kernel_depth // 2)
    end_idx = start_idx + kernel_depth
    return matrix[:, :, start_idx:end_idx]  # Extracted slice. Python slicing is exclusive on the upper bound, here: "end_idx"

# Initialize arrays to store convolution results
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), zeilen, spalten))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), zeilen, spalten))

# Iterate over lambda and alpha first
for k, lambd in enumerate(lambd_values):
    for j, alpha in enumerate(alpha_values):
        for s, size in enumerate(size_values):  

            # Fetch sigma dynamically based on kernel size
            sigma = sigma_values[size]

            # Generate 3D sine and cosine kernels
            sine_kernel = generate_rotational_sine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)
            cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)

            # Extract a slice if kernel depth is smaller than 64
            if size < 64:
                matrix_slice = extract_central_slice(matrix, size)
            else:
                matrix_slice = matrix  # Use full matrix for 64×64×64 kernel

            # Perform convolution
            sine_convolution = correlate(matrix_slice, sine_kernel, mode='valid', method='fft')
            cosine_convolution = correlate(matrix_slice, cosine_kernel, mode='valid', method='fft')

            # Remove singleton dimension from the result array
            sine_convolution = np.squeeze(sine_convolution)
            cosine_convolution = np.squeeze(cosine_convolution)

            # Store results in the appropriate index
            sine_convolutions[k, j, s] = sine_convolution  
            cosine_convolutions[k, j, s] = cosine_convolution



#%% Convolutions with Twisted Sine and Twisted Cosine Kernels on Stimulus using Correlation instead of Convolution from Scipy.Signal. With Kernel Size as separate Parameter Dimension Using Qubic Kernels with Including different Sigmas (1). With Padding to match Array Size
### Padding after Convolution ### 
### 1.0 (Slicing inside and Padding for all Sizes inside Convolution Loops) ###
### DAS SLICING KANN ICH AUCH EINMAL VORHER AUßERHALB DER CONVOLUTION LOOPS MACHEN UND ABSPEICHERN ALS LISTE, ARRAY O.Ä. UND IM LOOP DANN EINFACH DARAUS NEHMEN

import numpy as np
from scipy.signal import correlate


# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 16 und 16 x 16 x 64 Kernels
#zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels bei 32 x 32 x 32 Stimulus Array
#zeilen = 1    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1   # = n_columns of Input Array - k_columns of Kernel Array + 1

# Define the center slice function to extract the central slice from the third dimension
def extract_central_slice(matrix, kernel_depth):
    """ Extracts a central slice from the third dimension of the matrix """
    center_idx = matrix.shape[2] // 2  # Center frame index (64 // 2 = 32)
    start_idx = center_idx - (kernel_depth // 2)
    end_idx = start_idx + kernel_depth
    return matrix[:, :, start_idx:end_idx]  # Extracted slice. Python slicing is exclusive on the upper bound, here: "end_idx"

# Determine the largest output shape (corresponding to the smallest kernel)
#target_rows = matrix.shape[0] - min(size_values) + 1
#target_cols = matrix.shape[1] - min(size_values) + 1

# Define target output shape (from smallest kernel: size = 16)
target_rows = 1065  # = n_rows of Input Array - k_rows of Kernel Array + 1
target_cols = 1905  # = n_columns of Input Array - k_columns of Kernel Array + 1

# Function to apply padding to match target_rows, target_cols
def pad_to_target_shape(array, target_shape):
    """Pads a 2D array to match target_shape while centering the original values."""
    pad_rows = (target_shape[0] - array.shape[0])
    pad_cols = (target_shape[1] - array.shape[1])

    # Split padding evenly on both sides
    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top  # Used beacuse pad_rows can be odd 
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left  # Used beacuse pad_cols can be odd 

    return np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)  # Another option (edge-padding): np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')

""" FOR SLICING OUTSIDE OF CONVOLUTION LOOPS
# Precompute the central slices for sizes 16 and 32
sliced_inputs = {
    16: extract_central_slice(matrix, 16),
    32: extract_central_slice(matrix, 32),
    64: matrix,  # No slicing needed for size 64
}
"""

# Initialize arrays to store convolution results
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), target_rows, target_cols))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), target_rows, target_cols))

# Iterate over lambda and alpha first
for k, lambd in enumerate(lambd_values):
    for j, alpha in enumerate(alpha_values):
        for s, size in enumerate(size_values):  

            # Fetch sigma dynamically based on kernel size
            sigma = sigma_values[size]

            # Generate 3D sine and cosine kernels
            sine_kernel = generate_rotational_sine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)
            cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)

            # Extract a slice if kernel depth is smaller than 64
            if size < 64:
                matrix_slice = extract_central_slice(matrix, size)
            else:
                matrix_slice = matrix  # Use full matrix for 64×64×64 kernel

            """ FOR SLICING OUTSIDE OF CONVOLUTION LOOPS
            # Use precomputed extracted slices
           matrix_slice = sliced_inputs[size]
            """

            # Perform convolution
            sine_convolution = correlate(matrix_slice, sine_kernel, mode='valid', method='fft')
            cosine_convolution = correlate(matrix_slice, cosine_kernel, mode='valid', method='fft')

            # Remove singleton dimension from the result array
            sine_convolution = np.squeeze(sine_convolution)
            cosine_convolution = np.squeeze(cosine_convolution)
            
            # Pad the output to match the target shape
            sine_convolution_padded = pad_to_target_shape(sine_convolution, (target_rows, target_cols))
            cosine_convolution_padded = pad_to_target_shape(cosine_convolution, (target_rows, target_cols))

            # Store results in the appropriate index
            sine_convolutions[k, j, s] = sine_convolution_padded  
            cosine_convolutions[k, j, s] = cosine_convolution_padded
            
            """ FOR APPLYING PADDING ONLY TO THE CONVOLUTIONS ARRAYS OF SIZE 32 AND 64 KERNELS (AS PADDING FOR SIZE 16 DOES NOT YIELD ANY DIFFERENCE). MAYBE THIS IS FASTER RUNTIME, MAYBE NOT BECAUSE IT NEEDS TO DO THE 'IF'-LOOP ALL THE TIME
            # Only apply padding for sizes 32 and 64
            if size > 16:
                sine_convolution = pad_to_target_shape(sine_convolution, (target_rows, target_cols))
                cosine_convolution = pad_to_target_shape(cosine_convolution, (target_rows, target_cols))

            # Store results in the appropriate index
            sine_convolutions[k, j, s] = sine_convolution  
            cosine_convolutions[k, j, s] = cosine_convolution
            """


#%% Convolutions with Twisted Sine and Twisted Cosine Kernels on Stimulus using Correlation instead of Convolution from Scipy.Signal. With Kernel Size as separate Parameter Dimension Using Qubic Kernels with Including different Sigmas (1). With Padding to match Array Size
### Padding after Convolution ### 
### 2.0 (Slicing outside and Padding only for Sizes 32 and 64 inside Convolution Loops) ###

import numpy as np
from scipy.signal import correlate


# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 16 und 16 x 16 x 64 Kernels
#zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels bei 32 x 32 x 32 Stimulus Array
#zeilen = 1    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1   # = n_columns of Input Array - k_columns of Kernel Array + 1

# Define the center slice function to extract the central slice from the third dimension
def extract_central_slice(matrix, kernel_depth):
    """ Extracts a central slice from the third dimension of the matrix """
    center_idx = matrix.shape[2] // 2  # Center frame index (64 // 2 = 32)
    start_idx = center_idx - (kernel_depth // 2)
    end_idx = start_idx + kernel_depth
    return matrix[:, :, start_idx:end_idx]  # Extracted slice. Python slicing is exclusive on the upper bound, here: "end_idx"

# Determine the largest output shape (corresponding to the smallest kernel)
#target_rows = matrix.shape[0] - min(size_values) + 1
#target_cols = matrix.shape[1] - min(size_values) + 1

# Define target output shape (from smallest kernel: size = 16)
target_rows = 1065  # = n_rows of Input Array - k_rows of Kernel Array + 1
target_cols = 1905  # = n_columns of Input Array - k_columns of Kernel Array + 1

# Function to apply padding to match target_rows, target_cols
def pad_to_target_shape(array, target_shape):
    """Pads a 2D array to match target_shape while centering the original values."""
    pad_rows = (target_shape[0] - array.shape[0])
    pad_cols = (target_shape[1] - array.shape[1])

    # Split padding evenly on both sides
    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top  # Used beacuse pad_rows can be odd 
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left  # Used beacuse pad_cols can be odd 

    return np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)  # Another option (edge-padding): np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')

# FOR SLICING OUTSIDE OF CONVOLUTION LOOPS
# Precompute the central slices for sizes 16 and 32
sliced_inputs = {
    16: extract_central_slice(matrix, 16),
    32: extract_central_slice(matrix, 32),
    64: matrix,  # No slicing needed for size 64
}

# Initialize arrays to store convolution results
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), target_rows, target_cols))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), target_rows, target_cols))

# Iterate over lambda and alpha first
for k, lambd in enumerate(lambd_values):
    for j, alpha in enumerate(alpha_values):
        for s, size in enumerate(size_values):  

            # Fetch sigma dynamically based on kernel size
            sigma = sigma_values[size]

            # Generate 3D sine and cosine kernels
            sine_kernel = generate_rotational_sine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)
            cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)

            # Extract a slice if kernel depth is smaller than 64
            #if size < 64:
            #    matrix_slice = extract_central_slice(matrix, size)
            #else:
            #    matrix_slice = matrix  # Use full matrix for 64×64×64 kernel

            # FOR SLICING OUTSIDE OF CONVOLUTION LOOPS
            # Use precomputed extracted slices
            matrix_slice = sliced_inputs[size]
            
            # Perform convolution
            sine_convolution = correlate(matrix_slice, sine_kernel, mode='valid', method='fft')
            cosine_convolution = correlate(matrix_slice, cosine_kernel, mode='valid', method='fft')

            # Remove singleton dimension from the result array
            sine_convolution = np.squeeze(sine_convolution)
            cosine_convolution = np.squeeze(cosine_convolution)
            
            # Pad the output to match the target shape
            #sine_convolution_padded = pad_to_target_shape(sine_convolution, (target_rows, target_cols))
            #cosine_convolution_padded = pad_to_target_shape(cosine_convolution, (target_rows, target_cols))

            # Store results in the appropriate index
            #sine_convolutions[k, j, s] = sine_convolution_padded  
            #cosine_convolutions[k, j, s] = cosine_convolution_padded
            
            # FOR APPLYING PADDING ONLY TO THE CONVOLUTIONS ARRAYS OF SIZE 32 AND 64 KERNELS (AS PADDING FOR SIZE 16 DOES NOT YIELD ANY DIFFERENCE). MAYBE THIS IS FASTER RUNTIME, MAYBE NOT BECAUSE IT NEEDS TO DO THE 'IF'-LOOP ALL THE TIME
            # Only apply padding for sizes 32 and 64
            if size > 16:
                sine_convolution = pad_to_target_shape(sine_convolution, (target_rows, target_cols))
                cosine_convolution = pad_to_target_shape(cosine_convolution, (target_rows, target_cols))

            # Store results in the appropriate index
            sine_convolutions[k, j, s] = sine_convolution  
            cosine_convolutions[k, j, s] = cosine_convolution
              
            

#%% Convolutions with Twisted Sine and Twisted Cosine Kernels on Stimulus using Correlation instead of Convolution from Scipy.Signal. With Kernel Size as separate Parameter Dimension Using Qubic Kernels with Including different Sigmas (1). With Padding to match Array Size
### Padding after Convolution ### 
### 3.0 (Slicing outside and Padding only for Sizes 32 and 64 outside Convolution Loops with Dictionaries for (co)sine_convolutions_raw) ###

import numpy as np
from scipy.signal import correlate
import time


# Start tracking runtime
start_time = time.time()
print(f"Whole Execution started at {time.strftime('%Y-%m-%d %H:%M:%S')} \n")

# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 16 und 16 x 16 x 64 Kernels
#zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels bei 32 x 32 x 32 Stimulus Array
#zeilen = 1    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1   # = n_columns of Input Array - k_columns of Kernel Array + 1

"""
# Define the center slice function to extract the central slice from the third dimension
def extract_central_slice(matrix, kernel_depth):
    # Extracts a central slice from the third dimension of the matrix 
    center_idx = matrix.shape[2] // 2  # Center frame index (64 // 2 = 32)
    start_idx = center_idx - (kernel_depth // 2)
    end_idx = start_idx + kernel_depth
    return matrix[:, :, start_idx:end_idx]  # Extracted slice. Python slicing is exclusive on the upper bound, here: "end_idx"
"""

# Define the first frames slice function to extract the first slice from the third dimension
def extract_first_slice(matrix, kernel_depth):
    # Extracts a central slice from the third dimension of the matrix 
    return matrix[:, :, 0:kernel_depth]  # Extracted slice. First frames until kernel_depth


# Determine the largest output shape (corresponding to the smallest kernel)
#target_rows = matrix.shape[0] - min(size_values) + 1
#target_cols = matrix.shape[1] - min(size_values) + 1

# Define target output shape (from smallest kernel: size = 16)
target_rows = 1065  # = n_rows of Input Array - k_rows of Kernel Array + 1
target_cols = 1905  # = n_columns of Input Array - k_columns of Kernel Array + 1

"""
# Function to apply padding to match target_rows, target_cols
def pad_to_target_shape(array, target_shape):
    # Pads a 2D array to match target_shape while centering the original values.
    pad_rows = (target_shape[0] - array.shape[0])
    pad_cols = (target_shape[1] - array.shape[1])

    # Split padding evenly on both sides
    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top  # Used beacuse pad_rows can be odd 
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left  # Used beacuse pad_cols can be odd 

    return np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)  # Another option (edge-padding): np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
"""

def pad_to_target_shape_4D(array, target_shape):
    """Pads a 4D array (lambd_values, alpha_values, zeilen, spalten) to match target_rows, target_cols."""
    pad_rows = target_shape[0] - array.shape[2]  # Padding needed for rows
    pad_cols = target_shape[1] - array.shape[3]  # Padding needed for columns

    # Split padding evenly on both sides
    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top  # Handles odd numbers
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left  # Handles odd numbers

    # Apply padding only to the last two dimensions (rows, cols)
    return np.pad(array, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

"""
# FOR SLICING OUTSIDE OF CONVOLUTION LOOPS
# Precompute the central slices for sizes 16 and 32
sliced_inputs = {
    16: extract_central_slice(matrix, 16),
    32: extract_central_slice(matrix, 32),
    64: matrix,  # No slicing needed for size 64
}
"""

# Precompute the first slices for sizes 16 and 32. (Test 2.18, 02.04.25)
sliced_inputs = {
    16: extract_first_slice(matrix, 16),
    32: extract_first_slice(matrix, 32),
    64: matrix,  # No slicing needed for size 64
}

# Compute correct output shapes for each kernel size 
zeilen = {size: matrix.shape[0] - size + 1 for size in [16, 32, 64]}  # Rows per kernel size
spalten = {size: matrix.shape[1] - size + 1 for size in [16, 32, 64]}  # Columns per kernel size

# Store convolution results separately in dictionaries before applying padding
sine_convolutions_raw_dict = {size: np.zeros((len(lambd_values), len(alpha_values), zeilen[size], spalten[size])) for size in [16, 32, 64]}
cosine_convolutions_raw_dict = {size: np.zeros((len(lambd_values), len(alpha_values), zeilen[size], spalten[size])) for size in [16, 32, 64]}

# Iterate over lambda and alpha first
for k, lambd in enumerate(lambd_values):
    for j, alpha in enumerate(alpha_values):
        for s, size in enumerate(size_values):  

            # Fetch sigma dynamically based on kernel size
            sigma = sigma_values[size]

            # Generate 3D sine and cosine kernels
            sine_kernel = generate_rotational_sine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)
            cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)

            # FOR SLICING OUTSIDE OF CONVOLUTION LOOPS
            # Use precomputed extracted slices
            matrix_slice = sliced_inputs[size]  
            
            # Perform convolution
            sine_convolution = correlate(matrix_slice, sine_kernel, mode='valid', method='fft')
            cosine_convolution = correlate(matrix_slice, cosine_kernel, mode='valid', method='fft')

            # Remove singleton dimension from the result array
            sine_convolution = np.squeeze(sine_convolution)
            cosine_convolution = np.squeeze(cosine_convolution)
            
            # Store raw results without padding in dictionaries
            sine_convolutions_raw_dict[size][k, j] = sine_convolution
            cosine_convolutions_raw_dict[size][k, j] = cosine_convolution

            
# Tracking the runtime from start until here
end_time_convolutions = time.time() 
elapsed_time_start_until_convolutions = end_time_convolutions - start_time
minutes, seconds = divmod(elapsed_time_start_until_convolutions, 60)  # Get whole minutes and remaining seconds
print(f"Convolutions finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Execution time from start until including convolutions: {int(minutes)} minutes and {seconds:.0f} seconds \n")

# Start tracking runtime
start_time_padding = time.time()                       
            
# Apply padding after the loops to align all sizes
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), target_rows, target_cols))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), target_rows, target_cols))

# Apply padding after convolution loops
for s, size in enumerate(size_values):
    if size > 16:  # Only pad sizes 32 and 64
        sine_convolutions[:, :, s, :, :] = pad_to_target_shape_4D(sine_convolutions_raw_dict[size], (target_rows, target_cols))
        cosine_convolutions[:, :, s, :, :] = pad_to_target_shape_4D(cosine_convolutions_raw_dict[size], (target_rows, target_cols))
    else:  # No padding needed for size 16
        sine_convolutions[:, :, s, :, :] = sine_convolutions_raw_dict[size]
        cosine_convolutions[:, :, s, :, :] = cosine_convolutions_raw_dict[size]
        

# Tracking runtime from last start until here        
end_time_padding = time.time() 
elapsed_time_padding = end_time_padding - start_time_padding
minutes, seconds = divmod(elapsed_time_padding, 60)  # Get whole minutes and remaining seconds
print(f"Whole Execution finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Execution time for padding (from excluding convolutions until end): {int(minutes)} minutes and {seconds:.0f} seconds \n")

# Tracking total runtime from first start until end
elapsed_time_total = end_time_padding - start_time
minutes, seconds = divmod(elapsed_time_total, 60)  # Get whole minutes and remaining seconds
print(f"Total Execution time: {int(minutes)} minutes and {seconds:.0f} seconds")       


    
#%% Convolutions with Twisted Sine and Twisted Cosine Kernels on Stimulus using Correlation instead of Convolution from Scipy.Signal. With Kernel Size as separate Parameter Dimension Using Qubic Kernels with Including different Sigmas (1). With Padding to match Array Size
### Padding after Convolution ### 
### 4.0 (Slicing outside and Padding only for Sizes 32 and 64 outside Convolution Loops with Arrays for (co)sine_convolutions_raw instead of Dictionaries) ###

import numpy as np
from scipy.signal import correlate
import time


# Start tracking runtime
start_time = time.time()
print(f"Whole Execution started at {time.strftime('%Y-%m-%d %H:%M:%S')} \n")

# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 16 und 16 x 16 x 64 Kernels
#zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels bei 32 x 32 x 32 Stimulus Array
#zeilen = 1    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1   # = n_columns of Input Array - k_columns of Kernel Array + 1

# Define the center slice function to extract the central slice from the third dimension
def extract_central_slice(matrix, kernel_depth):
    """ Extracts a central slice from the third dimension of the matrix """
    center_idx = matrix.shape[2] // 2  # Center frame index (64 // 2 = 32)
    start_idx = center_idx - (kernel_depth // 2)
    end_idx = start_idx + kernel_depth
    return matrix[:, :, start_idx:end_idx]  # Extracted slice. Python slicing is exclusive on the upper bound, here: "end_idx"

# Determine the largest output shape (corresponding to the smallest kernel)
#target_rows = matrix.shape[0] - min(size_values) + 1
#target_cols = matrix.shape[1] - min(size_values) + 1

# Define target output shape (from smallest kernel: size = 16)
target_rows = 1065  # = n_rows of Input Array - k_rows of Kernel Array + 1
target_cols = 1905  # = n_columns of Input Array - k_columns of Kernel Array + 1

"""
# Function to apply padding to match target_rows, target_cols
def pad_to_target_shape(array, target_shape):
    # Pads a 2D array to match target_shape while centering the original values.
    pad_rows = (target_shape[0] - array.shape[0])
    pad_cols = (target_shape[1] - array.shape[1])

    # Split padding evenly on both sides
    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top  # Used beacuse pad_rows can be odd 
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left  # Used beacuse pad_cols can be odd 

    return np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)  # Another option (edge-padding): np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
"""

def pad_to_target_shape_4D(array, target_shape):
    """Pads a 4D array (lambd_values, alpha_values, zeilen, spalten) to match target_rows, target_cols."""
    pad_rows = target_shape[0] - array.shape[2]  # Padding needed for rows
    pad_cols = target_shape[1] - array.shape[3]  # Padding needed for columns

    # Split padding evenly on both sides
    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top  # Handles odd numbers
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left  # Handles odd numbers

    # Apply padding only to the last two dimensions (rows, cols)
    return np.pad(array, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

# FOR SLICING OUTSIDE OF CONVOLUTION LOOPS
# Precompute the central slices for sizes 16 and 32
sliced_inputs = {
    16: extract_central_slice(matrix, 16),
    32: extract_central_slice(matrix, 32),
    64: matrix,  # No slicing needed for size 64
}

# Compute correct output shapes for each kernel size 
zeilen = {size: matrix.shape[0] - size + 1 for size in [16, 32, 64]}  # Rows per kernel size
spalten = {size: matrix.shape[1] - size + 1 for size in [16, 32, 64]}  # Columns per kernel size

#zeilen = {16: 1065, 32: 1049, 64: 1017}
#spalten = {16: 1905, 32: 1889, 64: 1857}

# Store convolution results separately in arrays before applying padding
sine_convolutions_raw_array = np.zeros((len(lambd_values), len(alpha_values), len(size_values), max(zeilen.values()), max(spalten.values())))
cosine_convolutions_raw_array = np.zeros((len(lambd_values), len(alpha_values), len(size_values), max(zeilen.values()), max(spalten.values())))

# Iterate over lambda and alpha first
for k, lambd in enumerate(lambd_values):
    for j, alpha in enumerate(alpha_values):
        for s, size in enumerate(size_values):  

            # Fetch sigma dynamically based on kernel size
            sigma = sigma_values[size]

            # Generate 3D sine and cosine kernels
            sine_kernel = generate_rotational_sine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)
            cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)

            # FOR SLICING OUTSIDE OF CONVOLUTION LOOPS
            # Use precomputed extracted slices
            matrix_slice = sliced_inputs[size]
            
            # Perform convolution
            sine_convolution = correlate(matrix_slice, sine_kernel, mode='valid', method='fft')
            cosine_convolution = correlate(matrix_slice, cosine_kernel, mode='valid', method='fft')

            # Remove singleton dimension from the result array
            sine_convolution = np.squeeze(sine_convolution)
            cosine_convolution = np.squeeze(cosine_convolution)
            
            # Store raw results directly in 5D arrays
            sine_convolutions_raw_array[k, j, s, :zeilen[size], :spalten[size]] = sine_convolution
            cosine_convolutions_raw_array[k, j, s, :zeilen[size], :spalten[size]] = cosine_convolution

            
# Tracking the runtime from start until here
end_time_convolutions = time.time() 
elapsed_time_start_until_convolutions = end_time_convolutions - start_time
minutes, seconds = divmod(elapsed_time_start_until_convolutions, 60)  # Get whole minutes and remaining seconds
print(f"Convolutions finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Execution time from start until including convolutions: {int(minutes)} minutes and {seconds:.0f} seconds \n")

# Start tracking runtime
start_time_padding = time.time() 
            
# Apply padding after the loops to align all sizes
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), target_rows, target_cols))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), target_rows, target_cols))

# Apply padding after convolution loops
for s, size in enumerate(size_values):
    sine_convolutions_valid_region = sine_convolutions_raw_array[:, :, s, :zeilen[size], :spalten[size]]
    cosine_convolutions_valid_region = cosine_convolutions_raw_array[:, :, s, :zeilen[size], :spalten[size]]
    
    if size > 16:  # Only pad sizes 32 and 64
        sine_convolutions[:, :, s, :, :] = pad_to_target_shape_4D(sine_convolutions_valid_region, (target_rows, target_cols))
        cosine_convolutions[:, :, s, :, :] = pad_to_target_shape_4D(cosine_convolutions_valid_region, (target_rows, target_cols))
    else:  # No padding needed for size 16
        sine_convolutions[:, :, s, :, :] = sine_convolutions_valid_region
        cosine_convolutions[:, :, s, :, :] = cosine_convolutions_valid_region
        

# Tracking runtime from last start until here        
end_time_padding = time.time() 
elapsed_time_padding = end_time_padding - start_time_padding
minutes, seconds = divmod(elapsed_time_padding, 60)  # Get whole minutes and remaining seconds
print(f"Whole Execution finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Execution time for padding (from excluding convolutions until end): {int(minutes)} minutes and {seconds:.0f} seconds \n")

# Tracking total runtime from first start until end
elapsed_time_total = end_time_padding - start_time
minutes, seconds = divmod(elapsed_time_total, 60)  # Get whole minutes and remaining seconds
print(f"Total Execution time: {int(minutes)} minutes and {seconds:.0f} seconds") 



#%% Strided/Dilated Convolution Versuch (11.03) 

import numpy as np
from scipy.signal import correlate
from numpy.lib.stride_tricks import as_strided


def fast_strided_correlation(matrix, kernel, stride_x, stride_y, stride_z):
    """Performs strided correlation efficiently using NumPy strides instead of Python loops."""
    
    # Get output shape
    output_shape = (
        (matrix.shape[0] - kernel.shape[0]) // stride_x + 1,
        (matrix.shape[1] - kernel.shape[1]) // stride_y + 1,
        (matrix.shape[2] - kernel.shape[2]) // stride_z + 1
    )

    # Generate a view of the input matrix with strides
    strided_matrix = as_strided(
        matrix,
        shape=(output_shape[0], output_shape[1], output_shape[2], kernel.shape[0], kernel.shape[1], kernel.shape[2]),
        strides=(matrix.strides[0] * stride_x, matrix.strides[1] * stride_y, matrix.strides[2] * stride_z) + matrix.strides
    )

    # Perform element-wise multiplication and sum across the kernel dimensions
    result = np.einsum('ijklmn,lmn->ijk', strided_matrix, kernel)

    return result

# Example Usage
matrix = np.random.rand(1080, 1920, 64)  # Full-size matrix
kernel = np.random.rand(16, 16, 16)  # Example kernel
stride_x, stride_y, stride_z = 2, 2, 1  # Stride in x and y, keep full frames

# Perform strided correlation efficiently
correlation_result = fast_strided_correlation(matrix, kernel, stride_x, stride_y, stride_z)

print("Strided Correlation Output Shape:", correlation_result.shape)

        

#%% Convolutions with Twisted Sine and Twisted Cosine Kernels on Stimulus using Correlation instead of Convolution from Scipy.Signal. With Kernel Size as separate Parameter Dimension Using Qubic Kernels with Including different Sigmas (1). With Padding to match Array Size
### Padding before Convolution ### 2.0
### DAS SLICING KANN ICH AUCH EINMAL VORHER AUßERHALB DER CONVOLUTION LOOPS MACHEN UND ABSPEICHERN ALS LISTE, ARRAY O.Ä. UND IM LOOP DANN EINFACH DARAUS NEHMEN

import numpy as np
from scipy.signal import correlate


# Define the center slice function to extract the central slice from the third dimension
def extract_central_slice(matrix, kernel_depth):
    """ Extracts a central slice from the third dimension of the matrix """
    center_idx = matrix.shape[2] // 2  # Center frame index (64 // 2 = 32)
    start_idx = center_idx - (kernel_depth // 2)
    end_idx = start_idx + kernel_depth
    return matrix[:, :, start_idx:end_idx]  # Extracted slice. Python slicing is exclusive on the upper bound, here: "end_idx"

# Define target output shape (from smallest kernel: size = 16)
target_rows = 1065  # = n_rows of Input Array - k_rows of Kernel Array + 1
target_cols = 1905  # = n_columns of Input Array - k_columns of Kernel Array + 1

# Compute required input sizes
input_shapes = {
    16: (1080, 1920, 64),  # Same as original array (set as deafult because this size yields the biggest convolution output array)
    32: (1096, 1936, 64),  # Target output shape + kernel size -1
    64: (1128, 1968, 64),  # Target output shape + kernel size -1
}

# Function to create different input sizes with edge padding
def create_padded_input(matrix, target_shape):
    """Pads the input matrix symmetrically to reach the desired shape."""
    pad_rows = (target_shape[0] - matrix.shape[0])
    pad_cols = (target_shape[1] - matrix.shape[1])

    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left

    return np.pad(matrix, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')

# Create three differently padded versions of the input matrix
padded_inputs = {
    size: create_padded_input(matrix, input_shapes[size]) for size in [16, 32, 64]
}

""" FOR SLICING OUTSIDE OF CONVOLUTION LOOPS
# Precompute the central slices for sizes 16 and 32
sliced_inputs = {
    16: extract_central_slice(padded_inputs[16], 16),
    32: extract_central_slice(padded_inputs[32], 32),
    64: padded_inputs[64],  # No slicing needed for size 64
}
"""

# Initialize arrays to store convolution results
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), target_rows, target_cols))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), target_rows, target_cols))

# Iterate over lambda, alpha, and kernel sizes
for k, lambd in enumerate(lambd_values):
    for j, alpha in enumerate(alpha_values):
        for s, size in enumerate(size_values):
            
            sigma = sigma_values[size]

            # Generate 3D sine and cosine kernels
            sine_kernel = generate_rotational_sine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)
            cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)

            # Select the pre-padded input corresponding to the kernel size
            matrix_padded = padded_inputs[size]
            
            # Extract central slice for kernels smaller than full depth (= 64)
            if size < 64:
                matrix_padded = extract_central_slice(matrix_padded, size)
                
            """ FOR SLICING OUTSIDE OF CONVOLUTION LOOPS
            # Use precomputed extracted slices
           matrix_padded = sliced_inputs[size]
            """

            # Perform convolution with mode='valid' to get aligned output sizes
            sine_convolution = correlate(matrix_padded, sine_kernel, mode='valid', method='fft')
            cosine_convolution = correlate(matrix_padded, cosine_kernel, mode='valid', method='fft')
            
            # Remove singleton dimension from the result array
            sine_convolution = np.squeeze(sine_convolution)
            cosine_convolution = np.squeeze(cosine_convolution)

            # Store results in the final arrays
            sine_convolutions[k, j, s] = sine_convolution
            cosine_convolutions[k, j, s] = cosine_convolution
            
            
            
#%% Convolutions with Twisted Sine and Twisted Cosine Kernels on Stimulus using Correlation instead of Convolution from Scipy.Signal. With Kernel Size as separate Parameter Dimension Using Qubic Kernels with Including different Sigmas (2). Without Padding
### Optimized (Parallelizing and Using more Functions) by ChatGPT ###
### Stand 28.02.25: PC abgeschmiert, als ich diesen Code Abschnitt hab laufen lassen! ###
### PADDING FEHLT KOMPLETT ###
### DAS SLICING KANN ICH AUCH EINMAL VORHER AUßERHALB DER CONVOLUTION LOOPS MACHEN UND ABSPEICHERN ALS LISTE, ARRAY O.Ä. UND IM LOOP DANN EINFACH DARAUS NEHMEN

import numpy as np
from scipy.signal import correlate
from joblib import Parallel, delayed


# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 16 und 16 x 16 x 64 Kernels
#zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels bei 32 x 32 x 32 Stimulus Array
#zeilen = 1    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1   # = n_columns of Input Array - k_columns of Kernel Array + 1

# Define the center slice function to extract the central slice from the third dimension
def extract_central_slice(matrix, kernel_depth):
    """ Extracts a central slice from the third dimension of the matrix """
    center_idx = matrix.shape[2] // 2  # Center frame index (64 // 2 = 32)
    start_idx = center_idx - (kernel_depth // 2)
    end_idx = start_idx + kernel_depth  # Ensures exact slicing
    return matrix[:, :, start_idx:end_idx]  # Extracted slice

# Precompute kernels to avoid redundant calculations
precomputed_kernels = {}

def get_kernel(size, sigma, lambd, alpha, kernel_type="sine"):
    """ Retrieve or compute the kernel """
    key = (size, sigma, lambd, alpha, kernel_type)
    if key not in precomputed_kernels:
        if kernel_type == "sine":
            precomputed_kernels[key] = generate_rotational_sine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)
        else:
            precomputed_kernels[key] = generate_rotational_cosine_gabor_kernel_3d(size, size, size, sigma, lambd, alpha)
    return precomputed_kernels[key]

# Function to compute convolution for one parameter set
def compute_convolution(k, j, s, lambd, alpha, size):
    """ Computes the convolution for given parameters """

    sigma = sigma_values[size]  # Get correct sigma

    # Get precomputed kernels
    sine_kernel = get_kernel(size, sigma, lambd, alpha, "sine")
    cosine_kernel = get_kernel(size, sigma, lambd, alpha, "cosine")

    # Extract a slice if needed
    matrix_slice = extract_central_slice(matrix, size) if size < 64 else matrix

    # Perform convolution
    sine_convolution = correlate(matrix_slice, sine_kernel, mode='valid', method='fft')
    cosine_convolution = correlate(matrix_slice, cosine_kernel, mode='valid', method='fft')

    # Remove singleton dimension
    sine_convolution = np.squeeze(sine_convolution)
    cosine_convolution = np.squeeze(cosine_convolution)

    return k, j, s, sine_convolution, cosine_convolution

# Run in parallel
results = Parallel(n_jobs=4, batch_size=2, verbose=10)(  # Another option to n_jobs=-1: Parallel(n_jobs=4, batch_size=2, verbose=10)(...) -> 4 Cores used (n_jobs), 2 tasks processed simultaneously (batch_size), prints progress messages every 10 iterations (verbose)
    delayed(compute_convolution)(k, j, s, lambd, alpha, size)
    for k, lambd in enumerate(lambd_values)
    for j, alpha in enumerate(alpha_values)
    for s, size in enumerate(size_values)
)

# Initialize arrays to store results
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), zeilen, spalten))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), len(size_values), zeilen, spalten))

# Store results
for k, j, s, sine_convolution, cosine_convolution in results:
    sine_convolutions[k, j, s] = sine_convolution
    cosine_convolutions[k, j, s] = cosine_convolution
   
    
            
#%% Convolutions only in z-dimension Iterating over each Pixel at Position (x, y) (Ansatz 1). Stand 21.11: Das brauche ich für meine Actionpoints aktuell nicht benutzen

# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 64 und 16 x 16 x 16 Kernels
zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels bei 32 x 32 x 32 Stimulus Array für Convolution nur über z-Dimension (Frames)
zeilen = 32    # = n_rows of Input Array - k_rows of Kernel Array + 1
spalten = 32   # = n_columns of Input Array - k_columns of Kernel Array + 1

# Initialize output arrays
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), zeilen, spalten))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), zeilen, spalten))

# Iterate over lambd and alpha
for k, lambd in enumerate(lambd_values):
    for j, alpha in enumerate(alpha_values):
        
        # Generate 3D kernels of size 32x32x32
        sine_kernel = generate_rotational_sine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha)
        cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha)
        
        # Loop over each spatial coordinate (x, y)
        for y in range(zeilen):
            for x in range(spalten):
                
                # Extract z-dimension for the current (x, y) location
                z_vector = matrix[y, x, :]  # Shape: (32,)
                
                # Extract the corresponding kernel along the z-dimension
                sine_kernel_z = sine_kernel[y, x, :]  # Shape: (32,)
                cosine_kernel_z = cosine_kernel[y, x, :]  # Shape: (32,)
                
                # Perform convolution along the z-dimension (dot product)
                sine_convolutions[k, j, y, x] = np.sum(z_vector * sine_kernel_z)  # Element-wise multiplication and summation along z-dimension (dot product) without flipping any array before
                cosine_convolutions[k, j, y, x] = np.sum(z_vector * cosine_kernel_z)  # Element-wise multiplication and summation along z-dimension (dot product) without flipping any array before
                
                

#%% Convolutions only in z-dimension without Iterating over each Pixel at Position (x, y) (Ansatz 2). Stand 21.11: Das brauche ich für meine Actionpoints aktuell nicht benutzen

# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 64 und 16 x 16 x 16 Kernels
zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 32 x 32 x 32 Kernels bei 32 x 32 x 32 Stimulus Array für Convolution nur über z-Dimension (Frames)
zeilen = 32    # = n_rows of Input / Kernel Array
spalten = 32   # = n_columns of Input / Kernel Array

# Initialize output arrays
sine_convolutions = np.zeros((len(lambd_values), len(alpha_values), zeilen, spalten))
cosine_convolutions = np.zeros((len(lambd_values), len(alpha_values), zeilen, spalten))

for k, lambd in enumerate(lambd_values):
    for j, alpha in enumerate(alpha_values):
        
        # Generate 3D kernels
        sine_kernel = generate_rotational_sine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha)
        cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size_x, size_y, size_z, sigma, lambd, alpha)
        
        # Element-wise multiplication and summation along z-dimension (dot product)
        sine_convolutions[k, j] = np.sum(matrix * sine_kernel, axis=2)  # Element-wise multiplication and summation along z-dimension (dot product) without flipping any array before
        cosine_convolutions[k, j] = np.sum(matrix * cosine_kernel, axis=2)  # Element-wise multiplication and summation along z-dimension (dot product) without flipping any array before
    
        
        
#%% Quadrature and Sum of Quadrature Sine and Cosine Convolutions

# Take the quadrature of both sine and cosine convolution results
quadrature_sine_convolutions = np.square(sine_convolutions)
quadrature_cosine_convolutions = np.square(cosine_convolutions)

# Sum up the quadrature convolution results to get the final convolution result
summed_quadrature_convolutions = quadrature_sine_convolutions + quadrature_cosine_convolutions



#%% Reshaping and Finding Maximum Activation Kernel Index per Pixel (Ansatz 1)

# Reshape convolution_results to make it easier to work with
reshaped_results = summed_quadrature_convolutions.reshape((-1, zeilen, spalten))

# Retrieve the maximum activation indices for each pixel
max_activation_kernel_indices = np.argmax(reshaped_results, axis=0)



#%% Reshaping and Finding Maximum Activation Kernel Index per Pixel (Ansatz 2). Bei diesem Reshape-Ansatz sollte dasselbe rauskommen wie im Ansatz 1 darüber

# Reshape convolution_results to make it easier to work with
reshaped_results = summed_quadrature_convolutions.reshape((len(lambd_values) * len(alpha_values), zeilen, spalten))

# Find the index of the maximum convolution result across all kernels for each pixel
max_activation_kernel_indices = np.argmax(reshaped_results, axis=0)



#%% Reshaping and Finding Maximum Activation Kernel Index per Pixel mit Unravel-Befehl (Ansatz 1)

# Reshape convolution_results to make it easier to work with
reshaped_results = summed_quadrature_convolutions.reshape(-1, zeilen, spalten)

# Find the index of the maximum convolution result across all kernels for each pixel
max_activation_kernel_indices = np.argmax(reshaped_results, axis=0)

# Convert flat indices to (lambd, alpha) indices
max_activation_parameter_indices = np.unravel_index(max_activation_kernel_indices, (len(lambd_values), len(alpha_values)))  # Creates a tuple of 2 Matrices. One Matrix for each Parameter (lambd, alpha) with the corresponding Parameter Index in each Pixel

# Retrieve the lambd, alpha values corresponding to the maximally activated kernel index (or here parameter index) in each pixel
max_activation_lambd_array = np.array(lambd_values)[max_activation_parameter_indices[0]]
max_activation_alpha_array = np.array(alpha_values)[max_activation_parameter_indices[1]]



#%% Reshaping and Finding Maximum Activation Kernel Index per Pixel mit Unravel-Befehl (Ansatz 1). With Kernel Size as separate Parameter Dimension

# Reshape convolution_results to make it easier to work with
#reshaped_results = summed_quadrature_convolutions.reshape(-1, zeilen, spalten)
reshaped_results = summed_quadrature_convolutions.reshape(-1, target_rows, target_cols)

# Find the index of the maximum convolution result across all kernels for each pixel
max_activation_kernel_indices = np.argmax(reshaped_results, axis=0)

# Convert flat indices to (lambd, alpha, size) indices
max_activation_parameter_indices = np.unravel_index(max_activation_kernel_indices, (len(lambd_values), len(alpha_values), len(size_values)))  # Creates a tuple of 3 Matrices. One Matrix for each Parameter (lambd, alpha, size) with the corresponding Parameter Index in each Pixel

# Retrieve the lambd, alpha, size values corresponding to the maximally activated kernel index (or here parameter index) in each pixel
max_activation_lambd_array = np.array(lambd_values)[max_activation_parameter_indices[0]]
max_activation_alpha_array = np.array(alpha_values)[max_activation_parameter_indices[1]]
max_activation_size_array = np.array(size_values)[max_activation_parameter_indices[2]]



#%% Reshaping and Finding Maximum Activation Kernel Index per Pixel mit Unravel-Befehl (Ansatz 1). With Kernel Phase as separate Parameter Dimension

# Reshape convolution_results to make it easier to work with
reshaped_results = summed_quadrature_convolutions.reshape(-1, zeilen, spalten)

# Find the index of the maximum convolution result across all kernels for each pixel
max_activation_kernel_indices = np.argmax(reshaped_results, axis=0)

# Convert flat indices to (lambd, alpha) indices
max_activation_parameter_indices = np.unravel_index(max_activation_kernel_indices, (len(lambd_values), len(alpha_values), len(phase_values)))  # Creates a tuple of 3 Matrices. One Matrix for each Parameter (lambd, alpha, phase) with the corresponding Parameter Index in each Pixel

# Retrieve the lambd, alpha values corresponding to the maximally activated kernel index (or here parameter index) in each pixel
max_activation_lambd_array = np.array(lambd_values)[max_activation_parameter_indices[0]]
max_activation_alpha_array = np.array(alpha_values)[max_activation_parameter_indices[1]]
max_activation_phase_array = np.array(phase_values)[max_activation_parameter_indices[2]]



#%% Reshaping and Finding Maximum Activation Kernel Index per Pixel mit Unravel-Befehl (Ansatz 2). Bei diesem Reshape-Ansatz sollte dasselbe rauskommen wie im Ansatz 1 darüber

# Reshape convolution_results to make it easier to work with
reshaped_results = summed_quadrature_convolutions.reshape((len(lambd_values) * len(alpha_values), zeilen, spalten))

# Find the index of the maximum convolution result across all kernels for each pixel
max_activation_kernel_indices = np.argmax(reshaped_results, axis=0)

# Convert flat indices to (lambd, alpha) indices
max_activation_parameter_indices = np.unravel_index(max_activation_kernel_indices, (len(lambd_values), len(alpha_values)))  # Creates a tuple of 3 Matrices. One Matrix for each Parameter (lambd, alpha) with the corresponding Parameter Index in each Pixel

# Retrieve the lambd, alpha values corresponding to the maximally activated kernel index (or here parameter index) in each pixel
max_activation_lambd_array = np.array(lambd_values)[max_activation_parameter_indices[0]]
max_activation_alpha_array = np.array(alpha_values)[max_activation_parameter_indices[1]]



#%% One Convolution of Maximally Activated Kernel only in z-dimension without Iterating over each Pixel at Position (x, y) (Ansatz 3): Sine and Cosine Convolution

# Retrieving the one maximally activated parameter value for lambda and alpha
max_activation_lambd = max_activation_lambd_array[0][0]  # [0][0] accesses the element of the outer and inner list whereas [0] only accesses the element of the outer list which can be a list itself
max_activation_alpha = max_activation_alpha_array[0][0]  
# Assuming max_activation_alpha is a float
max_activation_alpha_fraction = Fraction(max_activation_alpha/np.pi).limit_denominator()  # Convert to fraction with limited denominator


# Retrieving more than one maximally activated parameter value for lambda and alpha through a loop
#for max_activation_lambd, max_activation_alpha in zip(max_activation_lambd_array, max_activation_alpha_array):
#    print(f"Max Activation Lambda: {max_activation_lambd}, Max Actvation Alpha: {max_activation_alpha}")

# Print the maximally activated Lambda and Alpha corresponding to the Kernel Index that was maximally activated in the previous Loop of Convolutions 
#print(f"Max Activation Lambda: {max_activation_lambd}, Max Actvation Alpha: {max_activation_alpha/np.pi}π")
print(f"Max Activation Lambda: {max_activation_lambd}, Max Actvation Alpha: {max_activation_alpha_fraction}π")  # Print alpha as fraction
        
# Generate 3D kernels
max_activation_sine_kernel = generate_rotational_sine_gabor_kernel_3d(size_x, size_y, size_z, sigma, max_activation_lambd, max_activation_alpha)
max_activation_cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size_x, size_y, size_z, sigma, max_activation_lambd, max_activation_alpha)
        
# Perform convolution along the z-dimension (dot product)
max_activation_sine_convolution = np.sum(matrix * max_activation_sine_kernel, axis=2)  # Element-wise multiplication and summation along z-dimension (dot product)
max_activation_cosine_convolution = np.sum(matrix * max_activation_cosine_kernel, axis=2)  # Element-wise multiplication and summation along z-dimension (dot product)

# Print the minimum and maximum values for both arrays
print("\nmax_activation_sine_convolution:")
print("Min value:", max_activation_sine_convolution.min())
print("Max value:", max_activation_sine_convolution.max())
print("Min absolute value:", np.abs(max_activation_sine_convolution).min())
print("Sum:", np.sum(max_activation_sine_convolution))

print("\nmax_activation_cosine_convolution:")
print("Min value:", max_activation_cosine_convolution.min())
print("Max value:", max_activation_cosine_convolution.max())
print("Min absolute value:", np.abs(max_activation_cosine_convolution).min())
print("Sum:", np.sum(max_activation_cosine_convolution))


# Find the common minimum and maximum across both arrays
vmin = min(max_activation_sine_convolution.min(), max_activation_cosine_convolution.min())
vmax = max(max_activation_sine_convolution.max(), max_activation_cosine_convolution.max())

# Plot of Maximally Activated Sine Convolution 
plt.imshow(max_activation_sine_convolution, cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)  # Use the common minimum vmin and maximum vmax as boundaries for the colormap
plt.colorbar()  
plt.title(f'Sine Convolution of Max Activated Kernel along z-axis\n'
    f'Lambda: {max_activation_lambd:.1f}, '  f'Alpha: {max_activation_alpha_fraction}π, ' f'Sigma: {sigma}\n'  # Or use max_activation_alpha/np.pi:.2f for decimal values        
    f'Min: {max_activation_sine_convolution.min():.1e}, '
    f'Max: {max_activation_sine_convolution.max():.1e}, '
    f'Abs Min: {np.abs(max_activation_sine_convolution).min():.1e}, '
    f'Sum: {np.sum(max_activation_sine_convolution):.1e}',
    pad=20)  # Pad parameter controls the distance between the title and the plot)
#plt.tight_layout()  # Optionally adjust the overall layout
plt.show()

# Plot of Maximally Activated Cosine Convolution 
plt.imshow(max_activation_cosine_convolution, cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)  # Use the common minimum vmin and maximum vmax as boundaries for the colormap
plt.colorbar()  
plt.title(f'Cosine Convolution of Max Activated Kernel along z-axis\n'
    f'Lambda: {max_activation_lambd:.1f}, '  f'Alpha: {max_activation_alpha_fraction}π, ' f'Sigma: {sigma}\n'  # Or use max_activation_alpha/np.pi:.2f for decimal values          
    f'Min: {max_activation_cosine_convolution.min():.1e}, '
    f'Max: {max_activation_cosine_convolution.max():.1e}, '
    f'Abs Min: {np.abs(max_activation_cosine_convolution).min():.1e}, '
    f'Sum: {np.sum(max_activation_cosine_convolution):.1e}',
    pad=20)  # Pad parameter controls the distance between the title and the plot)
#plt.tight_layout()  # Optionally adjust the overall layout
plt.show()



#%% One Convolution of Maximally Activated Kernel only in z-dimension without Iterating over each Pixel at Position (x, y) (Ansatz 3): Quadrature and Sum of Quadrature Sine and Cosine Convolution 
### HIER DAS MACHT NICHT SO VIEL SINN, DENN DAS IST NICHT DIE RICHTIGE DARSTELLUNG FÜR DAS, WAS ICH BETRACHTE. ICH DACHTE DIE SUMME AUS ALLEN EINTRÄGEN VON "max_activation_summed_quadrature_convolution" MÜSSTE DEM WERT VON "max_activation_values" ENTSPRECHEN, ABER DAFÜR MÜSSTE ICH ERST DIE SUMMEN AUS ALLEN EINTRÄGEN VON "max_activation_sine_convolution" UND "max_activation_cosine_convolution" BILDEN. DIESE SUMMEN DANN QUADRIEREN UND ADDIEREN, UM AUF DEN "max_activation_values" WERT ZU KOMMEN.
### FORTSETZUNG DER ZEILE DRÜBER: DAHER MACHT HIER DIE BERECHNUNG UND DER PLOT VON "max_activation_summed_quadrature_convolution" KEINEN SINN !

# Take the quadrature of both sine and cosine convolution results
max_activation_quadrature_sine_convolution = np.square(max_activation_sine_convolution)
max_activation_quadrature_cosine_convolution = np.square(max_activation_cosine_convolution)

# Sum up the quadrature convolution results to get the final convolution result
max_activation_summed_quadrature_convolution = max_activation_quadrature_sine_convolution  + max_activation_quadrature_cosine_convolution 

# Print the minimum and maximum values of the array
print("max_activation_summed_quadrature_convolution:")
print("Min value:", max_activation_summed_quadrature_convolution.min())
print("Max value:", max_activation_summed_quadrature_convolution.max())
print("Min absolute value:", np.abs(max_activation_summed_quadrature_convolution).min())

# Plot of Maximally Activated Sum of Quadrature Sine and Cosine Convolution 
plt.imshow(max_activation_summed_quadrature_convolution, cmap='gray', interpolation='none')  
plt.colorbar()  
plt.title(f'Summed Quadrature Convolutions of Max Activated Kernel along z-axis\n'
    f'Lambda: {max_activation_lambd:.1f}, '  f'Alpha: {max_activation_alpha_fraction}π, ' f'Sigma: {sigma}\n'  # Or use max_activation_alpha/np.pi:.2f for decimal values    
    f'Min: {max_activation_cosine_convolution.min():.1e}, '
    f'Max: {max_activation_cosine_convolution.max():.1e}, '
    f'Abs Min: {np.abs(max_activation_cosine_convolution).min():.1e}',
    pad=20)  # Pad parameter controls the distance between the title and the plot)
#plt.tight_layout()  # Optionally adjust the overall layout
plt.show()



#%% One Convolution of Maximally Activated Kernel in x- AND y-dimension without Iterating over each Pixel at Position (x, y) (Ansatz 3): Sine and Cosine Convolution. Plot of Convolution Value over Frame

# Retrieving the one maximally activated parameter value for lambda and alpha
#max_activation_lambd = max_activation_lambd_array[0][0]  # [0][0] accesses the element of the outer and inner list whereas [0] only accesses the element of the outer list which can be a list itself
#max_activation_alpha = max_activation_alpha_array[0][0]  
max_activation_lambd = 8.0
max_activation_alpha = 1/16 * np.pi  # Test 2.17 (Actionpoint 62) und Test 2.24 (Actionpoint 63)
#max_activation_alpha = 1/8 * np.pi  # Test 2.20 (Actionpoint 62, 03.04.25)
#max_activation_alpha = 1/15 * np.pi  # Test 2.21 (Actionpoint 62, 11.04.25)
#max_activation_alpha = 2/15 * np.pi  # Test 2.22 (Actionpoint 62, 11.04.25)
#max_activation_alpha = -1/3 * np.pi  # Test 2.27 (Actionpoint 63, 18.04.25)
# Assuming max_activation_alpha is a float
max_activation_alpha_fraction = Fraction(max_activation_alpha/np.pi).limit_denominator()  # Convert to fraction with limited denominator

#max_activation_phase = 1/2 * np.pi  # Test 2.24 (Actionpoint 63, 16.04.25)
#max_activation_phase = 3/2 * np.pi  # Test 2.24 (Actionpoint 63, 16.04.25)
#max_activation_phase = 1/4 * np.pi  # Test 2.27 (Actionpoint 63, 18.04.25)
max_activation_phase = 3/4 * np.pi 
#max_activation_phase = 0
# Assuming max_activation_phase is a float
max_activation_phase_fraction = Fraction(max_activation_phase/np.pi).limit_denominator()  # Convert to fraction with limited denominator

# Size and sigma parameter for the kernels
size = 16
size_x = 16  
size_y = 16 
size_z = 16 

sigma = 3.106  # Test vom Stand 25.12 (Actionpoint 38). For 16 x 16 x 16 Kernels
#sigma = 3.106/2


# Retrieving more than one maximally activated parameter value for lambda and alpha through a loop
#for max_activation_lambd, max_activation_alpha in zip(max_activation_lambd_array, max_activation_alpha_array):
#    print(f"Max Activation Lambda: {max_activation_lambd}, Max Actvation Alpha: {max_activation_alpha}")

# Print the maximally activated Lambda and Alpha (and Phase) corresponding to the Kernel Index that was maximally activated in the previous Loop of Convolutions 
#print(f"Max Activation Lambda: {max_activation_lambd}, Max Actvation Alpha: {max_activation_alpha/np.pi}π")
#print(f"Max Activation Lambda: {max_activation_lambd}, Max Actvation Alpha: {max_activation_alpha_fraction}π")  # Print alpha as fraction
#print(f"Max Activation Lambda: {max_activation_lambd}, Max Actvation Alpha: {max_activation_alpha_fraction}π, Sigma = {sigma}")  # Print alpha as fraction
print(f"Max Activation Lambda: {max_activation_lambd}, Max Actvation Alpha: {max_activation_alpha_fraction}π, Max Actvation Phase: {max_activation_phase_fraction}π, Sigma = {sigma}")  # Print alpha and phase as fraction. Mit Phase als Parameter

# Define the center slice function to extract the central slice from the third dimension
def extract_central_slice(matrix, kernel_depth):
    """ Extracts a central slice from the third dimension of the matrix """
    center_idx = matrix.shape[2] // 2  # Center frame index (64 // 2 = 32)
    start_idx = center_idx - (kernel_depth // 2)
    end_idx = start_idx + kernel_depth
    return matrix[:, :, start_idx:end_idx]  # Extracted slice. Python slicing is exclusive on the upper bound, here: "end_idx"

# FOR SLICING OUTSIDE OF CONVOLUTION LOOPS
# Precompute the central slices for sizes 16 and 32
sliced_inputs = {
    16: extract_central_slice(matrix, 16),
    32: extract_central_slice(matrix, 32),
    64: matrix,  # No slicing needed for size 64
}

# Define the xy-region for the matrix sclice
#row_start, row_end = 600 - size // 2, 600 + size // 2
#col_start, col_end = 500 - size // 2, 500 + size // 2
row_start, row_end = roi_y_start, roi_y_end  # Test 2.27, 2.28
col_start, col_end = roi_x_start, roi_x_end  # Test 2.27, 2.28

# Generate matrix slice in specific area of original matrix
matrix_slice_1 = sliced_inputs[size]  # Use precomputed extracted slices to slice in z-dimension
matrix_slice_2 = matrix_slice_1[row_start:row_end, col_start:col_end, :]  # Use specific region to slice in x- and y-dimension

# Generate 3D kernels
#max_activation_sine_kernel = generate_rotational_sine_gabor_kernel_3d(size_x, size_y, size_z, sigma, max_activation_lambd, max_activation_alpha)
#max_activation_cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size_x, size_y, size_z, sigma, max_activation_lambd, max_activation_alpha)
max_activation_sine_kernel = generate_rotational_sine_gabor_kernel_3d(size_x, size_y, size_z, sigma, max_activation_lambd, max_activation_alpha, max_activation_phase)  # Mit Phase als Parameter 
max_activation_cosine_kernel = generate_rotational_cosine_gabor_kernel_3d(size_x, size_y, size_z, sigma, max_activation_lambd, max_activation_alpha, max_activation_phase)  # Mit Phase als Parameter
        
# Convolution along x and y, resulting in a 1D array (one value per frame z)
#max_activation_sine_convolution = np.sum(matrix_slice_2 * max_activation_sine_kernel, axis=(0, 1))
#max_activation_cosine_convolution = np.sum(matrix_slice_2 * max_activation_cosine_kernel, axis=(0, 1))
max_activation_sine_convolution = np.sum(max_activation_sine_kernel * max_activation_sine_kernel, axis=(0, 1))  # Test 2.29 (Actionpoint 64, 22.04.25): Autokorrelation der Kernels
max_activation_cosine_convolution = np.sum(max_activation_cosine_kernel * max_activation_cosine_kernel, axis=(0, 1))  # Test 2.29 (Actionpoint 64, 22.04.25): Autokorrelation der Kernels

# Print the minimum and maximum values for both arrays
print("\nmax_activation_sine_convolution:")
print("Min value:", max_activation_sine_convolution.min())
print("Max value:", max_activation_sine_convolution.max())
print("Min absolute value:", np.abs(max_activation_sine_convolution).min())
print("Sum:", np.sum(max_activation_sine_convolution))

print("\nmax_activation_cosine_convolution:")
print("Min value:", max_activation_cosine_convolution.min())
print("Max value:", max_activation_cosine_convolution.max())
print("Min absolute value:", np.abs(max_activation_cosine_convolution).min())
print("Sum:", np.sum(max_activation_cosine_convolution))

# Find the common minimum and maximum across both arrays
#vmin = min(max_activation_sine_convolution.min(), max_activation_cosine_convolution.min())
#vmax = max(max_activation_sine_convolution.max(), max_activation_cosine_convolution.max())

frames = np.arange(size_z)  # X-axis: frame indices

# Plot of Maximally Activated Sine Convolution Over Frames
plt.figure(figsize=(8, 4))

plt.plot(frames, max_activation_sine_convolution, marker='o', linestyle='none', color='b', label='Sine Convolution')
plt.xlabel('Frame Index (z-axis)')
plt.ylabel('Convolution Value')
#plt.title(f'Sine Convolution Over Frames\nLambda: {max_activation_lambd:.1f}, Alpha: {max_activation_alpha_fraction}π, Sigma: {sigma}, Total Sum: {np.sum(max_activation_sine_convolution):.1e}')
plt.title(f'Sine Convolution Over Frames\nLambda: {max_activation_lambd:.1f}, Alpha: {max_activation_alpha_fraction}π, Phase: {max_activation_phase_fraction}π, Sigma: {sigma}, Total Sum: {np.sum(max_activation_sine_convolution):.2e}')  # Mit Phase als Parameter
plt.legend()
plt.grid(True)
plt.show()

# Plot of Maximally Activated Cosine Convolution Over Frames
plt.figure(figsize=(8, 4))
plt.plot(frames, max_activation_cosine_convolution, marker='o', linestyle='none', color='r', label='Cosine Convolution')
plt.xlabel('Frame Index (z-axis)')
plt.ylabel('Convolution Value')
#plt.title(f'Cosine Convolution Over Frames\nLambda: {max_activation_lambd:.1f}, Alpha: {max_activation_alpha_fraction}π, Sigma: {sigma}, Total Sum: {np.sum(max_activation_cosine_convolution):.1e}')
plt.title(f'Cosine Convolution Over Frames\nLambda: {max_activation_lambd:.1f}, Alpha: {max_activation_alpha_fraction}π, Phase: {max_activation_phase_fraction}π, Sigma: {sigma}, Total Sum: {np.sum(max_activation_cosine_convolution):.2e}')  # Mit Phase als Parameter
plt.legend()
plt.grid(True)
plt.show()



#%% Plot of Convolution Value over Frame of Code Snippet before with adjustable y-axis range and tick spacing

from matplotlib.ticker import MultipleLocator

# Plot of Maximally Activated Sine Convolution Over Frames
plt.figure(figsize=(8, 4))
plt.plot(frames, max_activation_sine_convolution, marker='o', linestyle='none', color='b', label='Sine Convolution')
plt.xlabel('Frame Index (z-axis)')
plt.ylabel('Convolution Value')
plt.title(f'Sine Convolution Over Frames\nLambda: {max_activation_lambd:.1f}, Alpha: {max_activation_alpha_fraction}π, Sigma: {sigma}, Total Sum: {np.sum(max_activation_sine_convolution):.1e}')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.002))  # Tick spacing
plt.ylim(0, 0.01)  # Y-axis range
plt.show()

# Plot of Maximally Activated Cosine Convolution Over Frames
plt.figure(figsize=(8, 4))
plt.plot(frames, max_activation_cosine_convolution, marker='o', linestyle='none', color='r', label='Cosine Convolution')
plt.xlabel('Frame Index (z-axis)')
plt.ylabel('Convolution Value')
plt.title(f'Cosine Convolution Over Frames\nLambda: {max_activation_lambd:.1f}, Alpha: {max_activation_alpha_fraction}π, Sigma: {sigma}, Total Sum: {np.sum(max_activation_cosine_convolution):.1e}')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.002))  # Tick spacing
plt.ylim(0, 0.01)  # Y-axis range
plt.show()



#%% One Convolution of Maximally Activated Kernel in x- AND y-dimension without Iterating over each Pixel at Position (x, y) (Ansatz 3): Sine and Cosine Convolution. Plot of Matrix and Kernel Slice per Frame 

import matplotlib.pyplot as plt
import numpy as np


# Iterate over each frame z and plot corresponding slices
for z in range(size_z):
#for z in [8]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Extract xy slices at frame z
    #matrix_xy_slice = matrix_slice_2[:, :, z]
    matrix_xy_slice = max_activation_sine_kernel[:, :, z]
    sine_kernel_xy_slice = max_activation_sine_kernel[:, :, z]

    # Plot matrix slice
    im1 = axes[0].imshow(matrix_xy_slice, cmap='gray')
    axes[0].set_title(f'Matrix Slice at z={z} \nSum (x, y): {np.sum(matrix_xy_slice, axis=(0, 1)):.1e}')
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)  # Add colorbar

    # Plot Sine Kernel slice
    im2 = axes[1].imshow(sine_kernel_xy_slice, cmap='gray')
    axes[1].set_title(f'Sine Kernel Slice at z={z} \nSum (x, y): {np.sum(sine_kernel_xy_slice, axis=(0, 1)):.3e}, Sum of Abs Values (x, y): {np.sum(np.abs(sine_kernel_xy_slice), axis=(0, 1)):.3e}')
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)  # Add colorbar

    plt.tight_layout()
    #plt.gca().set_aspect('equal')  # Adjust aspect ratio
    plt.show()

    # Now repeat for Cosine Kernel
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    matrix_xy_slice = max_activation_cosine_kernel[:, :, z]
    cosine_kernel_xy_slice = max_activation_cosine_kernel[:, :, z]

    # Plot matrix slice again
    im3 = axes[0].imshow(matrix_xy_slice, cmap='gray')
    axes[0].set_title(f'Matrix Slice at z={z} \nSum (x, y): {np.sum(matrix_xy_slice, axis=(0, 1)):.1e}')
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    fig.colorbar(im3, ax=axes[0], fraction=0.046, pad=0.04)  # Add colorbar

    # Plot Cosine Kernel slice
    im4 = axes[1].imshow(cosine_kernel_xy_slice, cmap='gray')
    axes[1].set_title(f'Cosine Kernel Slice at z={z} \nSum (x, y): {np.sum(cosine_kernel_xy_slice, axis=(0, 1)):.3e}, Sum of Abs Values (x, y): {np.sum(np.abs(cosine_kernel_xy_slice), axis=(0, 1)):.3e}')
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    fig.colorbar(im4, ax=axes[1], fraction=0.046, pad=0.04)  # Add colorbar

    plt.tight_layout()
    #plt.gca().set_aspect('equal')  # Adjust aspect ratio
    plt.show()



#%% One Convolution of Maximally Activated Kernel in x- AND y-dimension without Iterating over each Pixel at Position (x, y) (Ansatz 3): Sine and Cosine Convolution. Plot of Kernel Sum and Abs Sum per Frame 

import matplotlib.pyplot as plt
import numpy as np

# Vectorized sum over axis 0 and 1 → results in 1D array over z axis
sine_sum       = np.sum(max_activation_sine_kernel, axis=(0, 1))
sine_abs_sum   = np.sum(np.abs(max_activation_sine_kernel), axis=(0, 1))
cosine_sum     = np.sum(max_activation_cosine_kernel, axis=(0, 1))
cosine_abs_sum = np.sum(np.abs(max_activation_cosine_kernel), axis=(0, 1))

# Plot
z_frames = np.arange(sine_sum.shape[0])

plt.figure(figsize=(12, 6))

# --- Subplot 1: Raw Sums ---
plt.subplot(1, 2, 1)
plt.plot(z_frames, sine_sum, marker='o', linestyle='None', label=f'Sine Sum. Total: {np.sum(sine_sum):.3e}', color='blue')
plt.plot(z_frames, cosine_sum, marker='o', linestyle='None', label=f'Cosine Sum. Total: {np.sum(cosine_sum):.3e}', color='red')
plt.xlabel("Frame Index (z)")
plt.ylabel("Sum")
#plt.title(f"Sine & Cosine Kernel Sum \n Alpha: {max_activation_alpha_fraction}π")
plt.title(f"Sine & Cosine Kernel Sum \n Alpha: {max_activation_alpha_fraction}π, Phase: {max_activation_phase_fraction}π")
plt.legend()
plt.grid(True)

# --- Subplot 2: Absolute Sums ---
plt.subplot(1, 2, 2)
plt.plot(z_frames, sine_abs_sum, marker='o', linestyle='None', label=f'Sine Abs Sum. Total: {np.sum(sine_abs_sum):.3e}', color='skyblue')
plt.plot(z_frames, cosine_abs_sum, marker='o', linestyle='None', label=f'Cosine Abs Sum. Total: {np.sum(cosine_abs_sum):.3e}', color='salmon')
plt.xlabel("Frame Index (z)")
plt.ylabel("Abs Sum")
#plt.title(f"Sine & Cosine Kernel Abs Sum \n Alpha: {max_activation_alpha_fraction}π")
plt.title(f"Sine & Cosine Kernel Abs Sum \n Alpha: {max_activation_alpha_fraction}π, Phase: {max_activation_phase_fraction}π")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



#%% One Convolution of Curl Kernel with Translation as Stimulus with previous original Curl Kernels with One Phase Value

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

# --- Define your functions or import them if modularized ---
# Required:
# - generate_rotational_sine_gabor_kernel_with_translation
# - generate_rotational_cosine_gabor_kernel_with_translation
# - generate_rotational_sine_gabor_kernel_3d (original filter version)
# - generate_rotational_cosine_gabor_kernel_3d (original filter version)

# --- Parameters of the Stimulus ---
lambd_stimulus = 8.0

alpha_stimulus = 1/16 * np.pi
alpha_fraction_stimulus = Fraction(alpha_stimulus / np.pi).limit_denominator()

phase_stimulus = 0
phase_fraction_stimulus = Fraction(phase_stimulus / np.pi).limit_denominator()

#vx_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
vx_values = [-20.0, -15.0, -10.0, -4.0, -2.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0]  # Translation speeds of stimulus
#vx_values = np.linspace(-20.0, 20.0, num=41).tolist()
#vx_values = np.arange(-20.0, 20.0 + 0.2, 0.2).tolist()

# --- Parameters of the Kernels ---
size = 16  # Same also for the stimulus
sigma = 3.106  # Same also for the stimulus
lambd = 8.0   # Same also for the stimulus
#alpha_values = [0, -1/32, -1/16, -3/32, -1/8, 1/32, 1/16, 3/32, 1/8]  # Rotation speeds of filter kernels
alpha_values = [0, -1/64, -1/32, -1/16, -1/8, -1/4, -1/3, 1/64, 1/32, 1/16, 1/8, 1/4, 1/3] 

# Choose the actual alpha values not in multiples of pi
alpha_values = [value * np.pi for value in alpha_values] 

phase = 0


# --- Output storage ---
energy_matrix = np.zeros((len(alpha_values), len(vx_values)))

# --- Main loop ---
for vx_index, vx in enumerate(vx_values):
    # Stimulus kernels with translation
    stimulus_sine = generate_rotational_sine_gabor_kernel_with_translation(
        size, size, size, sigma, lambd_stimulus, alpha_stimulus, phase_stimulus, vx
    )
    stimulus_cosine = generate_rotational_cosine_gabor_kernel_with_translation(
        size, size, size, sigma, lambd_stimulus, alpha_stimulus, phase_stimulus, vx
    )

    for alpha_index, alpha in enumerate(alpha_values):
        # Filter kernels without translation
        filter_sine = generate_rotational_sine_gabor_kernel_3d(
            size, size, size, sigma, lambd, alpha, phase
        )
        filter_cosine = generate_rotational_cosine_gabor_kernel_3d(
            size, size, size, sigma, lambd, alpha, phase
        )

        # Correlation + Quadrature Energy
        conv_sine = np.sum(stimulus_sine * filter_sine)
        conv_cosine = np.sum(stimulus_cosine * filter_cosine)
        energy = conv_sine**2 + conv_cosine**2
        energy_matrix[alpha_index, vx_index] = energy


# --- Plotting ---
plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alpha_values):
    alpha_fraction = Fraction(alpha / np.pi).limit_denominator()
    plt.plot(vx_values, energy_matrix[i], marker='o',
             label=f'α = {alpha_fraction.numerator}/{alpha_fraction.denominator}π')

plt.xlabel('Translation Speed vx')
plt.ylabel('Curl Energy')
plt.title(f'Curl Energy vs. Translation Speed for Different α (Stimulus Rotation Speed = {alpha_fraction_stimulus.numerator}/{alpha_fraction_stimulus.denominator}π), Phase = {phase_stimulus}')
plt.legend(title='Filter α')
plt.grid(True)
plt.tight_layout()
plt.show()



#%% One Convolution of Curl Kernel with Translation as Stimulus with previous original Curl Kernels with Several Phase Values

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# --- Define your functions or import them if modularized ---
# Required:
# - generate_rotational_sine_gabor_kernel_with_translation
# - generate_rotational_cosine_gabor_kernel_with_translation
# - generate_rotational_sine_gabor_kernel_3d
# - generate_rotational_cosine_gabor_kernel_3d

# --- Parameters of the Stimulus ---
lambd_stimulus = 8.0
alpha_stimulus = 1/16 * np.pi
alpha_fraction_stimulus = Fraction(alpha_stimulus / np.pi).limit_denominator()
phase_stimulus = 0
phase_fraction_stimulus = Fraction(phase_stimulus / np.pi).limit_denominator()

# Translation speeds
#vx_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
vx_values = [-20.0, -15.0, -10.0, -4.0, -2.0, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0]  # Translation speeds of stimulus
#vx_values = np.linspace(-20.0, 20.0, num=41).tolist()
#vx_values = np.arange(-20.0, 20.0 + 0.2, 0.2).tolist()

# --- Parameters of the Kernels ---
size = 16
sigma = 3.106
lambd = 8.0

alpha_values = [0, -1/64, -1/32, -1/16, -1/8, -1/4, -1/3, 1/64, 1/32, 1/16, 1/8, 1/4, 1/3]
alpha_values = [value * np.pi for value in alpha_values]

phase_values = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
phase_values = [value * np.pi for value in phase_values]

# --- Output storage ---
energy_matrix = np.zeros((len(alpha_values), len(phase_values), len(vx_values)))

# --- Main loop ---
for vx_index, vx in enumerate(vx_values):
    # Stimulus kernels with translation
    stimulus_sine = generate_rotational_sine_gabor_kernel_with_translation(
        size, size, size, sigma, lambd_stimulus, alpha_stimulus, phase_stimulus, vx
    )
    stimulus_cosine = generate_rotational_cosine_gabor_kernel_with_translation(
        size, size, size, sigma, lambd_stimulus, alpha_stimulus, phase_stimulus, vx
    )

    for alpha_index, alpha in enumerate(alpha_values):
        for phase_index, phase in enumerate(phase_values):
            # Filter kernels without translation
            filter_sine = generate_rotational_sine_gabor_kernel_3d(
                size, size, size, sigma, lambd, alpha, phase
            )
            filter_cosine = generate_rotational_cosine_gabor_kernel_3d(
                size, size, size, sigma, lambd, alpha, phase
            )

            # Correlation + Quadrature Energy
            conv_sine = np.sum(stimulus_sine * filter_sine)
            conv_cosine = np.sum(stimulus_cosine * filter_cosine)
            energy = conv_sine**2 + conv_cosine**2
            energy_matrix[alpha_index, phase_index, vx_index] = energy
            
"""
# --- Plotting ---
plt.figure(figsize=(14, 8))
for alpha_index, alpha in enumerate(alpha_values):
    for phase_index, phase in enumerate(phase_values):
        alpha_frac = Fraction(alpha / np.pi).limit_denominator()
        phase_frac = Fraction(phase / np.pi).limit_denominator()
        label = f'α={alpha_frac.numerator}/{alpha_frac.denominator}π, ϕ={phase_frac.numerator}/{phase_frac.denominator}π'
        plt.plot(vx_values, energy_matrix[alpha_index, phase_index], label=label, linewidth=1)

plt.xlabel('Translation Speed vx')
plt.ylabel('Curl Energy')
plt.title(f'Curl Energy vs. Translation Speed\nStimulus: α = {alpha_fraction_stimulus.numerator}/{alpha_fraction_stimulus.denominator}π, ϕ = {phase_fraction_stimulus.numerator}/{phase_fraction_stimulus.denominator}π')
plt.legend(fontsize=7.5, title='Kernel (α, ϕ)', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
"""

# --- Plotting with unique colors with continuous colormap for alpha, line styles for phase ---
plt.figure(figsize=(14, 8))

# Continuous colormap for alpha groups
base_cmap = cm.get_cmap('turbo')  # Or 'nipy_spectral', 'hsv', etc.

# Line styles to alternate for different phase values
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (5, 1))]

for alpha_index, alpha in enumerate(alpha_values):
    # Assign a unique color to each alpha (evenly spaced)
    base_color = base_cmap(alpha_index / (len(alpha_values) - 1))

    for phase_index, phase in enumerate(phase_values):
        # Format alpha and phase as fractions of π for labeling
        alpha_frac = Fraction(alpha / np.pi).limit_denominator()
        phase_frac = Fraction(phase / np.pi).limit_denominator()
        label = f'α={alpha_frac.numerator}/{alpha_frac.denominator}π, ϕ={phase_frac.numerator}/{phase_frac.denominator}π'

        # Select line style by phase
        line_style = line_styles[phase_index % len(line_styles)]

        # Plot the energy curve
        plt.plot(
            vx_values,
            energy_matrix[alpha_index, phase_index],
            label=label,
            color=base_color,
            linestyle=line_style,
            linewidth=1
        )

# Plot labeling
plt.xlabel('Translation Speed vx')
plt.ylabel('Curl Energy')
plt.title(
    f'Curl Energy vs. Translation Speed\n'
    f'Stimulus: α = {alpha_fraction_stimulus.numerator}/{alpha_fraction_stimulus.denominator}π, '
    f'ϕ = {phase_fraction_stimulus.numerator}/{phase_fraction_stimulus.denominator}π'
)

# Legend settings
plt.legend(fontsize=7, title='Kernel (α, ϕ)', ncol=3, loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

########################################################################################################################

# --- Plotting with unique colors with continuous colormap for alpha, line styles and markers for phase ---
plt.figure(figsize=(14, 8))

# Colormap for unique alpha hues
base_cmap = cm.get_cmap('turbo')  # Or 'nipy_spectral', 'hsv', etc.

# Style banks for phases
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (5, 1))]
markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'x']  # 8 markers, matching your 8 phase values

for alpha_index, alpha in enumerate(alpha_values):
    base_color = base_cmap(alpha_index / (len(alpha_values) - 1))

    for phase_index, phase in enumerate(phase_values):
        alpha_frac = Fraction(alpha / np.pi).limit_denominator()
        phase_frac = Fraction(phase / np.pi).limit_denominator()
        label = f'α={alpha_frac.numerator}/{alpha_frac.denominator}π, ϕ={phase_frac.numerator}/{phase_frac.denominator}π'

        # Select style and marker by phase
        line_style = line_styles[phase_index % len(line_styles)]
        marker_style = markers[phase_index % len(markers)]

        plt.plot(
            vx_values,
            energy_matrix[alpha_index, phase_index],
            label=label,
            color=base_color,
            linestyle=line_style,
            marker=marker_style,
            markersize=4,
            linewidth=1
        )

plt.xlabel('Translation Speed vx')
plt.ylabel('Curl Energy')
plt.title(
    f'Curl Energy vs. Translation Speed\n'
    f'Stimulus: α = {alpha_fraction_stimulus.numerator}/{alpha_fraction_stimulus.denominator}π, '
    f'ϕ = {phase_fraction_stimulus.numerator}/{phase_fraction_stimulus.denominator}π'
)
plt.legend(fontsize=7, title='Kernel (α, ϕ)', ncol=3, loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()



#%% Plot of Maximally Activated Kernel per Pixel 

# Display the array indicating the kernel index with the highest convolution value at each pixel
plt.imshow(max_activation_kernel_indices[:, :], cmap='viridis', vmin=0, vmax=38, interpolation='none')  # Assuming there are 39 kernels
plt.colorbar(ticks=range(39))  # Set ticks for each kernel index
#plt.imshow(max_activation_kernel_indices[:, :], cmap='viridis', vmin=0, vmax=32, interpolation='none')  # Assuming there are 33 kernels
#plt.colorbar(ticks=range(33))  # Set ticks for each kernel index
#plt.imshow(max_activation_kernel_indices[:, :], cmap='viridis', vmin=0, vmax=116, interpolation='none')  # Assuming there are 117 kernels
#plt.colorbar(ticks=range(117))  # Set ticks for each kernel index
#plt.imshow(max_activation_kernel_indices[:, :], cmap='viridis', vmin=0, vmax=311, interpolation='none')  # Assuming there are 117 kernels
#plt.colorbar(ticks=range(312))  # Set ticks for each kernel index
#plt.imshow(max_activation_kernel_indices[:, :], cmap='viridis', vmin=0, vmax=623, interpolation='none')  # Assuming there are 117 kernels
#plt.colorbar(ticks=range(624))  # Set ticks for each kernel index
plt.title('Kernel Index with Highest Convolution Value')
plt.show()



#%% Plot of Maximum Activation Value per Pixel

# Retrieve the maximum activation values for each pixel
max_activation_values = np.max(reshaped_results, axis=0)

# Plot of Maximally Activated Kernel per Pixel with activation values as brightness
plt.imshow(max_activation_values, cmap='gray', interpolation='none')
plt.colorbar()  
plt.title('Maximum Activation Value per Pixel')
plt.show()



#%% Plot of Alpha Value per Pixel for Fixed X-Position in Original Alpha Map

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
from matplotlib.colors import BoundaryNorm
from fractions import Fraction

### FOR SPECIFIC INTERVAL OF X-AXIS OF THE PREVIOUS PLOT, I.E. IN Y-DIRECTION OF ORGINAL ALPHA MAP ###

# === Configuration ===
fixed_x = 142  # x-position to slice vertically
y_positions = np.arange(max_activation_alpha_array.shape[0])
alpha_column = max_activation_alpha_array[:, fixed_x]

# === Color setup using the same symmetric mapping ===
unique_alphas = np.unique(alpha_values)

# Define boundaries between alpha levels for symmetric color assignment
alpha_boundaries = [unique_alphas[0] - 0.1] + \
                   list((unique_alphas[:-1] + unique_alphas[1:]) / 2) + \
                   [unique_alphas[-1] + 0.1]

# Colormap and normalization consistent with alpha map
cmap = coolwarm
norm = BoundaryNorm(alpha_boundaries, ncolors=cmap.N)
colors = cmap(norm(alpha_column))  # Map alpha values to RGBA using BoundaryNorm

# === Plotting ===
plt.figure(figsize=(10, 6))
for y, alpha_val, color in zip(y_positions, alpha_column, colors):
    plt.plot(y, alpha_val, 'o', color=color, markersize=3)

# === Optional: connect dots ===
# plt.plot(y_positions, alpha_column, color='gray', linewidth=1)

# === Colorbar with same ticks and labels as alpha map ===
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=unique_alphas, spacing='uniform')
cbar.set_label('Alpha Value')
cbar.set_ticklabels([f'{Fraction(alpha/np.pi).limit_denominator()}π' for alpha in unique_alphas])

# === Formatting ===
plt.title(f'Alpha Values Across Y-axis at x = {fixed_x}')
plt.xlabel('Pixel Position (y)')
plt.ylabel('Alpha Value')
plt.grid(True)

# Match y-axis ticks to alpha values and format as fractions of π
plt.yticks(
    ticks=unique_alphas,
    labels=[f'{Fraction(alpha/np.pi).limit_denominator()}π' for alpha in unique_alphas]
)

plt.tight_layout()
plt.show()



#%% Plot of Phase Value per Pixel for Fixed X-Position in Original Phase Map

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
from matplotlib.colors import BoundaryNorm
from fractions import Fraction

### FOR SPECIFIC INTERVAL OF X-AXIS OF THE PREVIOUS PLOT, I.E. IN Y-DIRECTION OF ORIGINAL PHASE MAP ###

# === Configuration ===
fixed_x = 142  # x-position to slice vertically
y_positions = np.arange(max_activation_phase_array.shape[0])  # Assumes shape (285, 285)
phase_column = max_activation_phase_array[:, fixed_x]

# === Color setup using the same symmetric mapping ===
phase_values_array = np.array(phase_values)  # Ensure NumPy array
unique_phases = np.unique(phase_values_array)

# Define boundaries between phase levels for symmetric color assignment
phase_boundaries = [unique_phases[0] - 0.1] + \
                   list((unique_phases[:-1] + unique_phases[1:]) / 2) + \
                   [unique_phases[-1] + 0.1]

# Colormap and normalization consistent with phase map
cmap = coolwarm
norm = BoundaryNorm(phase_boundaries, ncolors=cmap.N)
colors = cmap(norm(phase_column))  # Map phase values to RGBA using BoundaryNorm

# === Plotting ===
plt.figure(figsize=(10, 6))
for y, phase_val, color in zip(y_positions, phase_column, colors):
    plt.plot(y, phase_val, 'o', color=color, markersize=3)

# === Optional: connect dots ===
# plt.plot(y_positions, phase_column, color='gray', linewidth=1)

# === Colorbar with same ticks and labels as phase map ===
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=unique_phases, spacing='uniform')
cbar.set_label('Phase Value')
cbar.set_ticklabels([f'{Fraction(phase/np.pi).limit_denominator()}π' for phase in unique_phases])

# === Formatting ===
plt.title(f'Phase Values Across Y-axis at x = {fixed_x}')
plt.xlabel('Pixel Position (y)')
plt.ylabel('Phase Value')
plt.grid(True)

# Match y-axis ticks to phase values and format as fractions of π
plt.yticks(
    ticks=unique_phases,
    labels=[f'{Fraction(phase/np.pi).limit_denominator()}π' for phase in unique_phases]
)

plt.tight_layout()
plt.show()



#%% Plot of Maximum Activation Value per Pixel per Alpha Value 

from matplotlib.cm import coolwarm
from matplotlib.colors import Normalize

# Assuming summed_quadrature_convolutions is of shape (3, 13, 8, 285, 285)

# Take the max across the lambda and phase dimensions (axis 0 and 2)
max_activation_values_alpha = np.max(summed_quadrature_convolutions, axis=(0, 2))  # shape: (13, 285, 285)

# Get global maximum energy across all alpha indices
global_max_energy_alpha = np.max(max_activation_values_alpha)


# Choose a fixed x-position (e.g., x = 142, the center of the image)
fixed_x = 142  # You can set this to any value from 0 to 284

# Plot
plt.figure(figsize=(10, 6))
y_positions = np.arange(285)  # Pixel positions along y-axis


### FOR COOLWARM COLOR MAPPING ###

# Example alpha values: np.linspace(-13/16, 13/16, 13)
alpha_values_array = np.array(alpha_values)  # Ensure it's a NumPy array

# Normalize alpha values to [0, 1] for colormap
norm = Normalize(vmin=np.min(alpha_values_array), vmax=np.max(alpha_values_array))
cmap = coolwarm

# Define unique linestyles for distinction
linestyles = ['-', '--', '-.', ':'] * ((len(alpha_values_array) // 4) + 1)

for alpha_idx, alpha in enumerate(alpha_values_array):
    energy_values = max_activation_values_alpha[alpha_idx, :, fixed_x]
    energy_values = energy_values / global_max_energy_alpha  # Normalize
    
    # Color from coolwarm based on alpha
    color = cmap(norm(alpha))
    
    # Fraction label (e.g., 1/16π)
    label = f'{Fraction(alpha/np.pi).limit_denominator()}π'
    
    # For different linestyles
    linestyle = linestyles[alpha_idx]
    
    plt.plot(y_positions, energy_values, label=label, color=color, linestyle=linestyle, marker='o', markersize=4)  # Punkte anstatt Linien mit: marker='o', markersize=2, linestyle='none'. Punkte und Linien: marker='o', markersize=2

plt.title(f'Motion Energy Profiles Across Y-axis at x = {fixed_x}')
plt.xlabel('Pixel Position (y)')
plt.ylabel('Motion Energy')
plt.legend(title='Alpha Value')
plt.grid(True)
plt.tight_layout()
plt.show()


### FOR SPECIFIC INTERVAL OF X-AXIS OF THE PREVIOUS PLOT, I.E. IN Y-DIRECTION OF ORGINAL ALPHA MAP ###

y_start = 50
y_end = 90

# Define the y range
y_positions = np.arange(y_start, y_end)

plt.figure(figsize=(10, 6))

# Define unique linestyles for distinction
linestyles = ['-', '--', '-.', ':'] * ((len(alpha_values_array) // 4) + 1)

for alpha_idx, alpha in enumerate(alpha_values_array):
    full_energy = max_activation_values_alpha[alpha_idx, :, fixed_x]
    energy_values = full_energy[y_start:y_end]  # Slice only the interval
    energy_values = energy_values / global_max_energy_alpha  # Normalize

    color = cmap(norm(alpha))
    label = f'{Fraction(alpha/np.pi).limit_denominator()}π'
    # For different linestyles
    linestyle = linestyles[alpha_idx]

    plt.plot(y_positions, energy_values, label=label, color=color, linestyle=linestyle, marker='o', markersize=4)  # Punkte anstatt Linien mit: marker='o', markersize=2, linestyle='none'. Punkte und Linien: marker='o', markersize=2

plt.title(f'Motion Energy Profiles (Y: {y_start}–{y_end}) at x = {fixed_x}')
plt.xlabel('Pixel Position (y)')
plt.ylabel('Motion Energy')
plt.legend(title='Alpha Value')
plt.grid(True)
plt.tight_layout()
plt.show()



#%% Plot of Maximum Activation Value per Pixel per Phase Value

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
from matplotlib.colors import Normalize
from fractions import Fraction

# Assume the following exist:
# summed_quadrature_convolutions: shape (3, 13, 8, 285, 285)
# phase_values: list or array of 8 phase values
# fixed_x: int, e.g. 142

# Max over lambda and alpha to get energy per phase
max_activation_values_phase = np.max(summed_quadrature_convolutions, axis=(0, 1))  # shape: (8, 285, 285)

# Global max for normalization
global_max_energy_phase = np.max(max_activation_values_phase)


# Setup
fixed_x = 142
y_positions = np.arange(285)


### FOR COOLWARM COLOR MAPPING ###

phase_values_array = np.array(phase_values)

# Colormap and normalization
norm = Normalize(vmin=np.min(phase_values_array), vmax=np.max(phase_values_array))
cmap = coolwarm

# Optional linestyles
linestyles = ['-', '--', '-.', ':'] * ((len(phase_values_array) // 4) + 1)

# Plot
plt.figure(figsize=(10, 6))

for phase_idx, phase in enumerate(phase_values_array):
    energy_values = max_activation_values_phase[phase_idx, :, fixed_x]
    energy_values = energy_values / global_max_energy_phase

    color = cmap(norm(phase))
    label = f'{Fraction(phase/np.pi).limit_denominator()}π'
    linestyle = linestyles[phase_idx]

    plt.plot(y_positions, energy_values, label=label, color=color, marker='o', markersize=4)

plt.title(f'Motion Energy Profiles Across Y-axis at x = {fixed_x}')
plt.xlabel('Pixel Position (y)')
plt.ylabel('Motion Energy')
plt.legend(title='Phase Value')
plt.grid(True)
plt.tight_layout()
plt.show()


### FOR SPECIFIC INTERVAL OF X-AXIS OF THE PREVIOUS PLOT, I.E. IN Y-DIRECTION OF ORGINAL ALPHA MAP ###

# Define interval
y_start = 0
y_end = 50
y_positions = np.arange(y_start, y_end)

plt.figure(figsize=(10, 6))

for phase_idx, phase in enumerate(phase_values_array):
    full_energy = max_activation_values_phase[phase_idx, :, fixed_x]
    energy_values = full_energy[y_start:y_end]
    energy_values = energy_values / global_max_energy_phase

    color = cmap(norm(phase))
    label = f'{Fraction(phase/np.pi).limit_denominator()}π'
    linestyle = linestyles[phase_idx]

    plt.plot(y_positions, energy_values, label=label, color=color,
             linestyle=linestyle, marker='o', markersize=4)

plt.title(f'Motion Energy Profiles (Y: {y_start}–{y_end}) at x = {fixed_x}')
plt.xlabel('Pixel Position (y)')
plt.ylabel('Motion Energy')
plt.legend(title='Phase Value')
plt.grid(True)
plt.tight_layout()
plt.show()



#%% Plot of Maximum Activation Value per Pixel per Alpha and Phase Value 

import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm
from matplotlib.colors import Normalize
from fractions import Fraction
import numpy as np

# Prepare parameters
fixed_x = 142
alpha_array = np.array(alpha_values)
phase_array = np.array(phase_values)
norm_alpha = Normalize(vmin=np.min(alpha_array), vmax=np.max(alpha_array))
cmap = coolwarm

# Define markers for phases and linestyles to improve visibility
phase_markers = ['o', 's', 'D', '^', 'v', '>', '<', 'P']
phase_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 2)), (0, (2, 1, 2, 1))]

# Get global max for normalization across all (alpha, phase)
max_energy_global = np.max(summed_quadrature_convolutions)

plt.figure(figsize=(14, 8))
y_positions = np.arange(285)


### FOR COOLWARM COLOR MAPPING ###

# Loop over all (alpha, phase) combinations
for alpha_idx, alpha in enumerate(alpha_array):
    for phase_idx, phase in enumerate(phase_array):
        energy = summed_quadrature_convolutions[:, alpha_idx, phase_idx, :, fixed_x]  # shape: (lambda, y)
        energy = np.max(energy, axis=0)  # max over lambda -> shape: (y,)
        energy /= max_energy_global  # normalize

        color = cmap(norm_alpha(alpha))
        label = f'{Fraction(alpha / np.pi).limit_denominator()}π, {Fraction(phase / np.pi).limit_denominator()}π'
        linestyle = phase_linestyles[phase_idx % len(phase_linestyles)]
        marker = phase_markers[phase_idx % len(phase_markers)]

        plt.plot(y_positions, energy, label=label, color=color,
                 linestyle=linestyle, marker=marker, markersize=3, linewidth=1)

plt.title(f'Motion Energy (Alpha × Phase Combinations) at x = {fixed_x}')
plt.xlabel('Pixel Position (y)')
plt.ylabel('Normalized Motion Energy')
plt.grid(True)
plt.legend(fontsize='small', title='α, φ', ncol=2, loc='upper right', bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.show()


### FOR SPECIFIC INTERVAL OF X-AXIS OF THE PREVIOUS PLOT, I.E. IN Y-DIRECTION OF ORGINAL ALPHA MAP ###

y_start, y_end = 50, 90
y_positions = np.arange(y_start, y_end)

plt.figure(figsize=(14, 8))

for alpha_idx, alpha in enumerate(alpha_array):
    for phase_idx, phase in enumerate(phase_array):
        energy = summed_quadrature_convolutions[:, alpha_idx, phase_idx, :, fixed_x]
        energy = np.max(energy, axis=0)[y_start:y_end]  # max over lambda, slice y
        energy /= max_energy_global

        color = cmap(norm_alpha(alpha))
        label = f'{Fraction(alpha / np.pi).limit_denominator()}π, {Fraction(phase / np.pi).limit_denominator()}π'
        linestyle = phase_linestyles[phase_idx % len(phase_linestyles)]
        marker = phase_markers[phase_idx % len(phase_markers)]

        plt.plot(y_positions, energy, label=label, color=color,
                 linestyle=linestyle, marker=marker, markersize=3, linewidth=1)

plt.title(f'Motion Energy (Y: {y_start}-{y_end}) at x = {fixed_x}')
plt.xlabel('Pixel Position (y)')
plt.ylabel('Normalized Motion Energy')
plt.grid(True)
plt.legend(fontsize='small', title='α, φ', ncol=2, loc='upper right', bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.show()



#%% Plot of Maximum Activation Value per Pixel for One Alpha Value

from matplotlib.cm import coolwarm
from matplotlib.colors import Normalize
from fractions import Fraction

# Assuming these are already defined
# alpha_values, summed_quadrature_convolutions

# Take the max across the lambda and phase dimensions (axis 0 and 2)
max_activation_values_alpha = np.max(summed_quadrature_convolutions, axis=(0, 2))  # shape: (13, 285, 285)

# Get global maximum energy across all alpha indices
global_max_energy_alpha = np.max(max_activation_values_alpha)


# Choose a fixed x-position
fixed_x = 142

# Define the alpha you want to plot (example: -1/2 π)
desired_alpha = 1/16 * np.pi

# Find index of that alpha
alpha_idx = np.where(np.isclose(alpha_values_array, desired_alpha))[0][0]

# Get energy values
energy_values = max_activation_values_alpha[alpha_idx, :, fixed_x]
energy_values = energy_values / global_max_energy_alpha  # Normalize

# Normalize alpha for color mapping
norm = Normalize(vmin=np.min(alpha_values_array), vmax=np.max(alpha_values_array))
cmap = coolwarm
color = cmap(norm(alpha_values_array[alpha_idx]))

# Plot
plt.figure(figsize=(10, 6))
y_positions = np.arange(285)
label = f'{Fraction(alpha_values[alpha_idx]/np.pi).limit_denominator()}π'

plt.plot(y_positions, energy_values, label=label, color=color, marker='o', linestyle='none', markersize=3)

plt.title(f'Motion Energy Profile at x = {fixed_x} for α = {label}')
plt.xlabel('Pixel Position (y)')
plt.ylabel('Motion Energy')
plt.legend(title='Alpha Value')
plt.grid(True)
plt.tight_layout()
plt.show()



#%% Plot of Maximum Activation Value per Pixel for One Phase Value

from matplotlib.cm import coolwarm
from matplotlib.colors import Normalize
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt

# Assuming these are already defined:
# phase_values, summed_quadrature_convolutions

# Take the max across the lambda and alpha dimensions (axis 0 and 1)
max_activation_values_phase = np.max(summed_quadrature_convolutions, axis=(0, 1))  # shape: (8, 285, 285)

# Get global maximum energy across all phase indices
global_max_energy_phase = np.max(max_activation_values_phase)

# Choose a fixed x-position
fixed_x = 142

# Define the phase you want to plot (example: 1/4 π)
desired_phase = 3/2 * np.pi

# Ensure phase_values is a NumPy array
phase_values_array = np.array(phase_values)

# Find index of the desired phase
phase_idx = np.where(np.isclose(phase_values_array, desired_phase))[0][0]

# Get energy values
energy_values = max_activation_values_phase[phase_idx, :, fixed_x]
energy_values = energy_values / global_max_energy_phase  # Normalize

# Normalize phase for color mapping
norm = Normalize(vmin=np.min(phase_values_array), vmax=np.max(phase_values_array))
cmap = coolwarm
color = cmap(norm(phase_values_array[phase_idx]))

# Plot
plt.figure(figsize=(10, 6))
y_positions = np.arange(285)
label = f'{Fraction(phase_values_array[phase_idx]/np.pi).limit_denominator()}π'

plt.plot(y_positions, energy_values, label=label, color=color, marker='o', linestyle='none', markersize=3)

plt.title(f'Motion Energy Profile at x = {fixed_x} for Phase = {label}')
plt.xlabel('Pixel Position (y)')
plt.ylabel('Motion Energy')
plt.legend(title='Phase Value')
plt.grid(True)
plt.tight_layout()
plt.show()



#%% Histogram Plot of Maximum Activation Values (Abundance over Max Activation Values)

# Flatten the 2D array into a 1D array
flattened_values = max_activation_values.flatten()

# Filter out values in the range you're interested in
start_value = 0  # Custom start value for the range
end_value = 0.09  # Custom end value for the range
#end_value = np.max(flattened_values)  # Set end value as maximum activation value
filtered_values = flattened_values[(flattened_values >= start_value) & (flattened_values <= end_value)]

# Define the bin size
bin_size = 0.0001

# Create bin edges from start_value to end_value with the specified bin size
bins = np.arange(start_value, end_value + bin_size, bin_size)

# Plot the histogram using the custom bin edges
plt.figure(figsize=(8, 6))
plt.hist(filtered_values, bins=bins, edgecolor='black')  # density=True for Relative Abundance

# Set axis labels and title
plt.xlabel('Activation Values')
plt.ylabel('Absolute Abundance (Frequency)')
plt.title(f'Distribution of Activation Values ({start_value} to {end_value}) with bin size {bin_size}')

# Adjust axis limits to match the custom start and end values
plt.xlim(start_value, end_value)
plt.ylim(0, 1000)

# Display the plot
plt.show()



#%% Histogram Fit to Find Threshold (Versuch 1)

from scipy.optimize import curve_fit

# Define a downward-facing parabola function for fitting
def downward_parabola(x, a, b, c):
    return a * x**2 + b * x + c

# Create the histogram
hist, bin_edges = np.histogram(filtered_values, bins=30, density=True)

# Calculate the bin centers
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Fit the histogram using curve fitting
#initial_guess = [-1e-6, 0, 0]  # Initial guess for parameters
initial_guess = [1, 0, 0]  # Initial guess for parameters
popt, pcov = curve_fit(downward_parabola, bin_centers, hist, p0=initial_guess)

# Extract the parameters
a, b, c = popt
print(f'Fitted parameters:\n a (quadratic term) = {a}\n b (linear term) = {b}\n c (constant term) = {c}')

parameter_errors = np.sqrt(np.diag(pcov))
print(f'Parameter errors: {parameter_errors}')

# Generate x values for the fitted curve
x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
y_fit = downward_parabola(x_fit, *popt)

# Find the vertex of the parabola
vertex_x = -b / (2 * a)
vertex_y = downward_parabola(vertex_x, *popt)

# Plotting the histogram and the fitted curve
plt.figure(figsize=(10, 6))
plt.hist(filtered_values, bins=30, density=True, alpha=0.5, edgecolor='black', label='Histogram')
plt.plot(x_fit, y_fit, color='red', label='Fitted Downward Parabola')
plt.scatter(vertex_x, vertex_y, color='blue', zorder=5, label=f'Maximum at x={vertex_x:.2f}, y={vertex_y:.2e}')

# Set axis labels and title
plt.xlabel('Activation Values')
plt.ylabel('Relative Abundance (Frequency)')
plt.title('Histogram with Downward-Facing Parabola Fit')
plt.legend()
plt.xlim(bin_edges[0], bin_edges[-1])
plt.show()

# Output the maximum found at the vertex
print(f'Maximum of fitted parabola at x={vertex_x:.2f} with value y={vertex_y:.2e}')


from scipy.optimize import curve_fit

# Define a downward-facing parabola function for fitting
def downward_parabola(x, a, b, c):
    return a * x**2 + b * x + c

# Create the histogram
hist, bin_edges = np.histogram(filtered_values, bins=30, density=True)

# Calculate the bin centers
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Fit the histogram using curve fitting
#initial_guess = [-1e-6, 0, 0]  # Initial guess for parameters
initial_guess = [1, 0, 0]  # Initial guess for parameters
popt, pcov = curve_fit(downward_parabola, bin_centers, hist, p0=initial_guess)

# Extract the parameters
a, b, c = popt
print(f'Fitted parameters:\n a (quadratic term) = {a}\n b (linear term) = {b}\n c (constant term) = {c}')

parameter_errors = np.sqrt(np.diag(pcov))
print(f'Parameter errors: {parameter_errors}')

# Generate x values for the fitted curve
x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
y_fit = downward_parabola(x_fit, *popt)

# Find the vertex of the parabola
vertex_x = -b / (2 * a)
vertex_y = downward_parabola(vertex_x, *popt)

# Plotting the histogram and the fitted curve
plt.figure(figsize=(10, 6))
plt.hist(filtered_values, bins=30, density=True, alpha=0.5, edgecolor='black', label='Histogram')
plt.plot(x_fit, y_fit, color='red', label='Fitted Downward Parabola')
plt.scatter(vertex_x, vertex_y, color='blue', zorder=5, label=f'Maximum at x={vertex_x:.2f}, y={vertex_y:.2e}')

# Set axis labels and title
plt.xlabel('Activation Values')
plt.ylabel('Relative Abundance (Frequency)')
plt.title('Histogram with Downward-Facing Parabola Fit')
plt.legend()
plt.xlim(bin_edges[0], bin_edges[-1])
plt.show()

# Output the maximum found at the vertex
print(f'Maximum of fitted parabola at x={vertex_x:.2f} with value y={vertex_y:.2e}')



#%% Plot of Maximum Activation Kernel Index per Pixel with Threshold

# Define the threshold value
#threshold = 1e-26  # Set your desired threshold value here
#threshold = 1  # Set your desired threshold value here
#threshold = 2  # Set your desired threshold value here
#threshold = 20  # Set your desired threshold value here
#threshold = 100  
#threshold = 200  
#threshold = 500  
#threshold = 0.0002  
#threshold = 0.0001
#threshold = 0.00001
threshold = 0.000001
#threshold = 0.000000001
#threshold = 0.02

# Apply the threshold: set kernel indices to 0 where the maximum activation value is below the threshold
max_activation_kernel_indices[max_activation_values < threshold] = 0

# Display the array indicating the kernel index with the highest convolution value at each pixel
#plt.imshow(max_activation_kernel_indices[:, :], cmap='viridis', vmin=0, vmax=38, interpolation='none')  # Assuming there are 39 kernels
#plt.colorbar(ticks=range(39))  # Set ticks for each kernel index
#plt.imshow(max_activation_kernel_indices[:, :], cmap='viridis', vmin=0, vmax=116, interpolation='none')  # Assuming there are 117 kernels
#plt.colorbar(ticks=range(117))  # Set ticks for each kernel index
plt.imshow(max_activation_kernel_indices[:, :], cmap='viridis', vmin=0, vmax=311, interpolation='none')  # Assuming there are 117 kernels
plt.colorbar(ticks=range(312)) 
plt.title(f'Kernel Index with Highest Convolution Value \n(Threshold: {threshold:.1e})', pad = 20)
plt.show()



#%% Plot of Parameters Lambda, Alpha corresponding to Maximally Activated Kernel per Pixel 4 (Ansatz 1)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


def index_to_parameters_alpha(index, lambd_values, alpha_values):
    """
    Maps a linear index to its corresponding lambda and alpha parameters.
    
    :param index: an integer index (a flattened index from the activation map)
    :param lambd_values: list of lambda values used in kernels
    :param alpha_values: list of alpha values used in kernels
    :return: Tuple (lambda, alpha)
    """
    num_lambda = len(lambd_values)  # Number of unique lambdas
    num_alpha = len(alpha_values)   # Number of unique alpha values
    
    # Calculate the lambda and alpha indices based on the linear index
    lambda_index = index // num_alpha  # Integer division to get the lambda index
    alpha_index = index % num_alpha   # Modulo to get the alpha index
    
    # Return the corresponding lambda and alpha values
    return lambd_values[lambda_index], alpha_values[alpha_index]

# Create a map for lambda and alpha
lambda_map = np.zeros_like(max_activation_kernel_indices, dtype=float)
alpha_map = np.zeros_like(max_activation_kernel_indices, dtype=float)

# Fill the maps with corresponding lambda and alpha values
for idx in range(max_activation_kernel_indices.size):
    y, x = np.unravel_index(idx, max_activation_kernel_indices.shape)
    lambda_val, alpha_val = index_to_parameters_alpha(max_activation_kernel_indices[y, x], lambd_values, alpha_values)
    alpha_map[y, x] = alpha_val
    lambda_map[y, x] = lambda_val

"""
# Plot Lambda Map
cmap_lambda = plt.get_cmap('viridis')  # Or any other perceptually uniform colormap
fig, ax = plt.subplots()
im_lambda = ax.imshow(lambda_map, cmap=cmap_lambda, interpolation='none')
cbar_lambda = fig.colorbar(im_lambda)
cbar_lambda.set_label('Lambda Value')
plt.title('Map of Lambda Values Corresponding to Max Kernel Activation')
plt.show()

# Plot Alpha Map
cmap_alpha = plt.get_cmap('viridis')
fig, ax = plt.subplots()
im_alpha = ax.imshow(alpha_map, cmap=cmap_alpha, interpolation='none')
cbar_alpha = fig.colorbar(im_alpha)
cbar_alpha.set_label('Alpha Value')
plt.title('Map of Alpha Values Corresponding to Max Kernel Activation')
plt.show()
"""


### Alpha map ######################################################################################


# Assuming you have defined cmap and alpha_values above, check these:
unique_alphas = np.unique(alpha_values)
print("Unique alphas in Map:", unique_alphas)  # This prints the actual unique values used in the map

# Define a colormap with enough distinct colors for each unique theta value
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink'])  # Change colors if more/fewer unique thetas exist
cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink', 'orange', 'brown', 'gray', 'olive', 'cyan']) 
# Reverse the colormap so it corresponds to the reversed values
cmap_reversed = cmap.reversed()  # Reverses the list of colors starting with the last entry and ending with the first

# Define boundaries that split the range between unique alpha values
# Correctly placing these boundaries is crucial for distinct color regions
#alpha_boundaries = [unique_alphas[0] - 0.01] + list((unique_alphas[:-1] + unique_alphas[1:]) / 2) + [unique_alphas[-1] + 0.01]
alpha_boundaries = [unique_alphas[0] - 0.1,  # a bit less than the smallest alpha
                  (unique_alphas[1] + unique_alphas[0]) / 2,  # Midpoint between first and second
                  (unique_alphas[2] + unique_alphas[1]) / 2,  # Midpoint between second and third
                  (unique_alphas[3] + unique_alphas[2]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[4] + unique_alphas[3]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[5] + unique_alphas[4]) / 2,  # Midpoint between fifth and sixth
                  (unique_alphas[6] + unique_alphas[5]) / 2,  # Midpoint between first and second
                  (unique_alphas[7] + unique_alphas[6]) / 2,  # Midpoint between second and third
                  (unique_alphas[8] + unique_alphas[7]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[9] + unique_alphas[8]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[10] + unique_alphas[9]) / 2, 
                  unique_alphas[10] + 0.1]  # a bit more than the largest alpha

#norm = BoundaryNorm(alpha_boundaries, ncolors=cmap.N)
norm = BoundaryNorm(alpha_boundaries, ncolors=cmap_reversed.N)

# Plotting alpha map with correct boundaries
fig, ax = plt.subplots()
#im = ax.imshow(alpha_map, cmap=cmap, norm=norm, interpolation='none')
im = ax.imshow(alpha_map, cmap=cmap_reversed, norm=norm, interpolation='none')
cbar = fig.colorbar(im, ticks=unique_alphas, spacing='proportional')
cbar.set_label('Alpha Value')

# Properly label each tick with its corresponding alpha value
#alpha_labels = [f'{alpha:.3f}π' for alpha in unique_alphas]
alpha_labels = [f'{Fraction(alpha/np.pi).limit_denominator()}π' for alpha in unique_alphas]  # Give alpha labels in fractions
cbar.ax.set_yticklabels(alpha_labels)

plt.title('Map of Alpha Values Corresponding to Max Kernel Activation')
plt.show()


### Lambda map ######################################################################################


# Assuming you have defined cmap and lambd_values above, check these:
unique_lambdas = np.unique(lambd_values)
print("Unique Lambdas in Map:", unique_lambdas)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
cmap = ListedColormap(['red', 'green'])  # Adjust this list if you have more/fewer unique lambdas

# We must ensure boundaries enclose the actual unique lambda values used
# Let's set manual boundaries for three values, assuming lambd_values are ranged properly
#lambda_boundaries = [unique_lambdas[0] - 0.1,  # a bit less than the smallest lambda
#        (unique_lambdas[1] + unique_lambdas[0]) / 2,  # Midpoint between first and second
#        (unique_lambdas[2] + unique_lambdas[1]) / 2,  # Midpoint between second and third
#        (unique_lambdas[3] + unique_lambdas[2]) / 2,  # Midpoint between third and fourth
#        unique_lambdas[3] + 0.1]  # a bit more than the largest lambda

lambda_boundaries = [0, 24.0, 40.0] 
    
norm = BoundaryNorm(lambda_boundaries, ncolors=cmap.N)

# Plotting lambda map with correct boundaries
fig, ax = plt.subplots()
im = ax.imshow(lambda_map, cmap=cmap, norm=norm, interpolation='none')
cbar = fig.colorbar(im, ticks=unique_lambdas, spacing='proportional')
cbar.set_label('Lambda Value')

# Properly label each tick with its corresponding lambda value
lambda_labels = [f'{lambd:.2f}' for lambd in unique_lambdas]
cbar.ax.set_yticklabels(lambda_labels)

plt.title('Map of Lambda Values Corresponding to Max Kernel Activation')
plt.show()



#%% Plot of Parameters Lambda, Alpha corresponding to Maximally Activated Kernel per Pixel 4 using Parameter Arrays (Ansatz 2)

import matplotlib.pyplot as plt  
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


### Alpha map ######################################################################################


# Define the region to plot (example: rows 50 to 100, and columns 150 to 200)
row_start, row_end = 450, 750
col_start, col_end = 350, 650

# Assuming you have defined cmap and alpha_values above, check these:
unique_alphas = np.unique(alpha_values)
print("Unique alphas in Map:", unique_alphas)  # This prints the actual unique values used in the map

# Define a colormap with enough distinct colors for each unique theta value
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink'])  # Change colors if more/fewer unique thetas exist
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink', 'orange', 'brown', 'gray', 'olive', 'cyan'])  
cmap = plt.cm.coolwarm
# Reverse the colormap so it corresponds to the reversed values
#cmap_reversed = cmap.reversed()  # Reverses the list of colors starting with the last entry and ending with the first

# Define boundaries that split the range between unique alpha values
# Correctly placing these boundaries is crucial for distinct color regions
#alpha_boundaries = [unique_alphas[0] - 0.01] + list((unique_alphas[:-1] + unique_alphas[1:]) / 2) + [unique_alphas[-1] + 0.01]
alpha_boundaries = [unique_alphas[0] - 0.1,  # a bit less than the smallest alpha
                  (unique_alphas[1] + unique_alphas[0]) / 2,  # Midpoint between first and second
                  (unique_alphas[2] + unique_alphas[1]) / 2,  # Midpoint between second and third
                  (unique_alphas[3] + unique_alphas[2]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[4] + unique_alphas[3]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[5] + unique_alphas[4]) / 2,  # Midpoint between fifth and sixth
                  (unique_alphas[6] + unique_alphas[5]) / 2,  # Midpoint between first and second
                  (unique_alphas[7] + unique_alphas[6]) / 2,  # Midpoint between second and third
                  (unique_alphas[8] + unique_alphas[7]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[9] + unique_alphas[8]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[10] + unique_alphas[9]) / 2, 
                  (unique_alphas[11] + unique_alphas[10]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[12] + unique_alphas[11]) / 2,
                  unique_alphas[12] + 0.1]
                  #unique_alphas[10] + 0.1]  # a bit more than the largest alpha

norm = BoundaryNorm(alpha_boundaries, ncolors=cmap.N)
#norm = BoundaryNorm(alpha_boundaries, ncolors=cmap_reversed.N)

# Plotting alpha map with correct boundaries
fig, ax = plt.subplots()
#im = ax.imshow(alpha_map, cmap=cmap, norm=norm, interpolation='none')
#im = ax.imshow(max_activation_alpha_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_alpha_array, cmap=cmap_reversed, norm=norm, interpolation='none') 

# Plotting specific region in the colormap
im = ax.imshow(max_activation_alpha_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_alpha_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_alphas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Alpha Value')

# Properly label each tick with its corresponding alpha value
#alpha_labels = [f'{alpha/np.pi:.3f}π' for alpha in unique_alphas]
alpha_labels = [f'{Fraction(alpha/np.pi).limit_denominator()}π' for alpha in unique_alphas]  # Give alpha labels in fractions
cbar.ax.set_yticklabels(alpha_labels)

#plt.title('Map of Alpha Values Corresponding to \nMax Kernel Activation', pad = 20)
plt.title(f'Map of Alpha Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end})', pad = 20)
plt.show()


### Lambda map ######################################################################################


# Assuming you have defined cmap and lambd_values above, check these:
unique_lambdas = np.unique(lambd_values)
print("Unique Lambdas in Map:", unique_lambdas)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
#cmap = ListedColormap(['red', 'green'])  # Adjust this list if you have more/fewer unique lambdas
cmap = ListedColormap(['blue', 'red', 'green'])  

# We must ensure boundaries enclose the actual unique lambda values used
# Let's set manual boundaries for three values, assuming lambd_values are ranged properly
#lambda_boundaries = [unique_lambdas[0] - 0.1,  # a bit less than the smallest lambda
#        (unique_lambdas[1] + unique_lambdas[0]) / 2,  # Midpoint between first and second
#        (unique_lambdas[2] + unique_lambdas[1]) / 2,  # Midpoint between second and third
#        (unique_lambdas[3] + unique_lambdas[2]) / 2,  # Midpoint between third and fourth
#        unique_lambdas[3] + 0.1]  # a bit more than the largest lambda

#lambda_boundaries = [0, 24.0, 40.0] 
#lambda_boundaries = [0, 12.0, 24.0, 40.0] 
#lambda_boundaries = [0, 12.0, 20.0]
lambda_boundaries = [0, 5.0, 7.0, 9.0] 
    
norm = BoundaryNorm(lambda_boundaries, ncolors=cmap.N)

# Plotting lambda map with correct boundaries
fig, ax = plt.subplots()
#im = ax.imshow(max_activation_lambd_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_lambd_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Plotting specific region in the colormap
im = ax.imshow(max_activation_lambd_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_lambd_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_lambdas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Lambda Value')

# Properly label each tick with its corresponding lambda value
lambda_labels = [f'{lambd:.2f}' for lambd in unique_lambdas]
cbar.ax.set_yticklabels(lambda_labels)

#plt.title('Map of Lambda Values Corresponding to \nMax Kernel Activation', pad = 20)
plt.title(f'Map of Lambda Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end})', pad = 20)
plt.show()



#%% Plot of Parameters Lambda, Alpha corresponding to Maximally Activated Kernel per Pixel 4 using Parameter Arrays (Ansatz 2). With Kernel Size as separate Parameter Dimension

import matplotlib.pyplot as plt  
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


### Alpha map ######################################################################################


# Define the region to plot (example: rows 50 to 100, and columns 150 to 200)
#row_start, row_end = 450, 750
#col_start, col_end = 350, 650

# Assuming you have defined cmap and alpha_values above, check these:
unique_alphas = np.unique(alpha_values)
print("Unique alphas in Map:", unique_alphas)  # This prints the actual unique values used in the map

# Define a colormap with enough distinct colors for each unique theta value
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink'])  # Change colors if more/fewer unique thetas exist
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink', 'orange', 'brown', 'gray', 'olive', 'cyan'])  
cmap = plt.cm.coolwarm
# Reverse the colormap so it corresponds to the reversed values
#cmap_reversed = cmap.reversed()  # Reverses the list of colors starting with the last entry and ending with the first

# Define boundaries that split the range between unique alpha values
# Correctly placing these boundaries is crucial for distinct color regions
#alpha_boundaries = [unique_alphas[0] - 0.01] + list((unique_alphas[:-1] + unique_alphas[1:]) / 2) + [unique_alphas[-1] + 0.01]
alpha_boundaries = [unique_alphas[0] - 0.1,  # a bit less than the smallest alpha
                  (unique_alphas[1] + unique_alphas[0]) / 2,  # Midpoint between first and second
                  (unique_alphas[2] + unique_alphas[1]) / 2,  # Midpoint between second and third
                  (unique_alphas[3] + unique_alphas[2]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[4] + unique_alphas[3]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[5] + unique_alphas[4]) / 2,  # Midpoint between fifth and sixth
                  (unique_alphas[6] + unique_alphas[5]) / 2,  # Midpoint between first and second
                  (unique_alphas[7] + unique_alphas[6]) / 2,  # Midpoint between second and third
                  (unique_alphas[8] + unique_alphas[7]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[9] + unique_alphas[8]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[10] + unique_alphas[9]) / 2, 
                  (unique_alphas[11] + unique_alphas[10]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[12] + unique_alphas[11]) / 2,
                  unique_alphas[12] + 0.1]  # a bit more than the largest alpha

norm = BoundaryNorm(alpha_boundaries, ncolors=cmap.N)
#norm = BoundaryNorm(alpha_boundaries, ncolors=cmap_reversed.N)

# Plotting alpha map with correct boundaries
fig, ax = plt.subplots()
#im = ax.imshow(alpha_map, cmap=cmap, norm=norm, interpolation='none')
im = ax.imshow(max_activation_alpha_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_alpha_array, cmap=cmap_reversed, norm=norm, interpolation='none') 

# Plotting specific region in the colormap
#im = ax.imshow(max_activation_alpha_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_alpha_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_alphas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Alpha Value')

# Properly label each tick with its corresponding alpha value
#alpha_labels = [f'{alpha/np.pi:.3f}π' for alpha in unique_alphas]
alpha_labels = [f'{Fraction(alpha/np.pi).limit_denominator()}π' for alpha in unique_alphas]  # Give alpha labels in fractions
cbar.ax.set_yticklabels(alpha_labels)

plt.title('Map of Alpha Values Corresponding to \nMax Kernel Activation', pad = 20)
#plt.title(f'Map of Alpha Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end})', pad = 20)
plt.show()


### Lambda map ######################################################################################


# Assuming you have defined cmap and lambd_values above, check these:
unique_lambdas = np.unique(lambd_values)
print("Unique Lambdas in Map:", unique_lambdas)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
#cmap = ListedColormap(['red', 'green'])  # Adjust this list if you have more/fewer unique lambdas
cmap = ListedColormap(['blue', 'red', 'green'])  

# We must ensure boundaries enclose the actual unique lambda values used
# Let's set manual boundaries for three values, assuming lambd_values are ranged properly
#lambda_boundaries = [unique_lambdas[0] - 0.1,  # a bit less than the smallest lambda
#        (unique_lambdas[1] + unique_lambdas[0]) / 2,  # Midpoint between first and second
#        (unique_lambdas[2] + unique_lambdas[1]) / 2,  # Midpoint between second and third
#        (unique_lambdas[3] + unique_lambdas[2]) / 2,  # Midpoint between third and fourth
#        unique_lambdas[3] + 0.1]  # a bit more than the largest lambda

#lambda_boundaries = [0, 24.0, 40.0] 
#lambda_boundaries = [0, 12.0, 24.0, 40.0] 
#lambda_boundaries = [0, 12.0, 20.0]
lambda_boundaries = [0, 5.0, 7.0, 9.0] 
    
norm = BoundaryNorm(lambda_boundaries, ncolors=cmap.N)

# Plotting lambda map with correct boundaries
fig, ax = plt.subplots()
im = ax.imshow(max_activation_lambd_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_lambd_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Plotting specific region in the colormap
#im = ax.imshow(max_activation_lambd_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_lambd_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_lambdas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Lambda Value')

# Properly label each tick with its corresponding lambda value
lambda_labels = [f'{lambd:.2f}' for lambd in unique_lambdas]
cbar.ax.set_yticklabels(lambda_labels)

plt.title('Map of Lambda Values Corresponding to \nMax Kernel Activation', pad = 20)
#plt.title(f'Map of Lambda Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end})', pad = 20)
plt.show()


### Size map ######################################################################################


# Assuming you have defined cmap and size_values above, check these:
unique_sizes = np.unique(size_values)
print("Unique Sizes in Map:", unique_sizes)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
#cmap = ListedColormap(['red', 'green'])  # Adjust this list if you have more/fewer unique sizes
cmap = ListedColormap(['blue', 'red', 'green'])  

# We must ensure boundaries enclose the actual unique sizes values used
# Let's set manual boundaries for three values, assuming size_values are ranged properly
#size_boundaries = [unique_sizes[0] - 0.1,  # a bit less than the smallest size
#        (unique_sizes[1] + unique_sizes[0]) / 2,  # Midpoint between first and second
#        (unique_sizes[2] + unique_sizes[1]) / 2,  # Midpoint between second and third
#        (unique_sizes[3] + unique_sizes[2]) / 2,  # Midpoint between third and fourth
#        unique_sizes[3] + 0.1]  # a bit more than the largest size

#size_boundaries = [0, 24.0, 40.0] 
#size_boundaries = [0, 12.0, 24.0, 40.0] 
#size_boundaries = [0, 12.0, 20.0]
size_boundaries = [0, 24.0, 40.0, 72.0] 
    
norm = BoundaryNorm(size_boundaries, ncolors=cmap.N)

# Plotting size map with correct boundaries
fig, ax = plt.subplots()
im = ax.imshow(max_activation_size_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_size_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Plotting specific region in the colormap
#im = ax.imshow(max_activation_size_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_size_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_size_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_sizes, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Size Value')

# Properly label each tick with its corresponding size value
size_labels = [f'{size:.2f}' for size in unique_sizes]
cbar.ax.set_yticklabels(size_labels)

plt.title('Map of Size Values Corresponding to \nMax Kernel Activation', pad = 20)
#plt.title(f'Map of Size Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end})', pad = 20)
plt.show()



#%% Plot of Parameters Lambda, Alpha corresponding to Maximally Activated Kernel per Pixel 4 using Parameter Arrays (Ansatz 2). With Kernel Phase as separate Parameter Dimension

import matplotlib.pyplot as plt  
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


### Alpha map ######################################################################################


# Define the region to plot (example: rows 50 to 100, and columns 150 to 200)
#row_start, row_end = 450, 750
#col_start, col_end = 350, 650

# Assuming you have defined cmap and alpha_values above, check these:
unique_alphas = np.unique(alpha_values)
print("Unique alphas in Map:", unique_alphas)  # This prints the actual unique values used in the map

# Define a colormap with enough distinct colors for each unique theta value
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink'])  # Change colors if more/fewer unique thetas exist
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink', 'orange', 'brown', 'gray', 'olive', 'cyan'])  
cmap = plt.cm.coolwarm
# Reverse the colormap so it corresponds to the reversed values
#cmap_reversed = cmap.reversed()  # Reverses the list of colors starting with the last entry and ending with the first

# Define boundaries that split the range between unique alpha values
# Correctly placing these boundaries is crucial for distinct color regions
#alpha_boundaries = [unique_alphas[0] - 0.01] + list((unique_alphas[:-1] + unique_alphas[1:]) / 2) + [unique_alphas[-1] + 0.01]
alpha_boundaries = [unique_alphas[0] - 0.1,  # a bit less than the smallest alpha
                  (unique_alphas[1] + unique_alphas[0]) / 2,  # Midpoint between first and second
                  (unique_alphas[2] + unique_alphas[1]) / 2,  # Midpoint between second and third
                  (unique_alphas[3] + unique_alphas[2]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[4] + unique_alphas[3]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[5] + unique_alphas[4]) / 2,  # Midpoint between fifth and sixth
                  (unique_alphas[6] + unique_alphas[5]) / 2,  # Midpoint between first and second
                  (unique_alphas[7] + unique_alphas[6]) / 2,  # Midpoint between second and third
                  (unique_alphas[8] + unique_alphas[7]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[9] + unique_alphas[8]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[10] + unique_alphas[9]) / 2, 
                  (unique_alphas[11] + unique_alphas[10]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[12] + unique_alphas[11]) / 2,
                  unique_alphas[12] + 0.1]  # a bit more than the largest alpha

norm = BoundaryNorm(alpha_boundaries, ncolors=cmap.N)
#norm = BoundaryNorm(alpha_boundaries, ncolors=cmap_reversed.N)

# Plotting alpha map with correct boundaries
fig, ax = plt.subplots()
#im = ax.imshow(alpha_map, cmap=cmap, norm=norm, interpolation='none')
im = ax.imshow(max_activation_alpha_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_alpha_array, cmap=cmap_reversed, norm=norm, interpolation='none') 

# Plotting specific region in the colormap
#im = ax.imshow(max_activation_alpha_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_alpha_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_alphas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Alpha Value')

# Properly label each tick with its corresponding alpha value
#alpha_labels = [f'{alpha/np.pi:.3f}π' for alpha in unique_alphas]
alpha_labels = [f'{Fraction(alpha/np.pi).limit_denominator()}π' for alpha in unique_alphas]  # Give alpha labels in fractions
cbar.ax.set_yticklabels(alpha_labels)

plt.title('Map of Alpha Values Corresponding to \nMax Kernel Activation', pad = 20)
#plt.title(f'Map of Alpha Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end})', pad = 20)
plt.show()


### Lambda map ######################################################################################


# Assuming you have defined cmap and lambd_values above, check these:
unique_lambdas = np.unique(lambd_values)
print("Unique Lambdas in Map:", unique_lambdas)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
#cmap = ListedColormap(['red', 'green'])  # Adjust this list if you have more/fewer unique lambdas
cmap = ListedColormap(['blue', 'red', 'green'])  

# We must ensure boundaries enclose the actual unique lambda values used
# Let's set manual boundaries for three values, assuming lambd_values are ranged properly
#lambda_boundaries = [unique_lambdas[0] - 0.1,  # a bit less than the smallest lambda
#        (unique_lambdas[1] + unique_lambdas[0]) / 2,  # Midpoint between first and second
#        (unique_lambdas[2] + unique_lambdas[1]) / 2,  # Midpoint between second and third
#        (unique_lambdas[3] + unique_lambdas[2]) / 2,  # Midpoint between third and fourth
#        unique_lambdas[3] + 0.1]  # a bit more than the largest lambda

#lambda_boundaries = [0, 24.0, 40.0] 
#lambda_boundaries = [0, 12.0, 24.0, 40.0] 
#lambda_boundaries = [0, 12.0, 20.0]
lambda_boundaries = [0, 5.0, 7.0, 9.0] 
    
norm = BoundaryNorm(lambda_boundaries, ncolors=cmap.N)

# Plotting lambda map with correct boundaries
fig, ax = plt.subplots()
im = ax.imshow(max_activation_lambd_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_lambd_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Plotting specific region in the colormap
#im = ax.imshow(max_activation_lambd_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_lambd_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_lambdas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Lambda Value')

# Properly label each tick with its corresponding lambda value
lambda_labels = [f'{lambd:.2f}' for lambd in unique_lambdas]
cbar.ax.set_yticklabels(lambda_labels)

plt.title('Map of Lambda Values Corresponding to \nMax Kernel Activation', pad = 20)
#plt.title(f'Map of Lambda Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end})', pad = 20)
plt.show()


### Phase map ######################################################################################


# Assuming you have defined cmap and phase_values above, check these:
unique_phases = np.unique(phase_values)
print("Unique Phases in Map:", unique_phases)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
#cmap = ListedColormap(['red', 'green'])  # Adjust this list if you have more/fewer unique phases
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink', 'orange', 'brown']) 
cmap = plt.cm.coolwarm


# We must ensure boundaries enclose the actual unique sizes values used
phase_boundaries = [unique_phases[0] - 0.1,  # a bit less than the smallest phase
                  (unique_phases[1] + unique_phases[0]) / 2,  # Midpoint between first and second
                  (unique_phases[2] + unique_phases[1]) / 2,  # Midpoint between second and third
                  (unique_phases[3] + unique_phases[2]) / 2,  # Midpoint between third and fourth
                  (unique_phases[4] + unique_phases[3]) / 2,  # Midpoint between fourth and fifth
                  (unique_phases[5] + unique_phases[4]) / 2,  # Midpoint between fifth and sixth
                  (unique_phases[6] + unique_phases[5]) / 2,  
                  (unique_phases[7] + unique_phases[6]) / 2,
                  unique_phases[7] + 0.1]  

"""
# We must ensure boundaries enclose the actual unique sizes values used
phase_boundaries = [unique_phases[0] - 0.1,  # a bit less than the smallest phase
                  (unique_phases[1] + unique_phases[0]) / 2,  # Midpoint between first and second
                  (unique_phases[2] + unique_phases[1]) / 2,  # Midpoint between second and third
                  (unique_phases[3] + unique_phases[2]) / 2,  # Midpoint between third and fourth
                  (unique_phases[4] + unique_phases[3]) / 2,  # Midpoint between fourth and fifth
                  (unique_phases[5] + unique_phases[4]) / 2,  # Midpoint between fifth and sixth
                  (unique_phases[6] + unique_phases[5]) / 2,  
                  (unique_phases[7] + unique_phases[6]) / 2,
                  (unique_phases[8] + unique_phases[7]) / 2,
                  (unique_phases[9] + unique_phases[8]) / 2,
                  (unique_phases[10] + unique_phases[9]) / 2,
                  (unique_phases[11] + unique_phases[10]) / 2,
                  (unique_phases[12] + unique_phases[11]) / 2,
                  (unique_phases[13] + unique_phases[12]) / 2,
                  (unique_phases[14] + unique_phases[13]) / 2, 
                  (unique_phases[15] + unique_phases[14]) / 2,
                  unique_phases[15] + 0.1]  
"""
  
norm = BoundaryNorm(phase_boundaries, ncolors=cmap.N)

# Plotting phase map with correct boundaries
fig, ax = plt.subplots()
im = ax.imshow(max_activation_phase_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_phase_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Plotting specific region in the colormap
#im = ax.imshow(max_activation_phase_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_phase_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_phase_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_phases, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Phase Value')

# Properly label each tick with its corresponding phase value
#phase_labels = [f'{phase:.2f}' for phase in unique_phases]
phase_labels = [f'{Fraction(phase/np.pi).limit_denominator()}π' for phase in unique_phases]  # Give alpha labels in fractions
cbar.ax.set_yticklabels(phase_labels)

plt.title('Map of Phase Values Corresponding to \nMax Kernel Activation', pad = 20)
#plt.title(f'Map of Phase Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end})', pad = 20)
plt.show()



#%% Plot of Parameters Lambda, Alpha corresponding to Maximally Activated Kernel per Pixel 4 with Treshold (Ansatz 1)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


def index_to_parameters_alpha(index, lambd_values, alpha_values):
    """
    Maps a linear index to its corresponding lambda and alpha parameters.
    
    :param index: an integer index (a flattened index from the activation map)
    :param lambd_values: list of lambda values used in kernels
    :param alpha_values: list of alpha values used in kernels
    :return: Tuple (lambda, alpha)
    """
    num_lambda = len(lambd_values)  # Number of unique lambdas
    num_alpha = len(alpha_values)   # Number of unique alpha values
    
    # Calculate the lambda and alpha indices based on the linear index
    lambda_index = index // num_alpha  # Integer division to get the lambda index
    alpha_index = index % num_alpha   # Modulo to get the alpha index
    
    # Return the corresponding lambda and alpha values
    return lambd_values[lambda_index], alpha_values[alpha_index]

# Create a map for lambda and alpha
lambda_map = np.zeros_like(max_activation_kernel_indices, dtype=float)
alpha_map = np.zeros_like(max_activation_kernel_indices, dtype=float)

# Fill the maps with corresponding lambda and alpha values
for idx in range(max_activation_kernel_indices.size):
    y, x = np.unravel_index(idx, max_activation_kernel_indices.shape)
    lambda_val, alpha_val = index_to_parameters_alpha(max_activation_kernel_indices[y, x], lambd_values, alpha_values)
    alpha_map[y, x] = alpha_val
    lambda_map[y, x] = lambda_val

"""
# Plot Lambda Map
cmap_lambda = plt.get_cmap('viridis')  # Or any other perceptually uniform colormap
fig, ax = plt.subplots()
im_lambda = ax.imshow(lambda_map, cmap=cmap_lambda, interpolation='none')
cbar_lambda = fig.colorbar(im_lambda)
cbar_lambda.set_label('Lambda Value')
plt.title('Map of Lambda Values Corresponding to Max Kernel Activation')
plt.show()

# Plot Alpha Map
cmap_alpha = plt.get_cmap('viridis')
fig, ax = plt.subplots()
im_alpha = ax.imshow(alpha_map, cmap=cmap_alpha, interpolation='none')
cbar_alpha = fig.colorbar(im_alpha)
cbar_alpha.set_label('Alpha Value')
plt.title('Map of Alpha Values Corresponding to Max Kernel Activation')
plt.show()
"""


### Alpha map ######################################################################################


# Assuming you have defined cmap and alpha_values above, check these:
unique_alphas = np.unique(alpha_values)
print("Unique alphas in Map:", unique_alphas)  # This prints the actual unique values used in the map

# Define a colormap with enough distinct colors for each unique theta value
cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink'])  # Change colors if more/fewer unique thetas exist
# Reverse the colormap so it corresponds to the reversed values
cmap_reversed = cmap.reversed()  # Reverses the list of colors starting with the last entry and ending with the first

# Define boundaries that split the range between unique alpha values
# Correctly placing these boundaries is crucial for distinct color regions
#alpha_boundaries = [unique_alphas[0] - 0.01] + list((unique_alphas[:-1] + unique_alphas[1:]) / 2) + [unique_alphas[-1] + 0.01]
alpha_boundaries = [unique_alphas[0] - 0.1,  # a bit less than the smallest alpha
                  (unique_alphas[1] + unique_alphas[0]) / 2,  # Midpoint between first and second
                  (unique_alphas[2] + unique_alphas[1]) / 2,  # Midpoint between second and third
                  (unique_alphas[3] + unique_alphas[2]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[4] + unique_alphas[3]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[5] + unique_alphas[4]) / 2,  # Midpoint between fifth and sixth
                  unique_alphas[5] + 0.1]  # a bit more than the largest alpha

#norm = BoundaryNorm(alpha_boundaries, ncolors=cmap.N)
norm = BoundaryNorm(alpha_boundaries, ncolors=cmap_reversed.N)

# Plotting alpha map with correct boundaries
fig, ax = plt.subplots()
#im = ax.imshow(alpha_map, cmap=cmap, norm=norm, interpolation='none')
im = ax.imshow(alpha_map, cmap=cmap_reversed, norm=norm, interpolation='none')
cbar = fig.colorbar(im, ticks=unique_alphas, spacing='proportional')
cbar.set_label('Alpha Value')

# Properly label each tick with its corresponding alpha value
alpha_labels = [f'{alpha/np.pi:.3f}π' for alpha in unique_alphas]
cbar.ax.set_yticklabels(alpha_labels)

plt.title(f'Map of Alpha Values Corresponding to Max Kernel Activation (Threshold: {threshold:.1f})')
plt.show()


### Lambda map ######################################################################################


# Assuming you have defined cmap and lambd_values above, check these:
unique_lambdas = np.unique(lambd_values)
print("Unique Lambdas in Map:", unique_lambdas)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
cmap = ListedColormap(['red', 'green'])  # Adjust this list if you have more/fewer unique lambdas

# We must ensure boundaries enclose the actual unique lambda values used
# Let's set manual boundaries for three values, assuming lambd_values are ranged properly
#lambda_boundaries = [unique_lambdas[0] - 0.1,  # a bit less than the smallest lambda
#        (unique_lambdas[1] + unique_lambdas[0]) / 2,  # Midpoint between first and second
#        (unique_lambdas[2] + unique_lambdas[1]) / 2,  # Midpoint between second and third
#        (unique_lambdas[3] + unique_lambdas[2]) / 2,  # Midpoint between third and fourth
#        unique_lambdas[3] + 0.1]  # a bit more than the largest lambda

lambda_boundaries = [0, 24.0, 40.0] 
    
norm = BoundaryNorm(lambda_boundaries, ncolors=cmap.N)

# Plotting lambda map with correct boundaries
fig, ax = plt.subplots()
im = ax.imshow(lambda_map, cmap=cmap, norm=norm, interpolation='none')
cbar = fig.colorbar(im, ticks=unique_lambdas, spacing='proportional')
cbar.set_label('Lambda Value')

# Properly label each tick with its corresponding lambda value
lambda_labels = [f'{lambd:.2f}' for lambd in unique_lambdas]
cbar.ax.set_yticklabels(lambda_labels)

plt.title(f'Map of Lambda Values Corresponding to Max Kernel Activation (Threshold: {threshold:.1f})')
plt.show()



#%% Plot of Parameters Lambda, Alpha corresponding to Maximally Activated Kernel per Pixel 4 with Treshold using Parameter Arrays (Ansatz 2)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


### Alpha map ######################################################################################


# Define the region to plot (example: rows 50 to 100, and columns 150 to 200)
row_start, row_end = 450, 750
col_start, col_end = 350, 650

# Assuming you have defined cmap and alpha_values above, check these:
unique_alphas = np.unique(alpha_values)
print("Unique alphas in Map:", unique_alphas)  # This prints the actual unique values used in the map

# Define a colormap with enough distinct colors for each unique theta value
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink'])  # Change colors if more/fewer unique thetas exist
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink', 'orange', 'brown', 'gray', 'olive', 'cyan'])  
cmap = plt.cm.coolwarm
# Reverse the colormap so it corresponds to the reversed values
#cmap_reversed = cmap.reversed()  # Reverses the list of colors starting with the last entry and ending with the first

# Define boundaries that split the range between unique alpha values
# Correctly placing these boundaries is crucial for distinct color regions
#alpha_boundaries = [unique_alphas[0] - 0.01] + list((unique_alphas[:-1] + unique_alphas[1:]) / 2) + [unique_alphas[-1] + 0.01]
alpha_boundaries = [unique_alphas[0] - 0.1,  # a bit less than the smallest alpha
                  (unique_alphas[1] + unique_alphas[0]) / 2,  # Midpoint between first and second
                  (unique_alphas[2] + unique_alphas[1]) / 2,  # Midpoint between second and third
                  (unique_alphas[3] + unique_alphas[2]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[4] + unique_alphas[3]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[5] + unique_alphas[4]) / 2,  # Midpoint between fifth and sixth
                  (unique_alphas[6] + unique_alphas[5]) / 2,  # Midpoint between first and second
                  (unique_alphas[7] + unique_alphas[6]) / 2,  # Midpoint between second and third
                  (unique_alphas[8] + unique_alphas[7]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[9] + unique_alphas[8]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[10] + unique_alphas[9]) / 2, 
                  (unique_alphas[11] + unique_alphas[10]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[12] + unique_alphas[11]) / 2,
                  unique_alphas[12] + 0.1]  # a bit more than the largest alpha

norm = BoundaryNorm(alpha_boundaries, ncolors=cmap.N)
#norm = BoundaryNorm(alpha_boundaries, ncolors=cmap_reversed.N)

# Plotting alpha map with correct boundaries
fig, ax = plt.subplots()
#im = ax.imshow(alpha_map, cmap=cmap, norm=norm, interpolation='none')
#im = ax.imshow(max_activation_alpha_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_alpha_array, cmap=cmap_reversed, norm=norm, interpolation='none') 

# Plotting specific region in the colormap
im = ax.imshow(max_activation_alpha_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_alpha_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_alphas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Alpha Value')

# Properly label each tick with its corresponding alpha value
#alpha_labels = [f'{alpha/np.pi:.3f}π' for alpha in unique_alphas]
alpha_labels = [f'{Fraction(alpha/np.pi).limit_denominator()}π' for alpha in unique_alphas]  # Give alpha labels in fractions
cbar.ax.set_yticklabels(alpha_labels)

#plt.title(f'Map of Alpha Values Corresponding to \nMax Kernel Activation (Threshold: {threshold:.2e})', pad = 20)
plt.title(f'Map of Alpha Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end}) \n(Threshold: {threshold:.2e})', pad = 20)

#plt.gca().set_aspect('equal')  # Adjust aspect ratio
plt.show()


### Lambda map ######################################################################################


# Assuming you have defined cmap and lambd_values above, check these:
unique_lambdas = np.unique(lambd_values)
print("Unique Lambdas in Map:", unique_lambdas)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
#cmap = ListedColormap(['red', 'green'])  # Adjust this list if you have more/fewer unique lambdas
cmap = ListedColormap(['blue', 'red', 'green'])

# We must ensure boundaries enclose the actual unique lambda values used
# Let's set manual boundaries for three values, assuming lambd_values are ranged properly
#lambda_boundaries = [unique_lambdas[0] - 0.1,  # a bit less than the smallest lambda
#        (unique_lambdas[1] + unique_lambdas[0]) / 2,  # Midpoint between first and second
#        (unique_lambdas[2] + unique_lambdas[1]) / 2,  # Midpoint between second and third
#        (unique_lambdas[3] + unique_lambdas[2]) / 2,  # Midpoint between third and fourth
#        unique_lambdas[3] + 0.1]  # a bit more than the largest lambda

#lambda_boundaries = [0, 24.0, 40.0] 
#lambda_boundaries = [0, 12.0, 24.0, 40.0] 
#lambda_boundaries = [0, 12.0, 20.0]
lambda_boundaries = [0, 5.0, 7.0, 9.0] 
    
norm = BoundaryNorm(lambda_boundaries, ncolors=cmap.N)

# Plotting lambda map with correct boundaries
fig, ax = plt.subplots()
#im = ax.imshow(max_activation_lambd_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_lambd_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Plotting specific region in the colormap
im = ax.imshow(max_activation_lambd_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_lambd_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_lambdas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Lambda Value')

# Properly label each tick with its corresponding lambda value
lambda_labels = [f'{lambd:.2f}' for lambd in unique_lambdas]
cbar.ax.set_yticklabels(lambda_labels)

#plt.title(f'Map of Lambda Values Corresponding to \nMax Kernel Activation (Threshold: {threshold:.2e})', pad = 20)
plt.title(f'Map of Lambda Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end}) \n(Threshold: {threshold:.2e})', pad = 20)

#plt.gca().set_aspect('equal')  # Adjust aspect ratio
plt.show()



#%% Plot of Parameters Lambda, Alpha corresponding to Maximally Activated Kernel per Pixel 4 with Treshold using Parameter Arrays (Ansatz 2). With Kernel Size as separate Parameter Dimension

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


### Alpha map ######################################################################################


# Define the region to plot (example: rows 50 to 100, and columns 150 to 200)
#row_start, row_end = 450, 750
#col_start, col_end = 350, 650

# Assuming you have defined cmap and alpha_values above, check these:
unique_alphas = np.unique(alpha_values)
print("Unique alphas in Map:", unique_alphas)  # This prints the actual unique values used in the map

# Define a colormap with enough distinct colors for each unique theta value
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink'])  # Change colors if more/fewer unique thetas exist
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink', 'orange', 'brown', 'gray', 'olive', 'cyan'])  
cmap = plt.cm.coolwarm
# Reverse the colormap so it corresponds to the reversed values
#cmap_reversed = cmap.reversed()  # Reverses the list of colors starting with the last entry and ending with the first

# Define boundaries that split the range between unique alpha values
# Correctly placing these boundaries is crucial for distinct color regions
#alpha_boundaries = [unique_alphas[0] - 0.01] + list((unique_alphas[:-1] + unique_alphas[1:]) / 2) + [unique_alphas[-1] + 0.01]
alpha_boundaries = [unique_alphas[0] - 0.1,  # a bit less than the smallest alpha
                  (unique_alphas[1] + unique_alphas[0]) / 2,  # Midpoint between first and second
                  (unique_alphas[2] + unique_alphas[1]) / 2,  # Midpoint between second and third
                  (unique_alphas[3] + unique_alphas[2]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[4] + unique_alphas[3]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[5] + unique_alphas[4]) / 2,  # Midpoint between fifth and sixth
                  (unique_alphas[6] + unique_alphas[5]) / 2,  # Midpoint between first and second
                  (unique_alphas[7] + unique_alphas[6]) / 2,  # Midpoint between second and third
                  (unique_alphas[8] + unique_alphas[7]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[9] + unique_alphas[8]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[10] + unique_alphas[9]) / 2, 
                  (unique_alphas[11] + unique_alphas[10]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[12] + unique_alphas[11]) / 2,
                  unique_alphas[12] + 0.1]  # a bit more than the largest alpha

norm = BoundaryNorm(alpha_boundaries, ncolors=cmap.N)
#norm = BoundaryNorm(alpha_boundaries, ncolors=cmap_reversed.N)

# Plotting alpha map with correct boundaries
fig, ax = plt.subplots()
#im = ax.imshow(alpha_map, cmap=cmap, norm=norm, interpolation='none')
im = ax.imshow(max_activation_alpha_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_alpha_array, cmap=cmap_reversed, norm=norm, interpolation='none') 

# Plotting specific region in the colormap
#im = ax.imshow(max_activation_alpha_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_alpha_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_alphas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Alpha Value')

# Properly label each tick with its corresponding alpha value
#alpha_labels = [f'{alpha/np.pi:.3f}π' for alpha in unique_alphas]
alpha_labels = [f'{Fraction(alpha/np.pi).limit_denominator()}π' for alpha in unique_alphas]  # Give alpha labels in fractions
cbar.ax.set_yticklabels(alpha_labels)

plt.title(f'Map of Alpha Values Corresponding to \nMax Kernel Activation (Threshold: {threshold:.2e})', pad = 20)
#plt.title(f'Map of Alpha Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end}) \n(Threshold: {threshold:.2e})', pad = 20)

#plt.gca().set_aspect('equal')  # Adjust aspect ratio
plt.show()


### Lambda map ######################################################################################


# Assuming you have defined cmap and lambd_values above, check these:
unique_lambdas = np.unique(lambd_values)
print("Unique Lambdas in Map:", unique_lambdas)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
#cmap = ListedColormap(['red', 'green'])  # Adjust this list if you have more/fewer unique lambdas
cmap = ListedColormap(['blue', 'red', 'green'])

# We must ensure boundaries enclose the actual unique lambda values used
# Let's set manual boundaries for three values, assuming lambd_values are ranged properly
#lambda_boundaries = [unique_lambdas[0] - 0.1,  # a bit less than the smallest lambda
#        (unique_lambdas[1] + unique_lambdas[0]) / 2,  # Midpoint between first and second
#        (unique_lambdas[2] + unique_lambdas[1]) / 2,  # Midpoint between second and third
#        (unique_lambdas[3] + unique_lambdas[2]) / 2,  # Midpoint between third and fourth
#        unique_lambdas[3] + 0.1]  # a bit more than the largest lambda

#lambda_boundaries = [0, 24.0, 40.0] 
#lambda_boundaries = [0, 12.0, 24.0, 40.0] 
#lambda_boundaries = [0, 12.0, 20.0]
lambda_boundaries = [0, 5.0, 7.0, 9.0] 
    
norm = BoundaryNorm(lambda_boundaries, ncolors=cmap.N)

# Plotting lambda map with correct boundaries
fig, ax = plt.subplots()
im = ax.imshow(max_activation_lambd_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_lambd_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Plotting specific region in the colormap
#im = ax.imshow(max_activation_lambd_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_lambd_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_lambdas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Lambda Value')

# Properly label each tick with its corresponding lambda value
lambda_labels = [f'{lambd:.2f}' for lambd in unique_lambdas]
cbar.ax.set_yticklabels(lambda_labels)

plt.title(f'Map of Lambda Values Corresponding to \nMax Kernel Activation (Threshold: {threshold:.2e})', pad = 20)
#plt.title(f'Map of Lambda Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end}) \n(Threshold: {threshold:.2e})', pad = 20)

#plt.gca().set_aspect('equal')  # Adjust aspect ratio
plt.show()


### Size map ######################################################################################


# Assuming you have defined cmap and size_values above, check these:
unique_sizes = np.unique(size_values)
print("Unique Sizes in Map:", unique_sizes)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
#cmap = ListedColormap(['red', 'green'])  # Adjust this list if you have more/fewer unique sizes
cmap = ListedColormap(['blue', 'red', 'green'])  

# We must ensure boundaries enclose the actual unique sizes values used
# Let's set manual boundaries for three values, assuming size_values are ranged properly
#size_boundaries = [unique_sizes[0] - 0.1,  # a bit less than the smallest size
#        (unique_sizes[1] + unique_sizes[0]) / 2,  # Midpoint between first and second
#        (unique_sizes[2] + unique_sizes[1]) / 2,  # Midpoint between second and third
#        (unique_sizes[3] + unique_sizes[2]) / 2,  # Midpoint between third and fourth
#        unique_sizes[3] + 0.1]  # a bit more than the largest size

#size_boundaries = [0, 24.0, 40.0] 
#size_boundaries = [0, 12.0, 24.0, 40.0] 
#size_boundaries = [0, 12.0, 20.0]
size_boundaries = [0, 24.0, 40.0, 72.0] 
    
norm = BoundaryNorm(size_boundaries, ncolors=cmap.N)

# Plotting size map with correct boundaries
fig, ax = plt.subplots()
im = ax.imshow(max_activation_size_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_size_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Plotting specific region in the colormap
#im = ax.imshow(max_activation_size_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_size_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_size_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_sizes, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Size Value')

# Properly label each tick with its corresponding size value
size_labels = [f'{size:.2f}' for size in unique_sizes]
cbar.ax.set_yticklabels(size_labels)

plt.title(f'Map of Size Values Corresponding to \nMax Kernel Activation (Threshold: {threshold:.2e})', pad = 20)
#plt.title(f'Map of Size Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end}) \n(Threshold: {threshold:.2e})', pad = 20)

#plt.gca().set_aspect('equal')  # Adjust aspect ratio
plt.show()



#%% Plot of Parameters Lambda, Alpha corresponding to Maximally Activated Kernel per Pixel 4 with Treshold using Parameter Arrays (Ansatz 2). With Kernel Phase as separate Parameter Dimension

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


### Alpha map ######################################################################################


# Define the region to plot (example: rows 50 to 100, and columns 150 to 200)
#row_start, row_end = 450, 750
#col_start, col_end = 350, 650

# Assuming you have defined cmap and alpha_values above, check these:
unique_alphas = np.unique(alpha_values)
print("Unique alphas in Map:", unique_alphas)  # This prints the actual unique values used in the map

# Define a colormap with enough distinct colors for each unique theta value
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink'])  # Change colors if more/fewer unique thetas exist
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink', 'orange', 'brown', 'gray', 'olive', 'cyan'])  
cmap = plt.cm.coolwarm
# Reverse the colormap so it corresponds to the reversed values
#cmap_reversed = cmap.reversed()  # Reverses the list of colors starting with the last entry and ending with the first

# Define boundaries that split the range between unique alpha values
# Correctly placing these boundaries is crucial for distinct color regions
#alpha_boundaries = [unique_alphas[0] - 0.01] + list((unique_alphas[:-1] + unique_alphas[1:]) / 2) + [unique_alphas[-1] + 0.01]
alpha_boundaries = [unique_alphas[0] - 0.1,  # a bit less than the smallest alpha
                  (unique_alphas[1] + unique_alphas[0]) / 2,  # Midpoint between first and second
                  (unique_alphas[2] + unique_alphas[1]) / 2,  # Midpoint between second and third
                  (unique_alphas[3] + unique_alphas[2]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[4] + unique_alphas[3]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[5] + unique_alphas[4]) / 2,  # Midpoint between fifth and sixth
                  (unique_alphas[6] + unique_alphas[5]) / 2,  # Midpoint between first and second
                  (unique_alphas[7] + unique_alphas[6]) / 2,  # Midpoint between second and third
                  (unique_alphas[8] + unique_alphas[7]) / 2,  # Midpoint between third and fourth
                  (unique_alphas[9] + unique_alphas[8]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[10] + unique_alphas[9]) / 2, 
                  (unique_alphas[11] + unique_alphas[10]) / 2,  # Midpoint between fourth and fifth
                  (unique_alphas[12] + unique_alphas[11]) / 2,
                  unique_alphas[12] + 0.1]  # a bit more than the largest alpha

norm = BoundaryNorm(alpha_boundaries, ncolors=cmap.N)
#norm = BoundaryNorm(alpha_boundaries, ncolors=cmap_reversed.N)

# Plotting alpha map with correct boundaries
fig, ax = plt.subplots()
#im = ax.imshow(alpha_map, cmap=cmap, norm=norm, interpolation='none')
im = ax.imshow(max_activation_alpha_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_alpha_array, cmap=cmap_reversed, norm=norm, interpolation='none') 

# Plotting specific region in the colormap
#im = ax.imshow(max_activation_alpha_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_alpha_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_alphas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Alpha Value')

# Properly label each tick with its corresponding alpha value
#alpha_labels = [f'{alpha/np.pi:.3f}π' for alpha in unique_alphas]
alpha_labels = [f'{Fraction(alpha/np.pi).limit_denominator()}π' for alpha in unique_alphas]  # Give alpha labels in fractions
cbar.ax.set_yticklabels(alpha_labels)

plt.title(f'Map of Alpha Values Corresponding to \nMax Kernel Activation (Threshold: {threshold:.2e})', pad = 20)
#plt.title(f'Map of Alpha Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end}) \n(Threshold: {threshold:.2e})', pad = 20)

#plt.gca().set_aspect('equal')  # Adjust aspect ratio
plt.show()


### Lambda map ######################################################################################


# Assuming you have defined cmap and lambd_values above, check these:
unique_lambdas = np.unique(lambd_values)
print("Unique Lambdas in Map:", unique_lambdas)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
#cmap = ListedColormap(['red', 'green'])  # Adjust this list if you have more/fewer unique lambdas
cmap = ListedColormap(['blue', 'red', 'green'])

# We must ensure boundaries enclose the actual unique lambda values used
# Let's set manual boundaries for three values, assuming lambd_values are ranged properly
#lambda_boundaries = [unique_lambdas[0] - 0.1,  # a bit less than the smallest lambda
#        (unique_lambdas[1] + unique_lambdas[0]) / 2,  # Midpoint between first and second
#        (unique_lambdas[2] + unique_lambdas[1]) / 2,  # Midpoint between second and third
#        (unique_lambdas[3] + unique_lambdas[2]) / 2,  # Midpoint between third and fourth
#        unique_lambdas[3] + 0.1]  # a bit more than the largest lambda

#lambda_boundaries = [0, 24.0, 40.0] 
#lambda_boundaries = [0, 12.0, 24.0, 40.0] 
#lambda_boundaries = [0, 12.0, 20.0]
lambda_boundaries = [0, 5.0, 7.0, 9.0] 
    
norm = BoundaryNorm(lambda_boundaries, ncolors=cmap.N)

# Plotting lambda map with correct boundaries
fig, ax = plt.subplots()
im = ax.imshow(max_activation_lambd_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_lambd_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Plotting specific region in the colormap
#im = ax.imshow(max_activation_lambd_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_lambd_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  # Using the parameter array "max_activation_alpha_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_lambdas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Lambda Value')

# Properly label each tick with its corresponding lambda value
lambda_labels = [f'{lambd:.2f}' for lambd in unique_lambdas]
cbar.ax.set_yticklabels(lambda_labels)

plt.title(f'Map of Lambda Values Corresponding to \nMax Kernel Activation (Threshold: {threshold:.2e})', pad = 20)
#plt.title(f'Map of Lambda Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end}) \n(Threshold: {threshold:.2e})', pad = 20)

#plt.gca().set_aspect('equal')  # Adjust aspect ratio
plt.show()


### Phase map ######################################################################################


# Assuming you have defined cmap and phase_values above, check these:
unique_phases = np.unique(phase_values)
print("Unique Phases in Map:", unique_phases)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
#cmap = ListedColormap(['red', 'green'])  # Adjust this list if you have more/fewer unique phases
#cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink', 'orange', 'brown']) 
cmap = plt.cm.coolwarm

# We must ensure boundaries enclose the actual unique sizes values used
phase_boundaries = [unique_phases[0] - 0.1,  # a bit less than the smallest phase
                  (unique_phases[1] + unique_phases[0]) / 2,  # Midpoint between first and second
                  (unique_phases[2] + unique_phases[1]) / 2,  # Midpoint between second and third
                  (unique_phases[3] + unique_phases[2]) / 2,  # Midpoint between third and fourth
                  (unique_phases[4] + unique_phases[3]) / 2,  # Midpoint between fourth and fifth
                  (unique_phases[5] + unique_phases[4]) / 2,  # Midpoint between fifth and sixth
                  (unique_phases[6] + unique_phases[5]) / 2,  
                  unique_phases[7] + 0.1]  
    
norm = BoundaryNorm(phase_boundaries, ncolors=cmap.N)

# Plotting phase map with correct boundaries
fig, ax = plt.subplots()
im = ax.imshow(max_activation_phase_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_phase_array" instead of exctracting the parameter values from the kernel indices by defining and using a function

# Plotting specific region in the colormap
#im = ax.imshow(max_activation_phase_array[row_start:row_end, col_start:col_end], cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_phase_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
#im = ax.imshow(max_activation_phase_array[row_start:row_end, col_start:col_end], cmap=cmap_reversed, norm=norm, interpolation='none')  

# Tuning the colorbar
cbar = fig.colorbar(im, ticks=unique_phases, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Phase Value')

# Properly label each tick with its corresponding phase value
phase_labels = [f'{phase:.2f}' for phase in unique_phases]
cbar.ax.set_yticklabels(phase_labels)

plt.title(f'Map of Phase Values Corresponding to \nMax Kernel Activation (Threshold: {threshold:.2e})', pad = 20)
#plt.title(f'Map of Phase Values (Rows {row_start}-{row_end}, Cols {col_start}-{col_end}) \n(Threshold: {threshold:.2e})', pad = 20)

#plt.gca().set_aspect('equal')  # Adjust aspect ratio
plt.show()



#%% Vector Field mit Array (Ansatz 1) für die maximal aktivierten Indizes anstatt Liste
# SO NICHT RICHTIG FÜR ALPHA ANSTATT THETA UND PHI
# FUNKTIONIERT SO JETZT ZUSAMMEN MIT VORHERIGEN UNRAVEL BEFEHL!

# Initialize arrays to store vector field data
vector_field_x = np.zeros((zeilen, spalten))
vector_field_y = np.zeros((zeilen, spalten))

# Iterate over each pixel
for i in range(zeilen):
    for j in range(spalten):
        # Retrieve the corresponding lambda, theta, and phi values for the maximum activation kernel at this pixel
        lambd_value = max_activation_lambd_array[i, j]
        alpha_value = max_activation_alpha_array[i, j]
        
        # Calculate the components of the vector using theta and phi
        vector_x = -np.cos(alpha_value)  # Negative sign for correct starting point at theta = 0 (tracking movement from right to left) and for clockwise rotation to match clockwise rotation of Kernel
        vector_y = np.sin(alpha_value)  # No negative sign here anymore to keep clockwise rotation to match clockwise rotation of Kernel
        
        # Calculate the length of the vector based on phi
        #vector_length = 1 / np.tan(phi_value - 1/2 * np.pi)  #  Fall: Phi zw. 90 und 180 Grad und Phi enthält arctan(Frames/Pixel). Adjust the multiplier to control the length of vectors. Phi = 0, 1 -> vector_length = 0, Phi = 0.5, 1.5 -> vector_length = ∞
        #vector_length = abs(np.tan(phi_value))  # Fall: Phi zw. 0 und 180 Grad und Phi enthält arctan(Pixel/Frames)
        #vector_length = np.tan(phi_value)  # Fall: Phi zw. 0 und 90 Grad und Phi enthält arctan(Pixel/Frames)
        
        # Scale the vector components by the length
        #vector_x *= vector_length
        #vector_y *= vector_length
        
        # Store the vector components in the respective arrays
        vector_field_x[i, j] = vector_x
        vector_field_y[i, j] = vector_y

# Plot the vector field
plt.figure(figsize=(10, 8))
#plt.quiver(np.arange(spalten), np.arange(zeilen), vector_field_x, vector_field_y, pivot='mid', scale=20)
step = 40  # Adjust the step size to control density. Larger step size -> Lower Density. Lower step size -> Larger Density.
plt.quiver(np.arange(0, spalten, step), np.arange(0, zeilen, step), 
           vector_field_x[::step, ::step], vector_field_y[::step, ::step], 
           pivot='tail', scale=40)  # Larger scale -> Lower Vector Length. Lower scale -> Larger Vector Length.

plt.gca().invert_yaxis()  # Invert y-axis to match matrix indexing
plt.title('Vector Field of Maximally Activated Kernels')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()



#%% Vector Field mit Array (Ansatz 1) für die maximal aktivierten Indizes anstatt Liste mit Farbkodierung
# DAS KLAPPT SO NICHT MIT DER FARBKODIERUNG FÜR ROTATION, DASS ROT GEGEN DEN UHRZEIGERSINN, BLAU IM UHRZEIGERSINN UND WEIß KEINE ROTATION ANZEIGT UND DER BETRAG (STÄRKE DER ROTATION, SPRICH DER ALPHA WERT) IN DER HELLIGKEIT KODIERT IST
# SO NICHT RICHTIG FÜR ALPHA ANSTATT THETA UND PHI
# FUNKTIONIERT SO JETZT ZUSAMMEN MIT VORHERIGEN UNRAVEL BEFEHL!

# Initialize arrays to store vector field data
vector_field_x = np.zeros((zeilen, spalten))
vector_field_y = np.zeros((zeilen, spalten))

# Iterate over each pixel
for i in range(zeilen):
    for j in range(spalten):
        # Retrieve the corresponding lambda, theta, and phi values for the maximum activation kernel at this pixel
        lambd_value = max_activation_lambd_array[i, j]
        alpha_value = max_activation_alpha_array[i, j]
        
        # Calculate the components of the vector using theta and phi
        vector_x = -np.cos(alpha_value)  # Negative sign for correct starting point at theta = 0 (tracking movement from right to left) and for clockwise rotation to match clockwise rotation of Kernel
        vector_y = np.sin(alpha_value)  # No negative sign here anymore to keep clockwise rotation to match clockwise rotation of Kernel
        
        # Calculate the length of the vector based on phi
        #vector_length = 1 / np.tan(phi_value - 1/2 * np.pi)  #  Fall: Phi zw. 90 und 180 Grad und Phi enthält arctan(Frames/Pixel). Adjust the multiplier to control the length of vectors. Phi = 0, 1 -> vector_length = 0, Phi = 0.5, 1.5 -> vector_length = ∞
        #vector_length = abs(np.tan(phi_value))  # Fall: Phi zw. 0 und 180 Grad und Phi enthält arctan(Pixel/Frames)
        #vector_length = np.tan(phi_value)  # Fall: Phi zw. 0 und 90 Grad und Phi enthält arctan(Pixel/Frames)
        
        # Scale the vector components by the length
        #vector_x *= vector_length
        #vector_y *= vector_length
        
        # Store the vector components in the respective arrays
        vector_field_x[i, j] = vector_x
        vector_field_y[i, j] = vector_y

# Plot the vector field
plt.figure(figsize=(10, 8))
#plt.quiver(np.arange(spalten), np.arange(zeilen), vector_field_x, vector_field_y, pivot='mid', scale=20)
step = 20  # Adjust the step size to control density. Larger step size -> Lower Density. Lower step size -> Larger Density.
plt.quiver(np.arange(0, spalten, step), np.arange(0, zeilen, step),
           vector_field_x[::step, ::step], vector_field_y[::step, ::step],
           max_activation_values[::step, ::step],  # Use max_activation_values as the color values (C) for the quiver plot
           pivot='tail', scale=20, cmap='coolwarm')  # Using a colormap for color

plt.gca().invert_yaxis()  # Invert y-axis to match matrix indexing
plt.colorbar(label='Vector Length (tan(phi))')  # Add colorbar to show length scale
plt.title('Vector Field of Maximally Activated Kernels')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()



#%% Curl Field Calculation with Using np.gradient (uses finite central differences) OR scipy.ndimage.sobel (is a convolutional filter)

from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
#from scipy.ndimage import sobel

# Compute gradients (partial derivatives)
vx_dy, vx_dx = np.gradient(vector_field_x)  # Default spacing between pixels = 1 -> dx=dy=1. If spacing ≠ 1: np.gradient(array, dy_spacing, dx_spacing)
vy_dy, vy_dx = np.gradient(vector_field_y)  # np.gradient nutzt den zentralen Differenzenquotienten (central differences), nicht den Vorwärts- oder Rückwärts-Differenzenquotient
#vx_dy, vx_dx = np.gradient(mean_flow_field_x)  
#vy_dy, vy_dx = np.gradient(mean_flow_field_y)

# sobel computes edge-based derivatives (good for spatial analysis)
#vx_dy = sobel(vector_field_x, axis=0)  # ∂vx/∂y
#vy_dx = sobel(vector_field_y, axis=1)  # ∂vy/∂x

# Compute 2D scalar curl (∂vy/∂x - ∂vx/∂y)
curl = vy_dx - vx_dy  # Negative curl value -> clockwise rotation, positive curl value -> counter-clockwise rotation

# Plot the curl field
plt.figure(figsize=(10, 8))
im = plt.imshow(curl, cmap='bwr')

# Choose ticks based on actual data range
#ticks = np.linspace(np.min(curl), np.max(curl), 7)  # Or change 7 to desired number of tick levels
#cbar.set_ticks(ticks)

# Use predefined tick values as multiples of π
#tick_values = [0, -1/64, -1/32, -1/16, -1/8, -1/4, -1/3, -1/2, -2/3, 1/64, 1/32, 1/16, 1/8, 1/4, 1/3, 1/2, 2/3]
tick_values = [0, -1/64, -1/32, -1/16, -1/8, -1/4, -1/3, -1/2, -2/3, -1, 1/64, 1/32, 1/16, 1/8, 1/4, 1/3, 1/2, 2/3, 1]
tick_positions = [tick * np.pi for tick in tick_values]  # Map to actual curl value scale
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_ticks(tick_positions)

# Format as fractional multiples of π
tick_labels = [f'{Fraction(tick).limit_denominator()}π' if tick != 0 else '0' for tick in tick_values]
cbar.ax.set_yticklabels(tick_labels)
cbar.set_label('Curl (in multiples of π)')  # rotation=270, labelpad=15

plt.title('Curl of the Vector Field')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()



#%% Saving or Loading Numpy Array "summed_quadrature_convolutions"  

import numpy as np

# Save reshaped_results to a .npy file
#np.save('/Users/marius_klassen/Desktop/Masterarbeit - AE Lappe/Coding/Projekt 2/Convolution/Twisted Kernel_3D_Rotation 2D_Rotationswinkel proportional zu Frames_2D oder 3D Gabor/Windmühle mit dreieckigen Rotorblättern/Gespeicherte Arrays/summed_quadrature_convolutions.npy', summed_quadrature_convolutions)
np.save('/Users/marius_klassen/Desktop/summed_quadrature_convolutions.npy', summed_quadrature_convolutions)


# Load the saved array from the .npy file
#summed_quadrature_convolutions = np.load('/Users/marius_klassen/Desktop/Masterarbeit - AE Lappe/Coding/Projekt 1.2/Convolution/Dynamischer Kernel/Pixel Bewegung/2 4 x 4 Pixel große Quadrate/Gespeicherte Arrays/summed_quadrature_convolutions.npy')
#summed_quadrature_convolutions = np.load('/Users/marius_klassen/Desktop/summed_quadrature_convolutions.npy')



#%% Saving or Loading Numpy Array "reshaped_results"  

import numpy as np

# Save reshaped_results to a .npy file
np.save('/Users/marius_klassen/Desktop/Masterarbeit - AE Lappe/Coding/Projekt 1.2/Convolution/Dynamischer Kernel/Pixel Bewegung/2 4 x 4 Pixel große Quadrate/Gespeicherte Arrays/reshaped_results.npy', reshaped_results)

# Load the saved array from the .npy file
#reshaped_results = np.load('/Users/marius_klassen/Desktop/Masterarbeit - AE Lappe/Coding/Projekt 1.2/Convolution/Dynamischer Kernel/Pixel Bewegung/2 4 x 4 Pixel große Quadrate/Gespeicherte Arrays/reshaped_results.npy')



#%% Vergleich zwischen vorherigen Argmax Befehl bei reshaped_results Ansatz und Argmax Befehl direkt bei summed_quadrature_convolutions Ansatz
### Stand 19.11: Kommt raus, dass es nicht dasselbe ist, weil die Matrizen unterschiedliche Shapes haben

# Choose the desired shape as the original convolution result shape
desired_shape = (zeilen, spalten) 

# Convert the "summed_quadrature_convolution" directly into max activation parameter values without reshaping
max_activation_lambd_indices_array_ohne_reshape = np.argmax(summed_quadrature_convolutions, axis=0)
#max_activation_lambd_indices_array_ohne_reshape = np.argmax(max_activation_lambd_indices_array_ohne_reshape, axis=0)
max_activation_alpha_indices_array_ohne_reshape = np.argmax(summed_quadrature_convolutions, axis=1)
#max_activation_alpha_indices_array_ohne_reshape = np.argmax(max_activation_alpha_indices_array_ohne_reshape, axis=0)

# Remove singleton dimension from the result array
#max_activation_lambd_indices_array_ohne_reshape = np.squeeze(max_activation_lambd_indices_array_ohne_reshape, axis=0)
#max_activation_alpha_indices_array_ohne_reshape = np.squeeze(summed_quadrature_convolutions, axis=0)

# Retrieve the lambd, alpha values corresponding to the maximally activated kernel index (or here parameter index) in each pixel
max_activation_lambd_array_ohne_reshape = np.array(lambd_values)[max_activation_lambd_indices_array_ohne_reshape]
max_activation_alpha_array_ohne_reshape = np.array(alpha_values)[max_activation_alpha_indices_array_ohne_reshape]

# Compare all entries of the two Ansätze for all Parameters  

are_identical_lambd = np.array_equal(max_activation_lambd_array, max_activation_lambd_array_ohne_reshape)

print("Are the lambd arrays equal?", are_identical_lambd)

are_identical_alpha = np.array_equal(max_activation_alpha_array, max_activation_alpha_array_ohne_reshape)

print("Are the alpha arrays equal?", are_identical_alpha)



# Compare all entries of the two Ansätze for all parameters with error tolerance

#are_identical_lambd = np.allclose(max_activation_lambd_array, max_activation_lambd_array_ohne_reshape, atol=1e-7)

#print("Are the lambd arrays close enough?", are_identical_lambd)

#are_identical_alpha = np.allclose(max_activation_alpha_array, max_activation_alpha_array_ohne_reshape, atol=1e-7)

#print("Are the alpha arrays close enough?", are_identical_alpha)



#%% Vergleich zwischen Convolutions Ergebnissen "summed_quadrature_convolutions" von Actionpoint 59 Test 2.10 mit Test 2.11
### Stand 07.03.2025: summed_quadrature_convolutions von Actionpoint 59 Test 2.10 und Test 2.11 sind identisch ('equal') ###

import numpy as np


# Load the saved array from the .npy file
summed_quadrature_convolutions_test_2_10 = np.load('/Users/marius_klassen/Desktop/summed_quadrature_convolutions.npy')

summed_quadrature_convolutions_test_2_11 = np.load('/Users/marius_klassen/Desktop/summed_quadrature_convolutions.npy')


# Compare all entries of the two arrays for all parameters  

are_identical_summed_quadrature_convolution = np.array_equal(summed_quadrature_convolutions_test_2_10, summed_quadrature_convolutions_test_2_11)

print("Are the summed_quadrature_convolutions arrays equal?", are_identical_summed_quadrature_convolution)





