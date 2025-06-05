#%% Stimuklus: Rotierende Perlin Noise Disc mit Rotate Function von scipy.ndimage mit Sharp Edge

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
col_start, col_end = 350, 650
row_start, row_end = 450, 750
matrix = matrix[row_start:row_end, col_start:col_end, :]


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



#%% Sine Kernels. Erst Theta, dann Phi Rotation. Mit variabler Kernelgröße. Mit symmetrischem Binning 

def generate_sine_gabor_kernel_3d(size_x, size_y, size_z, sigma, theta, phi, lambd):
    sine_kernel = np.zeros((size_y, size_x, size_z))
    
    center_x = size_x // 2
    center_y = size_y // 2
    center_z = size_z // 2

    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                
                # Shifting the whole Array / Plot from the (0,0,0) coordinates into the center with (32,32,32) coordinates
                x_prime = x - center_x + 0.5
                y_prime = y - center_y + 0.5
                z_prime = z - center_z + 0.5
                
                # Rotate in the x-y plane (around the z-axis) im Uhrzeigersinn
                x_rotated_xy = x_prime  * np.cos(theta) + y_prime  * np.sin(theta)
                y_rotated_xy = -x_prime  * np.sin(theta) + y_prime  * np.cos(theta)
                z_rotated_xy = z_prime
                
                # Rotate in the x-z plane (around the y-axis) im Uhrzeigersinn
                x_rotated = x_rotated_xy * np.cos(phi) + z_rotated_xy * np.sin(phi)
                y_rotated = y_rotated_xy
                z_rotated = -x_rotated_xy * np.sin(phi) + z_rotated_xy * np.cos(phi)
                
                gaussian = 1 / ((np.sqrt(2 * np.pi) * sigma)**3) * np.exp(-(x_rotated**2 + y_rotated**2 + z_rotated**2) / (2 * sigma**2))
                #gaussian = np.exp(-(x_prime**2 + y_prime**2 + z_prime**2) / (2 * sigma**2))  # Ohne Rotation der Gauß-Glocke. Das ist äquivalent zum Fall mit Rotation der Gauß-Glocke (Zeile drüber), da die Gauß-Glocke kugelsymmetrisch, also rotationssymmetrisch um beide Raumwinkel ist, denn sie hängt nur vom Radius ab
                sinusoidal = np.sin(2 * np.pi * x_rotated / lambd)
                gabor_value = gaussian * sinusoidal
                sine_kernel[y, x, z] = gabor_value
                
                # Man könnte auch räuml. Filterkern D_s und zeitl. Filterkern D_t nehmen und multiplizieren, denn sie sind raumzeitl. separable
                # D_s = np.exp(-(x_rotated**2 + y_rotated**2) / (2 * sigma**2)) * np.sin(2 * np.pi * x_rotated / lambd)
                # D_t = np.exp(-(z_rotated**2) / (2 * sigma**2))
                # D_ges = D_s * D_t = gabor_value
                # sine_kernel[y, x, z] = D_ges = gabor_value
                
    # Ensure the overall sum of elements is 0
    mean_value = np.mean(sine_kernel)
    sine_kernel -= mean_value
    
    # Normalize the sum of absolute values to 1
    abs_sum = np.sum(np.abs(sine_kernel))
    if abs_sum != 0:
        sine_kernel /= abs_sum
    
    #sine_kernel *= 1e2  # Um Kernel anderen Skalierungsfaktor zu geben 
    
    return sine_kernel



#%% Cosine Kernels. Erst Theta, dann Phi Rotation. Mit variabler Kernelgröße. Mit symmetrischem Binning

def generate_cosine_gabor_kernel_3d(size_x, size_y, size_z, sigma, theta, phi, lambd):
    cosine_kernel = np.zeros((size_x, size_y, size_z))
    
    center_x = size_x // 2
    center_y = size_y // 2
    center_z = size_z // 2

    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                
                # Shifting the whole Array / Plot from the (0,0,0) coordinates into the center with (32,32,32) coordinates
                x_prime = x - center_x + 0.5
                y_prime = y - center_y + 0.5
                z_prime = z - center_z + 0.5
                
                # Rotate in the x-y plane (around the z-axis) im Uhrzeigersinn
                x_rotated_xy = x_prime  * np.cos(theta) + y_prime  * np.sin(theta)
                y_rotated_xy = -x_prime  * np.sin(theta) + y_prime  * np.cos(theta)
                z_rotated_xy = z_prime
                
                # Rotate in the x-z plane (around the y-axis) im Uhrzeigersinn
                x_rotated = x_rotated_xy * np.cos(phi) + z_rotated_xy * np.sin(phi)
                y_rotated = y_rotated_xy
                z_rotated = -x_rotated_xy * np.sin(phi) + z_rotated_xy * np.cos(phi)
                
                gaussian = 1 / ((np.sqrt(2 * np.pi) * sigma)**3) * np.exp(-(x_rotated**2 + y_rotated**2 + z_rotated**2) / (2 * sigma**2))
                #gaussian = np.exp(-(x_prime**2 + y_prime**2 + z_prime**2) / (2 * sigma**2))  # Ohne Rotation der Gauß-Glocke. Das ist äquivalent zum Fall mit Rotation der Gauß-Glocke (Zeile drüber), da die Gauß-Glocke kugelsymmetrisch, also rotationssymmetrisch um beide Raumwinkel ist, denn sie hängt nur vom Radius ab
                cosinusoidal = np.cos(2 * np.pi * x_rotated / lambd)
                gabor_value = gaussian * cosinusoidal
                cosine_kernel[y, x, z] = gabor_value

    # Ensure the overall sum of elements is 0
    mean_value = np.mean(cosine_kernel)
    cosine_kernel -= mean_value
    
    # Normalize the sum of absolute values to 1
    abs_sum = np.sum(np.abs(cosine_kernel))
    if abs_sum != 0:
        cosine_kernel /= abs_sum 
    
    #cosine_kernel *= 1e2  # Um Kernel anderen Skalierungsfaktor zu geben 
    
    return cosine_kernel



#%% Parameters for the Gabor kernel

size = 16  # Kernel size in px x px x frames 
size_x = 16  # Kernel size in px in x-direction
size_y = 16  # Kernel size in px in y-direction
size_z = 16  # Kernel size in px in z-direction (Frames i.e. Time)
#size = 17  # Kernel size in px x px x frames 
#size_x = 17  # Kernel size in px in x-direction
#size_y = 17  # Kernel size in px in y-direction
#size_z = 17  # Kernel size in px in z-direction (Frames i.e. Time)

# Define possible cubic sizes
#size_values = [16, 32, 64]  # For 16 x 16 x 16, 32 x 32 x 32 and 64 x 64 x 64 Kernels
#size_values = [16] 

#sigma = 5.0  # Standard Deviation of the Gauß, gives the width of the Gauß
#sigma = 2.5  # Test vom Stand 30.10
sigma = 3.106  # Test vom Stand 25.12 (Actionpoint 38). For 16 x 16 x 16 Kernels
#sigma = 3.3  # Test vom Stand 31.12 (Actionpoint 38). For 17 x 17 x 17 Kernels

# Define sigma values for different kernel sizes
#sigma_values = {16: 3.106, 32: 6.211, 64: 12.422}

# Choose the lambda (lambd) values 
#lambd_values = [8.0, 16.0, 32.0]  # Spatial frequency or rather wavelength
lambd_values = [4.0, 6.4, 8.0]

# Choose the theta values in multiples of pi 
#theta_values = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # Angle of rotation in x-y plane
#theta_values = [0.0, 0.25, 0.5, 0.75]
theta_values = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875]

# Choose the actual theta values not in multiples of pi
theta_values = [value * np.pi for value in theta_values] 

# Choose the phi values in multiples of pi
# Im Argument des Arctan steht im Folgenden Frames/Pixel -> arctan(Frames/Pixel)
#phi_values = [1, 0.5+np.arctan(1/1)/np.pi, 0.5+np.arctan(1/np.sqrt(2))/np.pi, 0.5+np.arctan(1/3)/np.pi, 0.5+np.arctan(1/0.75)/np.pi]  # Angle of rotation in x-z plane
#phi_values = [0, 0.5+np.arctan(1/1)/np.pi, 0.5+np.arctan(1/np.sqrt(2))/np.pi] 
#phi_values = [1, 0.5+np.arctan(1/1)/np.pi, 0.5+np.arctan(1/np.sqrt(2))/np.pi] 
# Im Argument des Arctan steht im Folgenden Pixel/Frames -> arctan(Pixel/Frames). Phi muss zwischen 0 und π/2 (90°) liegen, wenn ich Theta zwischen 0 und 360° habe (Stand 21.02.25)
#phi_values = [0, 0+np.arctan(1/1)/np.pi, 0.5+np.arctan(1/1)/np.pi]  # Test, um zu checken, ob Phi eine Richtung präferiert, obwohl dies eig in Theta (360 Grad) einkodiert sein sollte.
#phi_values = [0, np.arctan(1/1)/np.pi, np.arctan(np.sqrt(2)/1)/np.pi]
#phi_values = [0, np.arctan(1/1)/np.pi, np.arctan(np.sqrt(2)/1)/np.pi]
phi_values = [0, np.arctan(1/1)/np.pi, np.arctan(np.sqrt(2)/1)/np.pi, np.arctan(20 * 1/16 * np.pi)/np.pi, np.arctan(40 * 1/16 * np.pi)/np.pi, np.arctan(60 * 1/16 * np.pi)/np.pi, np.arctan(80 * 1/16 * np.pi)/np.pi, np.arctan(100 * 1/16 * np.pi)/np.pi]  # AP 66 (28.05.2025)

# Choose the actual phi values not in multiples of pi
phi_values = [value * np.pi for value in phi_values] 

print(f"Length of Lambda Values: {len(lambd_values)}")    # Print the length of lambd_values  
print(f"Length of Theta Values: {len(theta_values)}")    # Print the length of theta_values    
print(f"Length of Phi Values: {len(phi_values)}")    # Print the length of phi_values  
#print(f"Length of Sizes: {len(size_values)}")    # Print the length of alpha_values  
#print(f"Length of Sigma Values: {len(sigma_values)}")    # Print the length of alpha_values 



#%% Print Parameters for the Gabor Kernel

import numpy as np

# Function to format values in multiples of π
def format_pi_multiples(values):
    return [f"{round(value / np.pi, 3)}π" for value in values]

# Function to print parameters
def print_parameters():
    print("===== Parameters for the Gabor Kernel =====")
    #print(f"Kernel size (size): {size}")
    #print(f"Kernel size in x-direction (size_x): {size_x}")
    #print(f"Kernel size in y-direction (size_y): {size_y}")
    #print(f"Kernel size in z-direction (size_z): {size_z}")
    #print(f"Sigma (sigma): {sigma:.3f}")
    #print(f"Sigma values (sigma_values): {sigma_values}")
    #print(f"Number of Sigma values: {len(sigma_values)}")
    print(f"Lambda values (lambd_values): {lambd_values}")
    print(f"Number of Lambda values: {len(lambd_values)}")
    print(f"Theta values (theta_values in multiples of π): {format_pi_multiples(theta_values)}")
    print(f"Number of Theta values: {len(theta_values)}")
    print(f"Phi values (phi_values in multiples of π): {format_pi_multiples(phi_values)}")
    print(f"Number of Phi values: {len(phi_values)}")
    #print(f"Kernel sizes (size_values): {size_values}")
    #print(f"Number of Kernel sizes: {len(size_values)}")
    print("===========================================")

# Call the function
print_parameters()


           
#%% Convolutions with Sine and Cosine Kernels on Stimulus using Correlation instead of Convolution from Scipy.Signal

from scipy.signal import correlate


# New dimensions of the convolutions' arrays
# 64 x 64 x 64 Kernels
#zeilen = 1017    # = n_rows of Input Array - k_rows of Kernel Array + 1 
#spalten = 1857   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 16 und 16 x 16 x 64 Kernels
#zeilen = 1065    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1905   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 17 x 17 x 17 Kernels
#zeilen = 1064    # = n_rows of Input Array - k_rows of Kernel Array + 1
#spalten = 1904   # = n_columns of Input Array - k_columns of Kernel Array + 1
# 16 x 16 x 16 Kernels on 300 x 300 Input Array (not Full HD like all the above)
zeilen = 285    # = n_rows of Input Array - k_rows of Kernel Array + 1
spalten = 285   # = n_columns of Input Array - k_columns of Kernel Array + 1

# Initialize arrays to store convolution results for sine and cosine kernels
sine_convolutions = np.zeros((len(lambd_values), len(theta_values), len(phi_values), zeilen, spalten))
cosine_convolutions = np.zeros((len(lambd_values), len(theta_values), len(phi_values), zeilen, spalten))

# Iterate over each kernel and perform convolution for both sine and cosine kernels
for k, lambd in enumerate(lambd_values):
    for j, theta in enumerate(theta_values):
        for i, phi in enumerate(phi_values):
            #sine_kernel = generate_sine_gabor_kernel_3d(size, sigma, theta, phi, lambd)
            #cosine_kernel = generate_cosine_gabor_kernel_3d(size, sigma, theta, phi, lambd)
            sine_kernel = generate_sine_gabor_kernel_3d(size_x, size_y, size_z, sigma, theta, phi, lambd)  
            cosine_kernel = generate_cosine_gabor_kernel_3d(size_x, size_y, size_z, sigma, theta, phi, lambd) 
            sine_convolution = correlate(matrix, sine_kernel, mode='valid', method='fft')  # Does not flip the kernel in all axes before convolution calculation
            cosine_convolution = correlate(matrix, cosine_kernel, mode='valid', method='fft')  # Does not flip the kernel in all axes before convolution calculation
            
            # Remove singleton dimension from the result array
            sine_convolution = np.squeeze(sine_convolution)
            cosine_convolution = np.squeeze(cosine_convolution)
            
            sine_convolutions[k, j, i] = sine_convolution  # Assign the result. sine_convolutions[k, j, i] = sine_convolutions[k, j, i, :, :]
            cosine_convolutions[k, j, i] = cosine_convolution  # Assign the result. cosine_convolutions[k, j, i] = cosine_convolutions[k, j, i, :, :]



#%% Quadrature and Sum of Quadrature Sine and Cosine Convolutions

# Take the quadrature of both sine and cosine convolution results
quadrature_sine_convolutions = np.square(sine_convolutions)
quadrature_cosine_convolutions = np.square(cosine_convolutions)

# Sum up the quadrature convolution results to get the final convolution result
summed_quadrature_convolutions = quadrature_sine_convolutions + quadrature_cosine_convolutions




#%% Reshaping and Finding Maximum Activation Kernel Index per Pixel mit Unravel-Befehl (Ansatz 1)

# Reshape convolution_results to make it easier to work with
reshaped_results = summed_quadrature_convolutions.reshape(-1, zeilen, spalten)

# Find the index of the maximum convolution result across all kernels for each pixel
max_activation_kernel_indices = np.argmax(reshaped_results, axis=0)

# Convert flat indices to (lambd, theta, phi) indices
max_activation_parameter_indices = np.unravel_index(max_activation_kernel_indices, (len(lambd_values), len(theta_values), len(phi_values)))  # Creates a tuple of 3 Matrices. One Matrix for each Parameter (lambd, theta, phi) with the corresponding Parameter Index in each Pixel

# Retrieve the lambd, theta, phi values corresponding to the maximally activated kernel index (or here parameter index) in each pixel
max_activation_lambd_array = np.array(lambd_values)[max_activation_parameter_indices[0]]
max_activation_theta_array = np.array(theta_values)[max_activation_parameter_indices[1]]
max_activation_phi_array = np.array(phi_values)[max_activation_parameter_indices[2]]



#%% Plot of Maximally Activated Kernel per Pixel 

# Display the array indicating the kernel index with the highest convolution value at each pixel
#plt.imshow(max_activation_kernel_indices[:, :], cmap='viridis', vmin=0, vmax=143, interpolation='none')  # Assuming there are 144 kernels
#plt.colorbar(ticks=range(144))  # Set ticks for each kernel index
#plt.imshow(max_activation_kernel_indices[:, :], cmap='viridis', vmin=0, vmax=431, interpolation='none')  # Assuming there are 432 kernels
#plt.colorbar(ticks=range(432))  # Set ticks for each kernel index
plt.imshow(max_activation_kernel_indices[:, :], cmap='viridis', vmin=0, vmax=383, interpolation='none')  # Assuming there are 432 kernels
plt.colorbar(ticks=range(384))  # Set ticks for each kernel index
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



#%% Plot of Parameters Lambda, Theta, Phi corresponding to Maximally Activated Kernel per Pixel 4 using Parameter Arrays (Ansatz 2)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


### Theta map ######################################################################################


# Retrieve unique theta values from the theta_map or rather from theta_values
#unique_thetas = np.unique(theta_map)
unique_thetas = np.unique(theta_values)
print("Unique Thetas in Map (Multiples of Pi):", unique_thetas/np.pi)  # Ensuring we know what unique values are present

# Define a colormap with enough distinct colors for each unique theta value
cmap = ListedColormap(['red', 'blue', 'yellow', 'green', 'purple', 'pink', 'orange', 'brown', 'gray', 'olive', 'cyan', 'black', 'white', 'lime', 'gold', 'navy'])  # Change colors if more/fewer unique thetas exist

# Define boundaries that split the range between unique theta values
# Correctly placing these boundaries is crucial for distinct color regions
#theta_boundaries = [unique_thetas[0] - 0.01] + list((unique_thetas[:-1] + unique_thetas[1:]) / 2) + [unique_thetas[-1] + 0.01]
theta_boundaries = [unique_thetas[0] - 0.1,  # a bit less than the smallest theta
                  (unique_thetas[1] + unique_thetas[0]) / 2,  # Midpoint between first and second
                  (unique_thetas[2] + unique_thetas[1]) / 2,  # Midpoint between second and third
                  (unique_thetas[3] + unique_thetas[2]) / 2,  # Midpoint between third and fourth
                  (unique_thetas[4] + unique_thetas[3]) / 2,  # Midpoint between fourth and fifth
                  (unique_thetas[5] + unique_thetas[4]) / 2,  # Midpoint between fifth and sixth
                  (unique_thetas[6] + unique_thetas[5]) / 2,  # Midpoint between sixth and seventh
                  (unique_thetas[7] + unique_thetas[6]) / 2,  # Midpoint between seventh and eighth
                  (unique_thetas[8] + unique_thetas[7]) / 2,
                  (unique_thetas[9] + unique_thetas[8]) / 2,
                  (unique_thetas[10] + unique_thetas[9]) / 2,
                  (unique_thetas[11] + unique_thetas[10]) / 2,
                  (unique_thetas[12] + unique_thetas[11]) / 2,
                  (unique_thetas[13] + unique_thetas[12]) / 2,
                  (unique_thetas[14] + unique_thetas[13]) / 2,
                  (unique_thetas[15] + unique_thetas[14]) / 2,
                  unique_thetas[15] + 0.1]  # a bit more than the largest theta

norm = BoundaryNorm(theta_boundaries, ncolors=cmap.N)

# Visualization
fig, ax = plt.subplots()
im = ax.imshow(max_activation_theta_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_theta_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
cbar = fig.colorbar(im, ticks=unique_thetas, spacing='proportional')
cbar.set_label('Theta Value')

# Label each tick with its corresponding theta value
theta_labels = [f'{theta/np.pi:.2f}π' for theta in unique_thetas]
cbar.ax.set_yticklabels(theta_labels)

plt.title('Map of Theta Values Corresponding to \nMax Kernel Activation', pad = 20)
plt.show()


### Phi map ######################################################################################


# Assuming you have defined cmap and phi_values above, check these:
#unique_phis = np.unique(phi_map)
unique_phis = np.unique(phi_values)
print("Unique Phis in Map (Multiples of Pi):", unique_phis/np.pi)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
#cmap = ListedColormap(['red', 'green', 'blue'])  # Adjust this list if you have more/fewer unique phis
cmap = ListedColormap(['red', 'green', 'blue', 'yellow', 'purple', 'pink', 'orange', 'brown'])

"""
# We must ensure boundaries enclose the actual unique phi values used
# Let's set manual boundaries for three values, assuming phi_values are ranged properly
phi_boundaries = [unique_phis[0] - 0.01,  # a bit less than the smallest phi
                  (unique_phis[1] + unique_phis[0]) / 2,  # Midpoint between first and second
                  (unique_phis[2] + unique_phis[1]) / 2,  # Midpoint between second and third
                  unique_phis[2] + 0.01]  # a bit more than the largest phi
"""

# We must ensure boundaries enclose the actual unique phi values used
# Let's set manual boundaries for three values, assuming phi_values are ranged properly
phi_boundaries = [unique_phis[0] - 0.01,  # a bit less than the smallest phi
                  (unique_phis[1] + unique_phis[0]) / 2,  # Midpoint between first and second
                  (unique_phis[2] + unique_phis[1]) / 2,  # Midpoint between second and third
                  (unique_phis[3] + unique_phis[2]) / 2, 
                  (unique_phis[4] + unique_phis[3]) / 2, 
                  (unique_phis[5] + unique_phis[4]) / 2, 
                  (unique_phis[6] + unique_phis[5]) / 2, 
                  (unique_phis[7] + unique_phis[6]) / 2, 
                  unique_phis[7] + 0.01]  # a bit more than the largest phi

norm = BoundaryNorm(phi_boundaries, ncolors=cmap.N)

# Plotting phi map with correct boundaries
fig, ax = plt.subplots()
im = ax.imshow(max_activation_phi_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_phi_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
cbar = fig.colorbar(im, ticks=unique_phis, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Phi Value')

# Properly label each tick with its corresponding phi value
phi_labels = [f'{phi/np.pi:.3f}π' for phi in unique_phis]
cbar.ax.set_yticklabels(phi_labels)

plt.title('Map of Phi Values Corresponding to \nMax Kernel Activation', pad = 20)
plt.show()


### Lambda map ######################################################################################


# Assuming you have defined cmap and lambd_values above, check these:
#unique_lambdas = np.unique(lambda_map)
unique_lambdas = np.unique(lambd_values)
print("Unique Lambdas in Map:", unique_lambdas)  # This prints the actual unique values used in the map

# Define a colormap with enough colors
cmap = ListedColormap(['red', 'green', 'blue'])  # Adjust this list if you have more/fewer unique lambdas

# We must ensure boundaries enclose the actual unique lambda values used
# Let's set manual boundaries for three values, assuming lambd_values are ranged properly
#lambda_boundaries = [unique_lambdas[0] - 0.1,  # a bit less than the smallest lambda
#        (unique_lambdas[1] + unique_lambdas[0]) / 2,  # Midpoint between first and second
#        (unique_lambdas[2] + unique_lambdas[1]) / 2,  # Midpoint between second and third
#        (unique_lambdas[3] + unique_lambdas[2]) / 2,  # Midpoint between third and fourth
#        unique_lambdas[3] + 0.1]  # a bit more than the largest lambda

lambda_boundaries = [0, 5.0, 7.0, 9.0] 
    
norm = BoundaryNorm(lambda_boundaries, ncolors=cmap.N)

# Plotting lambda map with correct boundaries
fig, ax = plt.subplots()
im = ax.imshow(max_activation_lambd_array, cmap=cmap, norm=norm, interpolation='none')  # Using the parameter array "max_activation_lambd_array" instead of exctracting the parameter values from the kernel indices by defining and using a function
cbar = fig.colorbar(im, ticks=unique_lambdas, spacing='uniform')  # Also spacing='proportional' possible to distribute colors according to the proportion of the data range they represent
cbar.set_label('Lambda Value')

# Properly label each tick with its corresponding lambda value
lambda_labels = [f'{lambd:.2f}' for lambd in unique_lambdas]
cbar.ax.set_yticklabels(lambda_labels)

plt.title('Map of Lambda Values Corresponding to \nMax Kernel Activation', pad = 20)
plt.show()



#%% Vector Field mit Array (Ansatz 1) für die maximal aktivierten Indizes anstatt Liste
# FUNKTIONIERT SO JETZT ZUSAMMEN MIT VORHERIGEN UNRAVEL BEFEHL!

# Initialize arrays to store vector field data
#zeilen = target_rows
#spalten = target_cols
vector_field_x = np.zeros((zeilen, spalten))
vector_field_y = np.zeros((zeilen, spalten))

# Iterate over each pixel
for i in range(zeilen):
    for j in range(spalten):
        # Retrieve the corresponding lambda, theta, and phi values for the maximum activation kernel at this pixel
        lambd_value = max_activation_lambd_array[i, j]
        theta_value = max_activation_theta_array[i, j]
        phi_value = max_activation_phi_array[i, j]
        
        # Calculate the components of the vector using theta and phi
        vector_x = -np.cos(theta_value)  # Negative sign for correct starting point at theta = 0 (tracking movement from right to left) and for clockwise rotation to match clockwise rotation of Kernel
        vector_y = np.sin(theta_value)  # No negative sign here anymore to keep clockwise rotation to match clockwise rotation of Kernel
        
        # Calculate the length of the vector based on phi
        #vector_length = 1 / np.tan(phi_value - 1/2 * np.pi)  #  Fall: Phi zw. 90 und 180 Grad und Phi enthält arctan(Frames/Pixel). Adjust the multiplier to control the length of vectors. Phi = 0, 1 -> vector_length = 0, Phi = 0.5, 1.5 -> vector_length = ∞
        #vector_length = abs(np.tan(phi_value))  # Fall: Phi zw. 0 und 180 Grad und Phi enthält arctan(Pixel/Frames)
        vector_length = np.tan(phi_value)  # Fall: Phi zw. 0 und 90 Grad und Phi enthält arctan(Pixel/Frames)
        
        # Scale the vector components by the length
        vector_x *= vector_length
        vector_y *= vector_length
        
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
plt.gca().set_aspect('equal')  # Adjust aspect ratio
plt.show()
