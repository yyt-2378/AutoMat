import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

# Define paths to your image folders
original_images_path = 'F:\\STEM_Data_ori\\SRDATA\\validation\\LR_original'
denoised_images_path = 'F:\\test_preprocess_model_different_hyparam\\version_8\\VAE_reconstructed'
noise_free_images_path = 'F:\\STEM_Data_ori\\SRDATA\\validation\\HR'
output_img_show = 'F:\\compare_noise'
os.makedirs(output_img_show, exist_ok=True)


# Function to read and resize a single image
def load_and_resize_image(image_path, new_size=(64, 64)):
    return transform.resize(io.imread(image_path), new_size, anti_aliasing=True)


# Loop through each image in the original images folder
for img_i in os.listdir(original_images_path):
    if img_i.endswith('.png'):
        original_image_path = os.path.join(original_images_path, img_i)
        denoised_image_path = os.path.join(denoised_images_path, 'VAE_reconstructed_' + img_i)
        noise_free_image_path = os.path.join(noise_free_images_path, img_i)

        # Load and resize images
        original_image = load_and_resize_image(original_image_path)
        denoised_image = load_and_resize_image(denoised_image_path)
        noise_free_image = load_and_resize_image(noise_free_image_path)

        # Calculate noise distributions
        ideal_noise = original_image - noise_free_image
        divae_noise = original_image - denoised_image

        # Visualizing the images
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(noise_free_image, cmap='gray')
        axes[0, 1].set_title('Noise-Free Image')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(ideal_noise, cmap='gray')
        axes[1, 0].set_title('Ideal Noise')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(divae_noise, cmap='gray')
        axes[1, 1].set_title('DIVAE Noise')
        axes[1, 1].axis('off')

        plt.suptitle('Noise Comparison')
        plt.savefig(os.path.join(output_img_show, img_i))
        plt.close()
