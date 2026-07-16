import torch
from torchvision import transforms
from PIL import Image
import yaml
import os

# preprocess_model
from preprocess_model.image_preprocess_model import DCVAESR

# SR model
from preprocess_model.configs.option import args


def calculate_error(output_label, hr_label):
    # Calculate the structural error based on output_label and hr_label (e.g., using MSE)
    error = torch.mean(abs(output_label - hr_label))
    return error.item()


def process_images(model_weights_path, image_folder, label_folder, image_output_folder):
    # Load the VAE and SR model configuration
    sr_model_args = args
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    vae_model_args = config['model_params']

    # Define and load the model
    preprocess_model = DCVAESR(sr_model_args, vae_model_args)
    checkpoint = torch.load(model_weights_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace("model.", "")  # Remove the "model." prefix
        new_state_dict[new_key] = state_dict[key]
    preprocess_model.load_state_dict(new_state_dict)

    preprocess_model.eval()

    # Image preprocessing transformation
    img_tran = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    label_tran = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Create output folders
    VAE_path = os.path.join(image_output_folder, "VAE_reconstructed")
    SR_path = os.path.join(image_output_folder, "SR_reconstructed")
    os.makedirs(VAE_path, exist_ok=True)
    os.makedirs(SR_path, exist_ok=True)

    # Initialize variables to track the minimum error and corresponding weights
    error_list = []

    # Process images in the specified folder
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file)
        image = Image.open(image_path)
        hr_label = Image.open(label_path)
        image_tensor = img_tran(image).unsqueeze(0)
        # For demonstration purposes, let's assume the HR label is a tensor of zeros
        hr_label_tensor = label_tran(hr_label).unsqueeze(0)

        with torch.no_grad():
            _ = torch.zeros_like(image_tensor)
            results = preprocess_model.forward(image_tensor, _, hr_label_tensor)
            vae_output, sr_output, output_label = results[1], results[2], results[3]

        # Calculate the structural error for this image
        error = calculate_error(sr_output, hr_label_tensor)
        error_list.append(error)

        # Save VAE reconstructed image
        VAE_reconstructed_image = vae_output.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze()
        VAE_reconstructed_image = (VAE_reconstructed_image - VAE_reconstructed_image.min()) / (VAE_reconstructed_image.max() - VAE_reconstructed_image.min())
        VAE_save_path = os.path.join(VAE_path, f"VAE_reconstructed_{image_file}")
        Image.fromarray((VAE_reconstructed_image * 255).astype("uint8"), mode='L').save(VAE_save_path)

        # Save SR reconstructed image
        SR_reconstructed_image = sr_output.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze()
        SR_reconstructed_image = (SR_reconstructed_image - SR_reconstructed_image.min()) / (
                    SR_reconstructed_image.max() - SR_reconstructed_image.min())
        SR_save_path = os.path.join(SR_path, f"SR_reconstructed_{image_file}")
        Image.fromarray((SR_reconstructed_image * 255).astype("uint8"), mode='L').save(SR_save_path)

        print(f"Error for image {image_file}: {error}")

    error_sum = sum(error_list)

    return error_sum


# Example usage
if __name__ == '__main__':
    model_weights_folder_path = "F:\\test_logs\\VanillaVAE"
    image_folder_path = "F:\\STEM_Data_ori\\SRDATA\\validation\\LR_original"
    label_folder_path = "F:\\STEM_Data_ori\\SRDATA\\validation\\HR"
    image_output_folder_path = "F:\\test_preprocess_model_different_hyparam"

    # Initialize variables to track the overall minimum error and corresponding weights
    overall_min_error = {}

    for model_path in os.listdir(model_weights_folder_path):
        model_path_checkpoint = os.path.join(model_weights_folder_path, model_path, 'checkpoints', 'last.ckpt')
        image_output_folder_path_sub = os.path.join(image_output_folder_path, model_path)
        os.makedirs(image_output_folder_path_sub, exist_ok=True)

        # Process images for this model and get the error and best weights
        error = process_images(model_path_checkpoint, image_folder_path, label_folder_path, image_output_folder_path_sub)
        overall_min_error[model_path] = error

    # Update overall minimum error and corresponding weights if needed
    with open(os.path.join(image_output_folder_path, 'weight_analysis_results.txt'), 'a+') as file:
        file.write(f"{overall_min_error}")
    # Update overall minimum error and corresponding weights if needed
    min_value = min(overall_min_error.values())
    for key, value in overall_min_error.items():
        if value == min_value:
            overall_best_weights = key
            print(f"Overall minimum error: {min_value}")
            print(f"Best weights corresponding to the minimum error: {overall_best_weights}")


