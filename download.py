import kagglehub

# Download latest version
path = kagglehub.model_download("kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2")

print("Path to model files:", path)