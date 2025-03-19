import torch
import torch.nn as nn
import streamlit as st
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Define Generator model (same as used in training)
def Generator(nz, ngf, nc):
    return nn.Sequential(
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        nn.Tanh()
    )

# Set model hyperparameters
nz = 100  # Latent vector size
ngf = 64  # Generator feature maps
nc = 3    # Number of channels (RGB)

# Load trained model
checkpoint_path = "checkpoint.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator(nz, ngf, nc).to(device)
G.load_state_dict(torch.load(checkpoint_path, map_location=device)['G_state_dict'])
G.eval()

# Streamlit UI
st.title("DCGAN Face Generator")
st.write("Click the button below to generate a 4x4 grid of AI-generated faces!")

if st.button("Generate Image"):
    noise = torch.randn(16, nz, 1, 1, device=device)  # Generate 16 images
    with torch.no_grad():
        fake_images = G(noise).cpu()
    
    # Create a 4x4 grid of images
    img_grid = vutils.make_grid(fake_images, nrow=4, normalize=True)
    img_np = img_grid.permute(1, 2, 0).numpy()
    
    # Display image grid
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.axis("off")
    st.pyplot(fig)
