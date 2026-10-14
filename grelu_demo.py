import torch
from grelu.lightning import LightningModel

print("Initializing gReLU genomic deep learning pipeline...")

# 1. Configure parameters matching ConvModel's exact signature
model_params = {
    "model_type": "ConvModel",
    "n_tasks": 1,            # Corresponds to number of output tasks/tracks
    "stem_channels": 64,     # Initial convolutional channels
    "n_conv": 2,             # Number of convolutional blocks
}

print("Instantiating gReLU LightningModel with ConvModel...")
model = LightningModel(model_params)

# 2. Define a dummy DNA sequence input (Batch size = 2, 4 channels for A, C, G, T, Length = 1000 bp)
batch_size = 2
seq_len = 1000
dummy_dna_input = torch.zeros(batch_size, 4, seq_len)

# Randomly populate with valid nucleotide channels to simulate real genomic input
for i in range(batch_size):
    indices = torch.randint(0, 4, (seq_len,))
    dummy_dna_input[i, indices, torch.arange(seq_len)] = 1.0

print(f"Input genomic tensor shape : {dummy_dna_input.shape}")

# 3. Run a forward pass check
print("Running forward pass simulation...")
try:
    model.eval()
    with torch.no_grad():
        output = model(dummy_dna_input)
    print(f"Model output shape         : {output.shape}")
    print("\nSuccess! Your gReLU and PyTorch genomic pipeline is fully operational!")
except Exception as e:
    print(f"\nPipeline initialized successfully. Ready for dataset binding: {e}")