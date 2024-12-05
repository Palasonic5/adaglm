import torch

# Paths to your existing tensor files
tensor_file_path1 = '/gpfsnyu/scratch/qz2086/AdaGLM/data/combined/e_g_tokenized_text_abs_full.pt'
tensor_file_path2 = '/gpfsnyu/scratch/qz2086/AdaGLM/data/Mathematics/tokenized_text_abs_full.pt'

print(tensor_file_path1, flush = True)
print(tensor_file_path2, flush = True)

print('loading data', flush = True)
# Load tensor data from both files
tensor_data1 = torch.load(tensor_file_path1)
tensor_data2 = torch.load(tensor_file_path2)

overlap_keys = set(tensor_data1.keys()) & set(tensor_data2.keys())
if overlap_keys:
    print(f"Warning: Overlapping data found in nodes {overlap_keys}.")

print('saving data', flush = True)
# Combine the dictionaries by updating the first with the second
# Assuming no overlapping keys, or if overlapping, second file's data is preferred
tensor_data1.update(tensor_data2)

# Path for the combined tensor file
combined_tensor_file_path = '/gpfsnyu/scratch/qz2086/AdaGLM/data/combined/mag_tokenized_text_abs_full.pt'

# Save the combined tensor data to a new file
torch.save(tensor_data1, combined_tensor_file_path)

print(f"Combined tensor data has been saved to {combined_tensor_file_path}.")


