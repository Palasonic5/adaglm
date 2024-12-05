import torch
import random

# Paths to your existing tensor files
file_path1 = '/gpfsnyu/scratch/qz2086/AdaGLM/data/combined/e_g/tensor_data_10_shuffled_1.pt'
file_path2 = '/gpfsnyu/scratch/qz2086/AdaGLM/data/combined/e_g/tensor_data_10_shuffled_2.pt'

print('loading 2 files', flush = True)
print(file_path1)
print(file_path2)

# Load data from both files
data1 = torch.load(file_path1)
data2 = torch.load(file_path2)

# print('shuffling data', flush = True)
# random.shuffle(data1)
# split_point = len(data1) // 2

# Split the data into two parts
# data_part1 = data1[:split_point]
# data_part2 = data1[split_point:]


print('combining data', flush = True)
# Combine the lists
combined_data = data1 + data2

# Path for the combined tensor file
combined_file_path1 = '/gpfsnyu/scratch/qz2086/AdaGLM/data/combined/e_g/tensor_data_10_shuffled_1.pt'
# combined_file_path2 = '/gpfsnyu/scratch/qz2086/AdaGLM/data/combined/tensor_data_10_shuffled_2.pt'

print('saving data', flush = True)
# Save the combined data to a new file
torch.save(combined_data, combined_file_path1)
# torch.save(data_part2, combined_file_path2)

print(f"Combined tensor data has been saved")
