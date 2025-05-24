import torch
import numpy as np

print("=== Initializing a Tensor")

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

print(x_data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(x_np)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)

rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

print("=== Attributes of a Tensor ==\n")

tensor = torch.rand(3,4)

print(f"Shape of tensor:\n {tensor.shape}")
print(f"Datatype of tensor:\n {tensor.dtype}")
print(f"Device tensor is stored on:\n {tensor.device}")

# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
    
print(f"Device tensor is stored on:\n {tensor.device}")

tensor = torch.ones(4,4)

print("\n")
print("first row:\n")
print(tensor[0])

print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

tensor[:,1] = 0

print(tensor)

print("\n")

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print("\n")
print(t1)

print("\n")