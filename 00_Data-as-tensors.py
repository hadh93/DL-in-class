#%%
print()
# %%
import torch
from PIL import Image
import numpy as np
#%%
# torch.zeros(10)
# torch.ones([2,3])
v = torch.ones(5,2,3,4)
print(v.size(3))

v = torch.ones(5,2)
v.view(2, -1) # Q: What does this mean?
"""
A: it will automatically figure out
in order to reshape the tensor
into something that's two by x, x needs to be five.
"""

I = Image.open('cat.jpg')
np.array(I)
#%% 
from torchvision import transforms
image_to_tensor = transforms.ToTensor()
image_tensor = image_to_tensor(I)
tensor_to_image = transforms.ToPILImage()
tensor_to_image(image_tensor)
# %%
