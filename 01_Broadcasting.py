#%%
import torch
#%%
a = torch.arange(10)
b = torch.arange(10)*10
print(a)
print(b)

#%%
a+b
#%%
a.shape, b.shape

#%%
print(a[None].shape) # Q. What is this?
"""
Now, in order to make a an order two tensor
we can index into it with the keyword None.
And so, what this will do is it will add,
it will make the tensor into the order two tensor,
and it will add a (mumbles), a dimension of size one
to the beginning of the tensor.
"""

#%%
print(a[:, None].shape) # Q. What is this?
"""
Now, if you would want to add this dimension of size one
to the end of the tensor you would call colon comma None.
And so, this will add an empty dimension,
or a size one dimension, to the end of the tensor.
"""

#%%
"""
We can also add it to the beginning and the end
by calling None (mumbles) colon None."""
print(a[None,:,None].shape)

#%%
c = a[None,:]
c.shape

#%%
c[0,:].shape

#%%
(a[None,:]+b[None,:])

#%%
torch.ones(3,2) + 10 # broadcasting #1
"""
So, what PyTorch does here is it takes the value 10,
and it adds it to every single element
inside of this tensor.
"""

#%%
torch.ones(3,1) + torch.ones(1,3)
# Broadcasting #2
"""
And so, this is another instance of broadcasting.
So, this is similar to the operation here
where you added the value of 10 to every single input.
We now add the second vector
to every single input, or to every single configuration
of the first vector.
And so, what happens internally in PyTorch is that
this first tensor here will be replicated
three times along the second dimension.
The second tensor will be replicated three times
along the first dimension, and then the two tensors
have the same shape and can be added together.
And this replication is referred to as broadcasting.
"""

#%%
a[None,:] + b[:,None]