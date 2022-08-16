import torch
import cppcuda

feats = torch.ones(2)
point = torch.zeros(2)

out = cppcuda.trilinear_interpolation(feats, point)

print(out)
