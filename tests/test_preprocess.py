import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# load tensor
x = torch.load("data/cropped/test_user/photo1.pt")

print(x.shape)  # should be [3, 160, 160]

# denormalize from [-1, 1] → [0, 1]
img = (x + 1) / 2

# convert CHW → HWC
img = img.permute(1, 2, 0).numpy()

plt.imshow(img)
plt.axis("off")
plt.show()