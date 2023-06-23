import torch
import kornia


def crop_image(image, boxes, output_size):
    # Convert boxes from (x1, y1, x2, y2) to (y1, x1, y2, x2) format expected by kornia.crop_and_resize()
    boxes = torch.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], dim=-1)

    # Normalize boxes to be between 0 and 1
    height, width = image.shape[-2], image.shape[-1]
    boxes[:, [0, 2]] /= height
    boxes[:, [1, 3]] /= width

    # Perform the crop using kornia.crop_and_resize()
    cropped = kornia.crop_and_resize(image, boxes, output_size)

    return cropped

import torch
import kornia

# Define input image tensor
batch_size = 2
image_size = 256
input_image = torch.randn(batch_size, 3, image_size, image_size)

# Define crop parameters

bbox = torch.stack([torch.tensor([[0, 0], [50, 0], [50,50], [0, 50]]), torch.tensor([[0, 0], [50, 0], [50,50], [0, 50]])])
#torch.tensor([[0.1, 0.2, 0.5, 0.6], [0.3, 0.4, 0.7, 0.8]])  # [xmin, ymin, xmax, ymax]

# Perform differential crop and resize
cropped_image = kornia.geometry.transform.crop_and_resize(input_image, bbox, (128, 128))

# Compute gradients


# Print output and gradient information
print('Cropped image shape:', cropped_image.shape)
print('Cropped image sum:', cropped_image.sum().item())

