import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def unnormalize(img_tensor, mean, std):
    # img_tensor: (3, H, W), torch.Tensor
    img = img_tensor.clone().cpu().numpy()
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))  # (H, W, 3)

def run_inference_and_gradcam(model, image_tensor, mean, std):
    """
    model: your SimpleModel (should be in eval mode)
    image_tensor: (3, H, W) torch.Tensor, normalized
    mean, std: normalization params (list of 3 floats)
    Returns: prediction (int), gradcam visualization (np.ndarray, H, W, 3)
    """
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))  # add batch dim
        pred_class = torch.argmax(output, dim=1).item()

    # GradCAM - find the correct target layer
    target_layer = None
    if hasattr(model, 'model') and hasattr(model.model, 'layer4'):
        target_layer = model.model.layer4[-1]
    elif hasattr(model, 'layer4'):
        target_layer = model.layer4[-1]
    else:
        # Try to find the last convolutional layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
    
    if target_layer is None:
        raise ValueError("Could not find a suitable target layer for GradCAM")
    
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0), targets=targets)[0]

    # Unnormalize image for visualization
    unnorm_img = unnormalize(image_tensor, mean, std)
    visualization = show_cam_on_image(unnorm_img, grayscale_cam, use_rgb=True)
    return pred_class, visualization
