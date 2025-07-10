from pytorch_lightning.callbacks import Callback
import torch
import os
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import wandb
import random

from data.transforms import denormalize


class GradCAMCallback(Callback):
    def __init__(self, model, config, outdir='results', log_every_n_epochs=10):
        """
        GradCAM code is designed for classification models where the output corresponds to class scores: output is a tensor of size (batch_size, num_classes).
        In FusionModel the model outputs embeddings, so using the class indices directly to index would not work.

        Args:
            model: The model to use for GradCAM visualizations.
            config: Configuration dict, should include the `use_wandb` flag.
            outdir: Directory to save GradCAM images locally.
            log_every_n_epochs: Interval at which GradCAM images are logged.
        """
        self.model = model
        self.config = config
        self.outdir = outdir
        self.log_every_n_epochs = log_every_n_epochs
        self.kp_included = config['preprocess_lvl'] >= 3 and config['model_architecture'] in ['FusionModel', 'ResNetPlusModel']

        self.arcface = True if config['arcface_loss']['activate'] else False



    def on_validation_epoch_end(self, trainer, pl_module):

        # Check if the current epoch is a multiple of log_every_n_epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return  # Skip GradCAM computation

        # Check if validation dataloader exists
        if not trainer.val_dataloaders:
            print("Validation dataloaders not found or empty.")
            return  # Exit if no validation dataloader is available

        if self.config['model_architecture'] in ['SimpleModel']:
            target_layer = self.get_target_layer(pl_module)
        else:
            target_layer = self.get_resnet50_layer4(pl_module)
        
        if target_layer is None:
            print("Target layer not found for GradCAM.")
            return

        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        # Proceed with GradCAM logic if dataloaders exist
        pl_module.eval()  # Ensure model is in evaluation mode
        for batch_idx, batch in enumerate(val_loader):
            # Handle both dict and tuple batches
            if isinstance(batch, dict):
                x = batch['img']
                target = batch['label']
            else:
                x, target = batch
            
            idx = random.randint(0, x.shape[0] - 1)  # Random idx
            # Move inputs and targets to the device
            x = x.to(pl_module.device)
            target = target.to(pl_module.device)

            if self.kp_included:
                # TODO: Implement GradCAM for higher dimension inputs
                pass
                # x = x[:, :3, :, :] # only take rgb channels
                # then run through forward with only rgb


            # For the visualization, denormalize
            x_np = x[idx].cpu().numpy()  # Convert from PyTorch tensor to numpy array
            unnormalized_x = denormalize(x_np, self.config['transforms']['mean'], self.config['transforms']['std'])
            unnormalized_x = unnormalized_x[:3]  # RGB only: (3, 224, 224)
            # Reformat unnormalized_x back to (224, 224, 3) for visualization
            unnormalized_x = np.transpose(unnormalized_x, (1, 2, 0))  # Shape is now (224, 224, 3)
            # x_debug = unnormalized_x #debug
            unnormalized_x = (unnormalized_x-np.min(unnormalized_x))/(np.max(unnormalized_x)-np.min(unnormalized_x))
            unnormalized_x = np.clip(unnormalized_x, 0, 1)

            with torch.no_grad():
                embeddings = pl_module(x)
            with torch.enable_grad():
                # cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])
                cam = GradCAM(model=pl_module, target_layers=[target_layer])

                if self.config['model_architecture'] in ['SimpleModel']:
                    # For classification models, use the predicted class
                    with torch.no_grad():
                        logits = pl_module(x[idx:idx+1])
                        pred_class = torch.argmax(logits, dim=1).item()
                    targets = [ClassifierOutputTarget(pred_class)]
                else:
                    # Following targets fails if self.model outputs embeddings of size 128, and class indices higher
                    # targets = [ClassifierOutputTarget(class_idx) for class_idx in target] # only works for class logits, NOT embedding models
                    # 
                    # alternative for embedding model:
                    embeddings = pl_module(x)
                    if self.arcface:
                        # ArcFace target for the query image
                        query_label = target[idx].item()
                        
                        # Access ArcFace loss module
                        arcface_loss = pl_module.loss_fn if hasattr(pl_module, 'loss_fn') else None
                        if arcface_loss is None or not pl_module.config["arcface_loss"]["activate"]:
                            print(f"Warning: ArcFace loss not found in batch {batch_idx}, using norm fallback")
                            targets = [lambda _: torch.norm(embeddings[idx], p=2)]
                        else:
                            def arcface_target(embeds):
                                # Compute logits for the single embedding
                                logits = arcface_loss.get_logits(embeds.unsqueeze(0))  #  ensure scalar
                                return logits[0, query_label]
                            targets = [arcface_target]
                    else:
                        # maximize sum of embeddings, forcing the model to focus on regions that contribute to a strong embedding (possibly discriminative features)
                        print('Using norm of query embedding as fallback')
                        targets = [lambda _: torch.sum(embeddings)]
                # 

                grayscale_cam = cam(input_tensor=x[idx:idx+1], targets=targets)[0, :]
                grayscale_cam = np.repeat(grayscale_cam[:, :, np.newaxis], 3, axis=2) # make it compatible with x - 3 channels
                visualization = show_cam_on_image(unnormalized_x, grayscale_cam, use_rgb=True)
                img = Image.fromarray((visualization * 255).astype(np.uint8))

                # Log GradCAM image to W&B if enabled
                if self.config.get('use_wandb', False):
                    wandb_img = wandb.Image(visualization, caption=f"GradCAM Epoch {trainer.current_epoch + 1} Batch {batch_idx} Image 0")
                    pl_module.logger.experiment.log({"GradCAM Images": wandb_img})

                    # #below is for debugging purposes
                    # x_img = Image.fromarray(x_debug)
                    # x_img.save(os.path.join(self.outdir, f'input image{trainer.current_epoch + 1}_batch{batch_idx}_img0.jpg'))
                    # wandb_img = wandb.Image(x_img, caption=f"image input {trainer.current_epoch + 1} Batch {batch_idx} Image 0")
                    # pl_module.logger.experiment.log({"Images": wandb_img})

                # Save locally
                os.makedirs(self.outdir, exist_ok=True)
                img.save(os.path.join(self.outdir, f'cam_image_val_epoch{trainer.current_epoch + 1}_batch{batch_idx}_img0.png'))    

            # Limit the number of batches for GradCAM to avoid excessive logs
            if batch_idx >= 2:
                break

        pl_module.eval()  # Reset to eval mode
        pl_module.train()  # Set the model back to training mode

    # def get_resnet50_layer4(self, model):
    #     """
    #     Retrieve the layer4 of a ResNet50 model.
    #     """
    #     try:
    #         backbone = model.backbone

    #         if hasattr(backbone, 'layer4'):
    #             return backbone.layer4[-1]  # Return the last block of layer4 for GradCAM
    #     except:
    #         print('model has no backbone')
            
    #     if hasattr(model, 'layer4'):
    #         return model.layer4[-1]  # Use the last block of layer4 for GradCAM
    #     elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
    #         # In case the backbone is wrapped inside another model
    #         return model.backbone.layer4[-1]
    #     elif hasattr(model, 'model') and hasattr(model.model, 'layer4'):
    #         # If wrapped inside a SimpleModel or another object
    #         return model.model.layer4[-1]
    #     else:
    #         print("layer4 not found in the model.")
    #         return None


    def get_resnet50_layer4(self, model):
        """
        Retrieve the layer4 of a ResNet50 model.
        """
        try:
            if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
                return model.backbone.layer4[-1]
            print("layer4 not found.")
            return None
            
        except Exception as e:
            print(f"Error accessing layer4: {e}")
            return None

    def get_target_layer(self, model):
        """
        Retrieve the target layer for GradCAM visualization.
        Supports ResNet variants (ResNet50, ResNet152, etc.) and other models.
        """
        try:
            # For SimpleModel and TransformerCategory with ResNet backbones
            if hasattr(model, 'model') and hasattr(model.model, 'layer4'):
                return model.model.layer4[-1]  # Return the last block of layer4 for GradCAM
            
            # For models with backbone attribute (embedding models)
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
                return model.backbone.layer4[-1]
            
            # For direct ResNet models
            if hasattr(model, 'layer4'):
                return model.layer4[-1]
            
            # For other architectures, try to find suitable layers
            if hasattr(model, 'model'):
                # Try to find the last convolutional layer
                for name, module in reversed(list(model.model.named_modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        print(f"Using convolutional layer '{name}' for GradCAM")
                        return module
            
            print("No suitable target layer found for GradCAM.")
            return None
            
        except Exception as e:
            print(f"Error accessing target layer: {e}")
            return None
