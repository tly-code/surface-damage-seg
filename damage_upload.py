import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio
import torchvision.transforms.functional as TF
import sklearn.metrics
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torchmetrics import JaccardIndex, F1Score, CohenKappa, Accuracy, Dice 
from segmentation_models_pytorch.losses import DiceLoss
from sklearn.metrics import matthews_corrcoef


class ToTensor:  # Preprocess the dataset to the [0, 1] range
    def __call__(self, image, label):
   
        image = torch.from_numpy(image).float() / 255.0  # Normalize to the [0, 1] range
        label = torch.from_numpy(label).float() / 255.0  # Normalize to the [0, 1] range
        label = label.unsqueeze(0) #(256,256)->(1,256,256)
        return image, label  #image (3,256,256) [0, 1]; label (1,256,256) [0, 1]

class Rotate_ToTensor:  # Preprocess the dataset and apply random rotation
    def __call__(self, image, label):
        # Randomly select the rotation angle
        angle = random.randint(-45, 45) # Randomly select a rotation angle between -45 and 45 degrees
        
        # Convert the image to a tensor and normalize it to the [0, 1] range
        image = torch.from_numpy(image).float() / 255.0
        label = torch.from_numpy(label).float() / 255.0
        label = label.unsqueeze(0)  # Change the label shape from (256, 256) to (1, 256, 256)
        
        # Rotate both the image and the label
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.NEAREST)
        label = TF.rotate(label, angle, interpolation=TF.InterpolationMode.NEAREST)

        return image, label # Return the processed image (3, 256, 256) in [0, 1] and label (1, 256, 256) in [0, 1]

class Rotate_BrightnessContrast_ToTensor:  # Preprocess the dataset and apply random rotation, random brightness adjustment, and random contrast adjustment
    def __call__(self, image, label):
        # Randomly select the rotation angle
        angle = random.randint(-45, 45)  # Randomly select a rotation angle between -45 and 45 degrees
        
        # Convert the image to a tensor and normalize it to the [0, 1] range
        image = torch.from_numpy(image).float() / 255.0
        label = torch.from_numpy(label).float() / 255.0
        label = label.unsqueeze(0)  # Change the label from (256, 256) to (1, 256, 256)
        
        # Rotate both the image and the label
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.NEAREST)
        label = TF.rotate(label, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Random brightness adjustment
        if random.random() < 0.5:
            brightness_factor = 1.0 + random.uniform(-0.25, 0.25)
            image = TF.adjust_brightness(image, brightness_factor)

        # Random contrast adjustment
        if random.random() < 0.5:
            contrast_factor = 1.0 + random.uniform(-0.20, 0.20)
            image = TF.adjust_contrast(image, contrast_factor)
        
        image = image.clamp(0.0, 1.0)
        
        return image, label # Return the processed image (3, 256, 256) in [0, 1] and label (1, 256, 256) in [0, 1]

class Rotate_Brightness_ToTensor:  # Preprocess the dataset and apply random rotation and random brightness adjustment
    def __call__(self, image, label):
       # Randomly select the rotation angle
        angle = random.randint(-45, 45)  # Randomly select a rotation angle between -45 and 45 degrees 
        
       # Convert the image to a tensor and normalize it to the [0, 1] range
        image = torch.from_numpy(image).float() / 255.0
        label = torch.from_numpy(label).float() / 255.0
        label = label.unsqueeze(0)  # Change the label from (256, 256) to (1, 256, 256)
        
        # Rotate both the image and the label
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.NEAREST)
        label = TF.rotate(label, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Random brightness adjustment
        if random.random() < 0.5:
            brightness_factor = 1.0 + random.uniform(-0.25, 0.25)
            image = TF.adjust_brightness(image, brightness_factor)
        
        image = image.clamp(0.0, 1.0)
        
        return image, label  # Return the processed image (3, 256, 256) in [0, 1] and label (1, 256, 256) in [0, 1]
    
class Flip_ToTensor:  # Preprocess the dataset and apply random flipping
    def __call__(self, image, label):    
        # Convert the image to a tensor and normalize it to the [0, 1] range
        image = torch.from_numpy(image).float() / 255.0       # C x H x W
        label = torch.from_numpy(label).float() / 255.0
        label = label.unsqueeze(0)  # # 1 x H x W
    
        # Random horizontal flipping
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])  # flip width
            label = torch.flip(label, dims=[2])

        # Random vertical flipping
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])  # flip height
            label = torch.flip(label, dims=[1])

        return image, label

class DamageDataset(Dataset):   #image->(channel=3, h, w)  label->(channel=1, h, w) 
    def __init__(self, image_dir, label_dir, list_file, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        with open(list_file, 'r') as f:
            self.file_names = [line.strip() for line in f if line.strip()]

        self.file_names = [os.path.splitext(name)[0] for name in self.file_names]

        if len(self.file_names) == 0:
            raise ValueError("Folder is empty")

        sample_image_path = os.path.join(self.image_dir, f"{self.file_names[0]}.tif")
        if not os.path.exists(sample_image_path):
            raise FileNotFoundError(f"Sample image not found: {sample_image_path}")
        
        with rasterio.open(sample_image_path) as src:
            self.input_channels = src.count 
            self.image_width = src.width
            self.image_height = src.height
        
        self.image_mean = np.zeros(self.input_channels)
        self.image_std = np.zeros(self.input_channels)
        self.calculate_stats()  
    
    def calculate_stats(self):
        all_images = []

        for file_name in self.file_names:
            image_path = os.path.join(self.image_dir, f"{file_name}.tif")

            with rasterio.open(image_path) as src:
                image = src.read()  
                all_images.append(image)

        all_images = np.stack(all_images)

        for band in range(self.input_channels):
            self.image_mean[band] = np.mean(all_images[:, band, :, :])
            self.image_std[band] = np.std(all_images[:, band, :, :])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image_path = os.path.join(self.image_dir, f"{file_name}.tif")
        label_path = os.path.join(self.label_dir, f"{file_name}_mask.tif")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.uint8)
    
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.uint8) # label is a single-channel image: 0 -> non-damage, 255 -> damage

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()
            mask = mask.repeat(3,1,1)

        return image, label

    def get_num_images(self):
        return len(self.file_names)

    def get_data_dimensions(self):
        if len(self.file_names) == 0:
            raise ValueError("No image is found.")

        file_name = self.file_names[0]
        image_path = os.path.join(self.image_dir, f"{file_name}.tif")
        label_path = os.path.join(self.label_dir, f"{file_name}_mask.tif")

        with rasterio.open(image_path) as src:
            image = src.read()
        with rasterio.open(label_path) as src:
            label = src.read()

        return image.shape, image.dtype, label.shape, label.dtype

    def print_info(self):
        num_images = self.get_num_images()
        try:
            image_shape, image_dtype, label_shape, label_dtype = self.get_data_dimensions()
            print(f"The dataset consists of {num_images} images.")
            print(f"The dimension of the image: {image_shape} (channels, height, width),dtype: {image_dtype}")
            print(f"The dimension of the label: {label_shape} (channels, height, width),dtype: {label_dtype}")
            print(f"The input channel of the image: {self.input_channels}")
            for band in range(self.input_channels):
                print(f"The MEAN and STD of channel {band} of the original input images are: {self.image_mean[band]} and {self.image_std[band]}")   
        except Exception as e:
            print(f"Cannot get the dimension info: {e}")

class BCEDiceLoss(nn.Module):
    """
    Binary Cross Entropy + dice_weight * Dice
    """
    def __init__(self,dice_weight=1.0, pos_weight = None):
        super().__init__()
        self.BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.Dice = DiceLoss(mode="binary", from_logits=True) 
        self.dice_weight = float(dice_weight)

    def forward(self, logits, target):
        """
        logits: [B, C, H, W]  
        target: [B, H, W]     
        """

        logit_fg = (logits[:, 1] - logits[:, 0]).unsqueeze(1)  # [B,1,H,W]
        target_fg = target.float()

        bce_loss = self.BCE(logit_fg, target_fg)
        dice_loss = self.Dice(logit_fg, target_fg)

        total = bce_loss + self.dice_weight * dice_loss

        return total, {"bce": bce_loss.detach(), "dice": dice_loss.detach()}
    

class SegFormerDamageModel(nn.Module):
    def __init__(self, num_channels=3, num_labels=2, pretrained_model_name='nvidia/segformer-b2-finetuned-ade-512-512'): 

        super(SegFormerDamageModel, self).__init__()

        if pretrained_model_name:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            id2label={0: "background", 1: "damage"} if num_labels == 2 else {i: f"class_{i}" for i in range(num_labels)},
            label2id={"background": 0, "damage": 1} if num_labels == 2 else {f"class_{i}": i for i in range(num_labels)},
            ignore_mismatched_sizes=True,
            )
        else:
            id2label = {0: "background", 1: "damage"} if num_labels == 2 else {i: f"class_{i}" for i in range(num_labels)}  
            label2id = {v: k for k, v in id2label.items()} 

            config = SegformerConfig(
                num_channels=num_channels,
                num_labels = num_labels,
                id2label = id2label,
                label2id = label2id
            )

            self.model = SegformerForSemanticSegmentation(config)
        
        self.config = self.model.config
        
        if num_labels != self.model.config.num_labels:
            self.model = change_num_classes(self.model, num_labels)

    
    def print_model(self):
        print(self.config)
        print(self.model)

    def forward(self, x):
        H, W = x.shape[-2:]
        out = self.model(pixel_values = x)
        logits = out.logits
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits   #[batch, 2, 512, 512]
 
def change_num_classes(model, num_labels):
    in_ch = model.decode_head.classifier.in_channels
    
    model.decode_head.classifier = nn.Conv2d(in_ch, num_labels, kernel_size=1)

    if hasattr(model, "auxiliary_head") and model.auxiliary_head is not None:
        in_ch_aux = model.auxiliary_head.classifier.in_channels        
        model.auxiliary_head.classifier = nn.Conv2d(in_ch_aux, num_labels, kernel_size=1)
    
    model.config.num_labels = num_labels    
    model.config.id2label = {0: "background", 1: "damage"} if num_labels == 2 else {i: f"class_{i}" for i in range(num_labels)}   
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    
    return model

def validate(model, dataloader, criterion, device, num_classes = 2): #return: miou, iou, OA, dice, f1, kappa
    model.eval()
    epoch_loss = 0.0
    epoch_loss_bce = 0.0
    epoch_loss_dice = 0.0
    dice_scores_per_class = [[] for _ in range(num_classes)]  
    f1_scores_per_class = [[] for _ in range(num_classes)]  
    iou_scores_per_class = [[] for _ in range(num_classes)]  
    recall_scores_per_class = [[] for _ in range(num_classes)]  
    precision_scores_per_class = [[] for _ in range(num_classes)]  
    oa = 0
    all_preds = []
    all_masks = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    metric_miou = JaccardIndex(task="multiclass", num_classes=num_classes, average="none").to(device)
    metric_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    metric_dice = Dice(num_classes=num_classes, average="macro").to(device)
    metric_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro").to(device)
    metric_kappa = CohenKappa(task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)    # [batch, 2, 512, 512]

            loss, losses = criterion(outputs, masks)            
            epoch_loss += loss.item()
            epoch_loss_bce += losses["bce"]
            epoch_loss_dice += losses["dice"]
            
            preds = outputs.argmax(dim=1)  # [batch, 512, 512]
            preds = preds.long()
            masks = masks.long()
            if masks.ndim == 4 and masks.size(1) == 1:
                masks = masks[:, 0]  # [batch, 512, 512]

            metric_miou.update(preds, masks)  
            metric_f1.update(preds, masks)
            metric_dice.update(preds, masks)
            metric_acc.update(preds, masks)
            metric_kappa.update(preds, masks)

            all_preds.append(preds.flatten().cpu().numpy())
            all_masks.append(masks.flatten().cpu().numpy())

        # Cohen's Kappa
        all_preds = np.concatenate(all_preds)
        all_masks = np.concatenate(all_masks)        
        kappa = sklearn.metrics.cohen_kappa_score(all_preds, all_masks)
        # Overall Accuracy (OA)
        oa = np.sum(all_preds == all_masks) / all_preds.shape[0]
        #Mcc
        mcc = matthews_corrcoef(all_masks, all_preds)

        for cls in range(num_classes):
            tp = ((all_preds == cls) & (all_masks == cls)).sum().astype(float)
            fp = ((all_preds == cls) & (all_masks != cls)).sum().astype(float)
            fn = ((all_preds != cls) & (all_masks == cls)).sum().astype(float)
            tn = ((all_preds != cls) & (all_masks != cls)).sum().astype(float)

            total_tp += tp.item()
            total_fp += fp.item()
            total_fn += fn.item()
            total_tn += tn.item()

            # IoU
            iou = tp / (tp + fp + fn + 1e-6) 
            iou = iou if (tp+fp+fn) > 0 else float('nan')
            iou_scores_per_class[cls].append(iou.item())

            # Recall
            recall = (tp / (tp + fn + 1e-6))
            recall = recall if (tp + fn) > 0 else float('nan')
            recall_scores_per_class[cls].append(recall.item())

            # Precision
            precision = tp / (tp + fp + 1e-6)
            precision = precision if (tp + fp) > 0 else float('nan')
            precision_scores_per_class[cls].append(precision.item())


            # F1-Score
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            f1_scores_per_class[cls].append(f1.item())

            # Dice 
            dice = (2. * tp) / (2. * tp + fp + fn + 1e-6)  
            dice_scores_per_class[cls].append(dice.item())


    mean_dice_per_class = [np.nanmean(dice_scores_per_class[cls]) for cls in range(num_classes)]
    mean_f1_per_class = [np.nanmean(f1_scores_per_class[cls]) for cls in range(num_classes)]
    mean_iou_per_class = [np.nanmean(iou_scores_per_class[cls]) for cls in range(num_classes)]
    mean_recall_per_class = [np.nanmean(recall_scores_per_class[cls]) for cls in range(num_classes)]
    mean_precision_per_class = [np.nanmean(precision_scores_per_class[cls]) for cls in range(num_classes)]

    micro_f1 = 2 * total_tp / ((total_tp + total_fp) + (total_tp + total_fn))  
    micro_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn) 

    metric_miou = metric_miou.compute()
    metric_f1 = metric_f1.compute()
    metric_dice = metric_dice.compute()
    metric_acc = metric_acc.compute()
    metric_kappa = metric_kappa.compute()

    return (
        epoch_loss / len(dataloader),  # avg loss
        epoch_loss_bce / len(dataloader),  #avg bce loss
        epoch_loss_dice / len(dataloader), #avg dice loss
        mean_dice_per_class,  # Dice of each class
        np.mean(mean_dice_per_class), #mean Dice across classes
        metric_dice,
        micro_dice, 
        mean_iou_per_class,  # IoU of each class
        np.mean(mean_iou_per_class),  # mean IoU across classes
        metric_miou.mean().item(), #miou
        mean_recall_per_class,  # Recall of each class
        np.mean(mean_recall_per_class),  # mean Recall across classes
        metric_acc,
        mean_precision_per_class, # precision of each class
        np.mean(mean_precision_per_class), # mean precision across classes
        mean_f1_per_class, # F1-Score of each class
        np.mean(mean_f1_per_class), # mean F1 across classes
        metric_f1,
        micro_f1, 
        kappa,  # Cohen's Kappa
        metric_kappa,
        oa,   # OA
        mcc #MCC
    )

def save_geotiff(prediction, profile, output_path):
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write((prediction.cpu().numpy()* 255).astype(np.uint8))

def process_image(image_path, model, device):
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.uint8)
        image = torch.from_numpy(image).float() / 255.0
        image = image.unsqueeze(0)  
        profile = src.profile.copy()   
        image = image.to(device)

    with torch.no_grad():
        prediction = model(image)
        prediction = prediction.argmax(dim=1).to(torch.uint8).cpu()    # [1,H,W] uint8
        profile.update(count=1, dtype='uint8')   
    return prediction, profile


def predict_folder_geotiffs(image_folder,output_folder, model, device):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(image_folder):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            image_path = os.path.join(image_folder, filename)

            output_path = os.path.join(output_folder, filename[:-4]+'_pred.tif')
            
            prediction, profile = process_image(image_path, model, device)
            
            save_geotiff(prediction, profile, output_path)
            # print(f"Saved result to {output_path}")
    print(f"Saved result to {output_folder}")

