# Import packages

import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader

# Run on GPU
from labels_color import color

device = torch.device("cuda:0")
print(device)
print('Using {} device\n'.format(device))


# Importing dataset

class Data(Dataset):
    def __init__(self, img_dir, gt_dir, seg_dir):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.seg_dir = seg_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.images = sorted(os.listdir(img_dir))
        self.gts = sorted(os.listdir(gt_dir))
        self.segs = sorted(os.listdir(seg_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        gt_path = os.path.join(self.gt_dir, self.gts[index])
        seg_path = os.path.join(self.seg_dir, self.segs[index])
        image = Image.open(img_path)
        trans_image = self.transform(image)
        gt = Image.open(gt_path)
        trans_gt = self.transform(gt)
        seg = Image.open(seg_path)
        trans_seg = self.transform(seg)
        return trans_image, torch.tensor(np.array(gt)), trans_seg


img_dir = 'training/image_2'
gt_dir = 'training/semantic'
seg_dir = 'training/semantic_rgb'
data_set = Data(img_dir, gt_dir, seg_dir)
data_set_train, data_val1 = train_test_split(data_set, test_size=0.3, train_size=0.7, shuffle=False)
dataset_val, dataset_test = train_test_split(data_val1, test_size=0.5, train_size=0.5, shuffle=False)

# Loading the data

train_dataloader = DataLoader(data_set_train, batch_size=1, shuffle=True)
test_dataloader = DataLoader(dataset_test, batch_size=1)
val_dataloader = DataLoader(dataset_val, batch_size=1)

# FCN-32 Model

n_classes = 34


class FCN32(nn.Module):
    def __init__(self):
        super(FCN32, self).__init__()

        self.Resnet = models.resnet18(pretrained=True)
        for parameters in self.Resnet.parameters():
            parameters.requires_grad = False
        self.modules = list(self.Resnet.children())[:-2]  # Till last fc layer.
        self.resnet = nn.Sequential(*self.modules)
        self.conv = nn.Conv2d(512, 512, kernel_size=2)  # Instead of 100 padding
        self.conv1 = nn.Conv2d(512, n_classes + 1, kernel_size=1)
        self.upsample = nn.ConvTranspose2d(in_channels=n_classes + 1, out_channels=n_classes + 1, kernel_size=64,
                                           stride=32)

    def forward(self, x):
        gt = x
        x = self.resnet(x)
        x = F.relu(self.conv(x))
        x = F.relu(self.conv1(x))
        x = self.upsample(x)
        output = x
        output = output[:, :, output.shape[2] - gt.shape[2]:output.shape[2] - gt.shape[2] + output.size()[2],
                 output.shape[3] - gt.shape[3]:output.shape[3] - gt.shape[3] + output.size()[3]].contiguous()
        return output


# Hyperparameters

fcn32_model = FCN32()
epochs = 6
lr = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fcn32_model.parameters(), lr=lr)


# Train and validate function

def train_and_val(model):
    # Training step

    model.train()
    train_running_loss = 0
    for i, data in enumerate(train_dataloader):
        inputs, gts, segs = data
        inputs = Variable(inputs)
        optimizer.zero_grad()
        outputs = fcn32_model(inputs.float())
        loss = criterion(outputs.float(), gts.long())
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()
    train_loss = train_running_loss / len(train_dataloader)
    print(f'Training loss = {train_loss}')

    # Validation step

    model.eval()
    val_running_loss = 0
    for i, data in enumerate(val_dataloader):
        inputs, gts, segs = data
        inputs = Variable(inputs)
        outputs = fcn32_model(inputs.float())
        val_loss = criterion(outputs.float(), gts.long())
        val_running_loss += val_loss.item()
    val_loss = val_running_loss / len(val_dataloader)
    print(f'validation loss = {val_loss}')

    return train_loss, val_loss


# Training and validation

train_losses = []
val_losses = []
for epoch in range(epochs):
    print(f'Epoch : {epoch + 1}')
    if epoch >= 1:
        fcn32_model.load_state_dict(torch.load(path))  # Load the previous model
        if epoch % 20 == 0:
            lr *= 0.5  # Decaying learning rate
    optimizer = torch.optim.Adam(fcn32_model.parameters(), lr=lr)
    tr_loss, val_loss = train_and_val(fcn32_model)
    train_losses.append(tr_loss)
    val_losses.append(val_loss)
    path = 'fcn_model.pt'
    torch.save(fcn32_model.state_dict(), path)  # Save the model
    print('Model Saved!\n')


# Plotting the outputs

def seg_plot(pred, gr_truth, seg_rgb):
    image, gt, seg = pred, gr_truth, seg_rgb
    pred_img = torch.argmax(image.data, 1)

    # Mapping pixels to color
    color_pred_img = torch.zeros((pred_img[0].shape[0], pred_img[0].shape[1], 3))
    for i in range(pred_img[0].shape[0]):
        for j in range(pred_img[0].shape[1]):
            color_pred_img[i][j] = (1/255) * color(pred_img[0][i][j])
    color_pred_image = torch.detach(color_pred_img).numpy()
    gt = torch.squeeze(gt).detach().numpy()
    seg = torch.squeeze(seg)
    seg = seg.permute(1,2,0)    # Reshaping seg_rgb
    seg = torch.detach(seg).numpy()
    fig, (img_axis, gt_axis, seg_rgb_axis) = plt.subplots(1, 3, figsize=(20, 15))
    img_axis.imshow(color_pred_image)
    img_axis.set_title('Image')
    gt_axis.imshow(gt)
    gt_axis.set_title('Ground truth')
    seg_rgb_axis.imshow(seg)
    seg_rgb_axis.set_title('Seg_rgb')
    plt.show()


# IOU calculation

def pixel_IOU(pred, gt):  # Pixel level IOU
    with torch.no_grad():
        pred_class = torch.argmax(F.softmax(pred, dim=1), dim=1)
        tp = torch.eq(pred_class, gt).int()
        pix_iou = float(tp.sum()) / float(tp.numel())
    return pix_iou


def mean_IOU(gt_pred, gt_act, rate=1e-10, n_classes=35):  # Mean IOU and per class IOU
    with torch.no_grad():
        gt_pred = F.softmax(gt_pred, dim=1)  # The probability of classes
        gt_pred = torch.argmax(gt_pred, dim=1)  # Class with max probability
        gt_pred = gt_pred.contiguous().view(-1)
        gt_act = gt_act.contiguous().view(-1)

        class_iou = []
        for classes in range(0, n_classes):  # loop per pixel class
            tp_classes = gt_pred == classes
            tp_segs = gt_act == classes

            if tp_segs.long().sum().item() == 0:  # no exist label in this loop
                class_iou.append(np.nan)
            else:
                intersect = torch.logical_and(tp_classes, tp_segs).sum().float().item()
                union = torch.logical_or(tp_classes, tp_segs).sum().float().item()

                iou = (intersect + rate) / (union + rate)
                class_iou.append(iou)
        mean_class_iou = np.nanmean(class_iou)
        return mean_class_iou, class_iou

# Testing the model

def test():
    model = fcn32_model
    path = 'fcn_model.pt'
    model.load_state_dict(torch.load(path))
    model.eval()
    for i, data in enumerate(test_dataloader):
        inputs, gts, segs = data
        inputs = Variable(inputs)
        outputs = model(inputs.float())
        seg_plot(outputs, gts, segs)
        test_loss = criterion(outputs.float(), gts.long())
        print(f'Test loss = {test_loss.item()}')
        MIoU, PIoU = mean_IOU(outputs, gts)
        print(f'Mean IoU = {MIoU}, Per Class IoU = {PIoU}')
        pixel_iou = pixel_IOU(outputs, gts)
        print(f'Pixel level IOU = {pixel_iou}')


test()
