from torchvision import transforms, datasets
import torch


img_size = 224
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                transforms.CenterCrop(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


def get_dataloader(data_dir, batch_size, num_workers, aug=False):
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transform["train" if aug else "val"])
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size, shuffle=aug,
                                         num_workers=num_workers)

    return loader