from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
import glob

class CustomDataset(Dataset):
    def __init__(self, source_path, input_channels):
        self.input_channels = input_channels
        self.source_path = source_path
        self.source_images = sorted(glob.glob(source_path + '*'))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, idx):
        image_path = self.source_images[idx]
        image = cv2.imread(image_path)
        if self.input_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, dsize=(64,64))
        image_tensor = self.to_tensor(image)
        image_tensor = self.normalize(image_tensor)
        return image_tensor



#!   test
if __name__ == "__main__":
    dataset_path = r"D:\\Projects\\Datasets\\faces\\"
    dataset = CustomDataset(dataset_path, 3)
    dataloader = DataLoader(dataset, 1, shuffle=False)
    real_batch = next(iter(dataloader))
    tensor_as_image = real_batch[0].cpu().numpy().transpose(1,2,0)
    cv2.imshow("frame", tensor_as_image)
    cv2.waitKey(0)