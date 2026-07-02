import argparse
import random
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from PIL import Image


class _BinaryPetDataset(torch.utils.data.Dataset):
    """Wraps OxfordIIITPet and maps 37-breed labels to cat=0 / dog=1.
    Breeds 0-11 (capitalised names) are cats; 12-36 are dogs.
    """
    _N_CAT_BREEDS: int = 12

    def __init__(self, base: datasets.OxfordIIITPet) -> None:
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple:
        img, label = self._base[idx]
        return img, int(label >= self._N_CAT_BREEDS)


class ImageClassifier:
    DATA_DIR: str = "data"
    IMAGE_SIZE: tuple[int, int] = (224, 224)
    BATCH_SIZE: int = 32
    EPOCHS: int = 10
    LR: float = 1e-4
    _WEIGHTS_DIR: Path = Path("weights")
    _RESNET18_FILENAME: str = "resnet18-f37072fd.pth"
    _RESNET18_URL: str = "https://download.pytorch.org/models/resnet18-f37072fd.pth"

    _IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
    _IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

    def __init__(self) -> None:
        self.model: nn.Module | None = None
        self._classes: list[str] | None = None
        self.weights_path: str | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_loaders(self) -> tuple[DataLoader, DataLoader]:
        train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(self.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(self._IMAGENET_MEAN, self._IMAGENET_STD),
        ])
        test_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(self._IMAGENET_MEAN, self._IMAGENET_STD),
        ])
        train_ds = _BinaryPetDataset(
            datasets.OxfordIIITPet(self.DATA_DIR, split="trainval", download=True, transform=train_tf)
        )
        test_ds = _BinaryPetDataset(
            datasets.OxfordIIITPet(self.DATA_DIR, split="test", download=True, transform=test_tf)
        )
        self._classes = ["cat", "dog"]
        return (
            DataLoader(train_ds, batch_size=self.BATCH_SIZE, shuffle=True),
            DataLoader(test_ds, batch_size=self.BATCH_SIZE),
        )

    def _build_model(self, imagenet_weights_path: str | None = None) -> nn.Module:
        local = Path(imagenet_weights_path) if imagenet_weights_path else (
            self._WEIGHTS_DIR / self._RESNET18_FILENAME
        )
        if local.exists():
            print(f"Loading ResNet18 weights from {local}")
            model = models.resnet18(weights=None)
            model.load_state_dict(
                torch.load(local, map_location="cpu", weights_only=True)
            )
        else:
            print(f"Weights not found at {local} — downloading from PyTorch Hub...")
            try:
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self._WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), local)
                print(f"Weights cached to {local} for future runs.")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download ResNet18 pretrained weights.\n"
                    f"Download manually from:\n"
                    f"  {self._RESNET18_URL}\n"
                    f"Save to: {self._WEIGHTS_DIR / self._RESNET18_FILENAME}\n"
                    "Or pass --weights-path <path to .pth file>"
                ) from exc
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid(),
        )
        return model

    def train(self) -> None:
        train_loader, test_loader = self._build_loaders()
        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        criterion = nn.BCELoss()

        epoch_bar = tqdm(range(1, self.EPOCHS + 1), desc="Epochs", unit="epoch")
        for epoch in epoch_bar:
            self.model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"  Epoch {epoch:02d}", leave=False, unit="batch"):
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            self.model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc="  Validation", leave=False, unit="batch"):
                    images, labels = images.to(self.device), labels.to(self.device)
                    preds = (self.model(images).squeeze(1) > 0.5).long()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            epoch_bar.set_postfix(
                loss=f"{running_loss / len(train_loader):.4f}",
                val_acc=f"{correct / total:.4f}",
            )

    def predict(self, image_path: str) -> str:
        self.model.eval()
        inference_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(self._IMAGENET_MEAN, self._IMAGENET_STD),
        ])
        img = Image.open(image_path).convert("RGB")
        tensor = inference_tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            label_idx = int(self.model(tensor).item() > 0.5)
        return self._classes[label_idx]

    def save_model(self, path: str = "weights/cnn_model.pt") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def demo_predictions(self, n: int = 5) -> None:
        """Pick n random images from the cached dataset and print predicted vs true label."""
        images_dir = Path(self.DATA_DIR) / "oxford-iiit-pet" / "images"
        if not images_dir.exists():
            print("Dataset images not found — skipping prediction demo.")
            return
        samples = random.sample(sorted(images_dir.glob("*.jpg")), n)
        print(f"\n--- Prediction demo ({n} random images) ---")
        correct = 0
        for img_path in samples:
            # Oxford-IIIT Pet naming: Uppercase first letter = cat breed, lowercase = dog breed
            true_label = "cat" if img_path.stem[0].isupper() else "dog"
            predicted = self.predict(str(img_path))
            match = "OK" if predicted == true_label else "WRONG"
            print(f"  [{match:<5}]  {img_path.name:<45}  true={true_label:<4}  pred={predicted}")
            correct += predicted == true_label
        print(f"  {correct}/{n} correct")
        print("-------------------------------------------\n")

    def load_model(self, path: str = "weights/cnn_model.pt") -> None:
        # Rebuild architecture without downloading ImageNet weights
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
        model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model = model.to(self.device)
        self.model.eval()
        self._classes = ["cat", "dog"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cat vs dog image classification — CNN (PyTorch)")
    parser.add_argument("--epochs", type=int, default=ImageClassifier.EPOCHS,
                        help=f"training epochs (default: {ImageClassifier.EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=ImageClassifier.BATCH_SIZE,
                        help=f"batch size (default: {ImageClassifier.BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=ImageClassifier.LR,
                        help=f"learning rate (default: {ImageClassifier.LR})")
    parser.add_argument("--data-dir", type=str, default=ImageClassifier.DATA_DIR,
                        help="dataset download/cache directory (default: data)")
    parser.add_argument("--save-model", type=str, default="weights/cnn_model.pt",
                        help="path to save trained weights (default: weights/cnn_model.pt)")
    parser.add_argument("--weights-path", type=str, default=None,
                        help="path to a locally downloaded resnet18-f37072fd.pth (skips network download)")
    args = parser.parse_args()

    classifier = ImageClassifier()
    classifier.EPOCHS = args.epochs
    classifier.BATCH_SIZE = args.batch_size
    classifier.LR = args.lr
    classifier.DATA_DIR = args.data_dir
    classifier.weights_path = args.weights_path

    print("[1/2] Preparing dataset (Oxford-IIIT Pet, ~800 MB — downloading on first run)...")
    print(f"[2/2] Training CNN for {args.epochs} epochs on {classifier.device}...")
    classifier.train()
    classifier.save_model(args.save_model)
    print("Training complete.")
    print(f"Model saved to {args.save_model}")
    classifier.demo_predictions(n=5)

if __name__ == "__main__":
    main()
