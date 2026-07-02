# Cat vs. Dog Image Classification — CNN

A binary image classifier built with a Convolutional Neural Network that distinguishes cats from dogs. Classic deep learning hello-world project.

## The dataset

This project uses the **Oxford-IIIT Pet dataset**, downloaded automatically on first run (~800 MB cached to `data/`). It contains 37 cat and dog breeds. The 12 capitalised breeds (Abyssinian, Bengal, etc.) are mapped to `cat=0`; the 25 lowercase breeds (american_bulldog, beagle, etc.) are mapped to `dog=1`.

Split: ~3,680 training images (`trainval`) and ~3,669 test images (`test`).

## The architecture

Pretrained **ResNet18** (ImageNet weights) with the final fully-connected layer replaced by a single sigmoid output node. The entire network is fine-tuned end-to-end rather than freezing the backbone — the Oxford-IIIT Pet images differ enough from ImageNet that full fine-tuning gives better results.

```
ResNet18 backbone (pretrained on ImageNet)
  └── fc: Linear(512 → 1) + Sigmoid
```

Input images are resized to 256×256 then randomly cropped to 224×224 during training (standard ResNet preprocessing). ImageNet mean/std normalisation is applied so the pretrained weights remain meaningful.

## Expected results

After 10 epochs of fine-tuning, validation accuracy is typically **88–92%**. Transfer learning from ImageNet weights gives a large head-start — the network already knows edges, textures, and shapes before seeing a single pet image.

## How to run

```bash
python convolutional_neural_network.py
```

On first run the script looks for ResNet18 pretrained weights in `weights/resnet18-f37072fd.pth`. If the file isn't there it downloads it automatically and caches it in that folder for future runs. You can also point to an existing file manually:

```bash
python convolutional_neural_network.py --weights-path /path/to/resnet18-f37072fd.pth
```

Training takes 10–20 minutes on CPU (much faster on GPU). Per-epoch loss and validation accuracy are shown via tqdm.

## Code structure

```
ImageClassifier
├── _build_loaders()          → downloads OxfordIIITPet, wraps with _BinaryPetDataset, returns DataLoaders
├── _build_model()            → loads pretrained ResNet18, replaces fc with Linear(512→1) + Sigmoid
├── train()                   → fine-tunes all layers with Adam, per-epoch tqdm bars
├── predict(image_path)       → loads a single image and returns "cat" or "dog"
├── save_model()              → saves model weights to a .pt file
└── load_model()              → restores a previously saved model
```

## Notes

This is the only project that downloads its dataset at runtime — you don't need to prepare anything manually. The Oxford-IIIT Pet dataset (~800 MB) is cached to `data/` after the first run so subsequent runs are fast.

ResNet18 pretrained weights (~44 MB) are cached to `weights/` after the first download. Neither folder is tracked by git.

Trained weights are saved to `cnn_model.pt` after training. Use `--save-model path.pt` to change the path, and `load_model()` to restore them for inference without retraining.
