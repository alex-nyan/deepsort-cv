"""
Re-ID feature extractor.

Extracts appearance embeddings from detected bounding boxes using a
pre-trained CNN. The standard DeepSORT uses a model trained on the
Mars pedestrian re-identification dataset.

For this project, we support:
    1. Loading a pre-trained .t7 or .pth Re-ID model
    2. Using torchvision ResNet-based feature extraction as fallback
    3. Dummy features for ablation (isolating motion-only effects)
"""

import numpy as np

try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet18

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DummyExtractor:
    """
    Returns zero-vectors. Used when we want to evaluate motion-only
    performance without Re-ID influence.
    """

    def __init__(self, feature_dim=128):
        self.feature_dim = feature_dim

    def __call__(self, image, boxes):
        return np.zeros((len(boxes), self.feature_dim), dtype=np.float32)


class ResNetExtractor:
    """
    Simple ResNet18-based feature extractor.
    Crops each bounding box, resizes to 128x64, and extracts features
    from the avgpool layer.
    """

    def __init__(self, model_path=None, device=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for ResNetExtractor")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Use ResNet18 with modified final layer as feature extractor
        self.model = resnet18(weights=None)
        self.feature_dim = self.model.fc.in_features  # 512
        self.model.fc = torch.nn.Identity()

        if model_path is not None:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, image, boxes):
        """
        Extract features for each bounding box.

        Parameters
        ----------
        image : ndarray (H, W, 3)
            BGR frame.
        boxes : ndarray (N, 4)
            Bounding boxes in tlwh format.

        Returns
        -------
        features : ndarray (N, feature_dim)
        """
        if len(boxes) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        h, w = image.shape[:2]
        crops = []

        for box in boxes:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w, int(box[0] + box[2]))
            y2 = min(h, int(box[1] + box[3]))

            if x2 <= x1 or y2 <= y1:
                # Degenerate box — use zero patch
                crop = np.zeros((128, 64, 3), dtype=np.uint8)
            else:
                crop = image[y1:y2, x1:x2]

            crops.append(self.transform(crop))

        batch = torch.stack(crops).to(self.device)
        features = self.model(batch).cpu().numpy()

        # L2 normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-12)

        return features


def build_extractor(extractor_type="resnet", model_path=None, device=None):
    """
    Factory for feature extractors.

    Parameters
    ----------
    extractor_type : str
        "resnet", "dummy", or "custom"
    model_path : str | None
    device : str | None

    Returns
    -------
    extractor : callable
    """
    if extractor_type == "dummy":
        return DummyExtractor()
    elif extractor_type == "resnet":
        return ResNetExtractor(model_path=model_path, device=device)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
