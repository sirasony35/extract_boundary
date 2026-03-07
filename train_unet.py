import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


# 1. 커스텀 데이터셋 클래스 (데이터를 불러와 AI에게 먹여주는 역할)
class FieldDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        self.masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 이미지와 마스크 읽기
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        # 정규화 (0~255 값을 0.0~1.0 사이로 변환)
        mask = (mask / 255.0).astype(np.float32)
        # 차원 추가 (H, W) -> (H, W, 1)
        mask = np.expand_dims(mask, axis=-1)

        # Albumentations를 이용한 데이터 증강 및 텐서 변환 적용
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # PyTorch의 채널 순서 맞춤 처리: 이미지(C,H,W), 마스크(1,H,W)
        mask = mask.permute(2, 0, 1)

        return image, mask


# 2. 메인 학습 함수
def train_model():
    print("🚀 U-Net AI 학습을 시작합니다! (RTX 3060 가동 준비)")

    # 하이퍼파라미터 설정
    DATA_DIR = "unet_dataset"
    BATCH_SIZE = 8  # RTX 3060(12GB)에 무리가 안 가는 적정 사이즈
    EPOCHS = 30  # 전체 데이터를 30번 반복 학습
    LEARNING_RATE = 0.0001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ 사용 장치: {DEVICE}")

    # 데이터 전처리 파이프라인 (이미지를 PyTorch가 좋아하는 형식으로 변환)
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 데이터 로더 설정
    train_dataset = FieldDataset(
        images_dir=os.path.join(DATA_DIR, "images"),
        masks_dir=os.path.join(DATA_DIR, "masks"),
        transform=transform
    )

    if len(train_dataset) == 0:
        print("❌ 학습할 데이터가 없습니다! 전처리가 잘 되었는지 확인해 주세요.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 3. 모델 정의 (ResNet34 기반의 U-Net)
    # imagenet으로 미리 똑똑해진 가중치를 가져와서 필지 찾기에만 집중시킵니다.
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,  # RGB 이미지 입력
        classes=1,  # 필지(1) vs 배경(0)
        activation=None
    ).to(DEVICE)

    # 오차 계산 및 최적화 도구 설정
    criterion = torch.nn.BCEWithLogitsLoss()  # 픽셀 단위로 0인지 1인지 맞추는 데 특화된 손실함수
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 학습 루프 진행
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # [핵심] tqdm을 씌워서 진행률 바를 만듭니다!
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]")

        for images, masks in pbar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # 진행률 바 옆에 실시간으로 현재 오차(loss)를 보여줍니다.
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"✅ Epoch [{epoch + 1}/{EPOCHS}] 완료 - 평균 오차: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/best_unet_field.pth")
            print("   ⭐ 최고 성능 모델 저장됨!")

    print("\n🎉 모든 학습이 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    train_model()