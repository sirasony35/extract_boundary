import os
import glob
import cv2
import torch
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
from scipy.ndimage import binary_closing, binary_fill_holes
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import warnings

warnings.filterwarnings("ignore")


def load_unet_model(weights_path, device):
    print(f"🔄 AI 모델을 불러옵니다: {weights_path}")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    if not os.path.exists(weights_path):
        print(f"❌ [에러] 모델 가중치 파일이 없습니다: {weights_path}")
        return None

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def extract_and_save_boundary_sliding(tif_path, model, device, output_folder, tile_size=512):
    """
    [핵심 수정] 거대한 원본 해상도를 유지한 채 512x512 조각으로 나누어 예측하고 병합합니다.
    """
    filename = os.path.basename(tif_path)
    base_name = os.path.splitext(filename)[0]
    field_code = base_name.split('_')[0]

    output_shp_name = f"{field_code}_boundary.shp"
    output_shp_path = os.path.join(output_folder, output_shp_name)

    print(f"  -> [{field_code}] 정밀 슬라이딩 윈도우 분석 중... ({filename})")

    try:
        with rasterio.open(tif_path) as src:
            band_count = src.count
            if band_count >= 3:
                img_array = src.read((1, 2, 3))
            else:
                img_array = src.read(1)
                img_array = np.stack((img_array,) * 3, axis=0)

            transform = src.transform
            crs = src.crs

            img_cv2 = np.moveaxis(img_array, 0, -1)
            orig_h, orig_w = img_cv2.shape[:2]

            # 1. 원본 크기 그대로 빈 캔버스(마스크) 생성
            mask_full = np.zeros((orig_h, orig_w), dtype=np.float32)

            preprocess = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

            # 2. 이미지를 tile_size(512) 단위로 잘라서 예측
            # 이미지 가장자리가 잘리지 않도록 여백(Padding) 계산
            pad_h = (tile_size - orig_h % tile_size) % tile_size
            pad_w = (tile_size - orig_w % tile_size) % tile_size
            img_padded = np.pad(img_cv2, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            # 바둑판처럼 이동하며 예측 (Sliding Window)
            with torch.no_grad():
                for y in range(0, orig_h, tile_size):
                    for x in range(0, orig_w, tile_size):
                        # 512x512 조각 잘라내기
                        tile = img_padded[y:y + tile_size, x:x + tile_size]
                        tensor_img = preprocess(image=tile)['image'].unsqueeze(0).to(device)

                        # 조각 예측
                        output = model(tensor_img)
                        prob = torch.sigmoid(output).squeeze().cpu().numpy()

                        # 원본 사이즈에 맞게 조각을 캔버스에 붙여넣기
                        valid_h = min(tile_size, orig_h - y)
                        valid_w = min(tile_size, orig_w - x)
                        mask_full[y:y + valid_h, x:x + valid_w] = prob[:valid_h, :valid_w]

            # 3. 0.5 기준으로 이진화 (0 또는 1)
            binary_mask = (mask_full > 0.5).astype(np.uint8)

            # 4. 자잘한 노이즈 제거 및 구멍 메우기
            binary_mask = binary_closing(binary_mask, structure=np.ones((15, 15)))  # 구조를 좀 더 크게 잡음
            binary_mask = binary_fill_holes(binary_mask).astype(np.uint8)

            # 5. 윤곽선 추출 (가장 큰 덩어리 위주로 추출)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("    [실패] AI가 필지를 찾지 못했습니다.")
                return

            # 면적이 너무 작은 노이즈 덩어리는 무시하고, 상위의 큰 덩어리들을 그립니다.
            final_mask = np.zeros_like(binary_mask)
            min_area = (orig_h * orig_w) * 0.05  # 전체 이미지의 5% 이상되는 덩어리만 취급

            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

            if not valid_contours:
                # 기준치를 넘는 게 없다면 가장 큰 거 하나만 그림
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(final_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
            else:
                cv2.drawContours(final_mask, valid_contours, -1, 1, thickness=cv2.FILLED)

            # 6. 폴리곤 변환 및 Shapefile 저장
            polygons = []
            shapes_gen = shapes(final_mask, mask=(final_mask == 1), transform=transform)
            for geom, val in shapes_gen:
                if val == 1:
                    polygons.append(shape(geom))

            if not polygons:
                return

            gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)

            # 농경지 특성상 테두리를 조금 더 반듯하게 단순화
            gdf['geometry'] = gdf.geometry.simplify(0.5)

            gdf.to_file(output_shp_path, encoding='euc-kr')
            print(f"    ✅ 성공! 저장 완료: {output_shp_name}")

    except Exception as e:
        print(f"    [에러] {filename} 처리 중 문제 발생: {e}")


def main():
    print("🚀 [사전 작업] 정밀 바운더리 자동 추출(Sliding Window)을 시작합니다.")

    INPUT_TIFF_FOLDER = "new_data"
    OUTPUT_SHP_FOLDER = "result/ShapeFile"
    MODEL_WEIGHTS = "weights/best_unet_field.pth"

    os.makedirs(OUTPUT_SHP_FOLDER, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_unet_model(MODEL_WEIGHTS, device)
    if model is None: return

    tif_files = list(set(glob.glob(os.path.join(INPUT_TIFF_FOLDER, '*.[tT][iI][fF]'))))

    if not tif_files:
        print(f"폴더에 처리할 TIF 이미지가 없습니다.")
        return

    for tif_path in tif_files:
        if 'GNDVI' in os.path.basename(tif_path).upper():
            continue

        extract_and_save_boundary_sliding(tif_path, model, device, OUTPUT_SHP_FOLDER)

    print("\n🎉 모든 바운더리 추출 작업이 완료되었습니다!")


if __name__ == "__main__":
    main()