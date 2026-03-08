import os
import glob
import cv2
import torch
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
from scipy.ndimage import binary_fill_holes
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
        print(f"❌ [에러] 모델 가중치 파일이 없습니다.")
        return None

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def extract_and_save_boundary_sliding(tif_path, model, device, output_folder, tile_size=512):
    filename = os.path.basename(tif_path)
    base_name = os.path.splitext(filename)[0]
    field_code = base_name.split('_')[0]

    output_shp_name = f"{field_code}_boundary.shp"
    output_shp_path = os.path.join(output_folder, output_shp_name)

    print(f"  -> [{field_code}] 정밀 분석 및 외곽선 보정 중... ({filename})")

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

            mask_full = np.zeros((orig_h, orig_w), dtype=np.float32)

            preprocess = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

            pad_h = (tile_size - orig_h % tile_size) % tile_size
            pad_w = (tile_size - orig_w % tile_size) % tile_size
            img_padded = np.pad(img_cv2, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            with torch.no_grad():
                for y in range(0, orig_h, tile_size):
                    for x in range(0, orig_w, tile_size):
                        tile = img_padded[y:y + tile_size, x:x + tile_size]
                        tensor_img = preprocess(image=tile)['image'].unsqueeze(0).to(device)

                        output = model(tensor_img)
                        prob = torch.sigmoid(output).squeeze().cpu().numpy()

                        valid_h = min(tile_size, orig_h - y)
                        valid_w = min(tile_size, orig_w - x)
                        mask_full[y:y + valid_h, x:x + valid_w] = prob[:valid_h, :valid_w]

            # [수정 1] AI의 확신 기준을 낮춰서 가장자리 흙 부분도 최대한 포함시킵니다 (0.5 -> 0.3)
            binary_mask = (mask_full > 0.3).astype(np.uint8)

            # [수정 2] 강력한 형태학적 보정 (Morphology)
            # 1. 팽창(Dilation): 영역을 바깥으로 강제로 밀어내어 비어있는 테두리를 덮어버립니다.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=2)

            # 2. 닫힘(Closing) 및 구멍 메우기: 내부의 파인 곳을 꽉 채웁니다.
            closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)
            final_binary_mask = binary_fill_holes(closed_mask).astype(np.uint8)

            # 윤곽선 추출
            contours, _ = cv2.findContours(final_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("    [실패] AI가 필지를 찾지 못했습니다.")
                return

            final_mask = np.zeros_like(final_binary_mask)
            min_area = (orig_h * orig_w) * 0.02  # 2% 이상 크기만 인정
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

            if not valid_contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(final_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
            else:
                cv2.drawContours(final_mask, valid_contours, -1, 1, thickness=cv2.FILLED)

            # [수정 3] 폴리곤 직선화 (다각형 단순화)
            polygons = []
            shapes_gen = shapes(final_mask, mask=(final_mask == 1), transform=transform)
            for geom, val in shapes_gen:
                if val == 1:
                    polygons.append(shape(geom))

            if not polygons:
                return

            gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)

            # 톱니바퀴 같은 선을 반듯하게 펴줍니다. (수치를 올릴수록 더 각진 다각형이 됩니다)
            # 만약 좌표계가 WGS84(위경도)라면 0.0001, 미터 단위(UTM 등)라면 2~5 정도가 적당합니다.
            # geopandas의 simplify는 알아서 단위를 따라갑니다.

            # 우선 상대적으로 강하게(tolerance=2.0) 펴주도록 설정합니다. (좌표계에 따라 조절 필요)
            try:
                gdf['geometry'] = gdf.geometry.simplify(tolerance=2.0, preserve_topology=True)
            except:
                gdf['geometry'] = gdf.geometry.simplify(0.00005, preserve_topology=True)  # 위경도일 경우 대비

            gdf.to_file(output_shp_path, encoding='euc-kr')
            print(f"    ✅ 성공! 가장자리 팽창 및 직선화 완료: {output_shp_name}")

    except Exception as e:
        print(f"    [에러] {filename} 처리 중 문제 발생: {e}")


def main():
    print("🚀 [사전 작업] 정밀 바운더리 자동 추출(팽창 및 직선화 적용)을 시작합니다.")
    INPUT_TIFF_FOLDER = "new_data"
    OUTPUT_SHP_FOLDER = "result/ShapeFile"
    MODEL_WEIGHTS = "weights/best_unet_field.pth"

    os.makedirs(OUTPUT_SHP_FOLDER, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_unet_model(MODEL_WEIGHTS, device)
    if model is None: return

    tif_files = list(set(glob.glob(os.path.join(INPUT_TIFF_FOLDER, '*.[tT][iI][fF]'))))
    for tif_path in tif_files:
        if 'GNDVI' in os.path.basename(tif_path).upper():
            continue
        extract_and_save_boundary_sliding(tif_path, model, device, OUTPUT_SHP_FOLDER)
    print("\n🎉 모든 바운더리 추출 작업이 완료되었습니다!")


if __name__ == "__main__":
    main()