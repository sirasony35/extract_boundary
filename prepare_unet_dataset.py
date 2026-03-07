import os
import glob
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import cv2


def create_unet_dataset(tif_path, shp_path, output_dir, tile_size=512, overlap=50):
    """
    RGB GeoTIFF와 Shapefile을 읽어 U-Net 학습용 (이미지, 흑백 마스크) 쌍을 생성합니다.
    """
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    print(f"  -> [{base_name}] U-Net용 데이터 자르기 시작...")

    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        print(f"  [오류] Shapefile을 읽을 수 없습니다: {e}")
        return

    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        width, height = src.width, src.height
        stride = tile_size - overlap
        tile_count = 0

        for y in range(0, height, stride):
            for x in range(0, width, stride):
                window = Window(x, y, tile_size, tile_size)
                transform = src.window_transform(window)

                minx, miny = transform * (0, tile_size)
                maxx, maxy = transform * (tile_size, 0)
                tile_box = box(minx, miny, maxx, maxy)

                intersected_gdf = gdf[gdf.intersects(tile_box)]

                # 배경만 있는 타일도 조금은 학습하는 것이 좋지만, 너무 많으면 비효율적이므로
                # 필지가 조금이라도 포함된(intersect) 영역만 잘라냅니다.
                if intersected_gdf.empty:
                    continue

                # 1. RGB 이미지 데이터 추출
                img_data = src.read((1, 2, 3), window=window)
                if img_data.shape != (3, tile_size, tile_size) or np.all(img_data == 0):
                    continue

                img_cv2 = np.moveaxis(img_data, 0, -1)
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

                # 2. U-Net용 흑백 정답지(Mask) 이미지 생성 (여기가 핵심입니다!)
                # Shapefile의 폴리곤을 이미지의 픽셀(흰색: 255, 검은색: 0)로 변환(Rasterize)합니다.
                shapes = ((geom, 255) for geom in intersected_gdf.geometry)

                try:
                    mask_img = rasterize(
                        shapes=shapes,
                        out_shape=(tile_size, tile_size),
                        transform=transform,
                        fill=0,  # 배경은 0 (검은색)
                        dtype='uint8'
                    )
                except ValueError:
                    continue  # 유효한 폴리곤이 잘리지 않은 경우 패스

                # 결과물 저장
                tile_name = f"{base_name}_x{x}_y{y}"
                img_save_path = os.path.join(output_dir, "images", f"{tile_name}.jpg")
                mask_save_path = os.path.join(output_dir, "masks", f"{tile_name}.png")  # 마스크는 손실 없는 png 권장

                cv2.imwrite(img_save_path, img_cv2)
                cv2.imwrite(mask_save_path, mask_img)

                tile_count += 1

    print(f"  -> [{base_name}] 완료! {tile_count}개의 (이미지+마스크) 생성됨.")


def process_all_folders(tif_folder, shp_folder, output_dir):
    # U-Net용 폴더 구조 생성
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    tif_files = []
    for ext in ('*.tif', '*.TIF'):
        tif_files.extend(glob.glob(os.path.join(tif_folder, ext)))

    print(f"총 {len(tif_files)}개의 TIF 파일을 찾았습니다. 자동 매칭을 시작합니다...\n")

    for tif_path in tif_files:
        filename = os.path.basename(tif_path)

        # [중요] 식생지수(GNDVI) 파일은 RGB가 아니므로 스킵합니다.
        if 'GNDVI' in filename.upper():
            print(f"  [스킵] {filename} (GNDVI 데이터 제외)")
            continue

        base_name = os.path.splitext(filename)[0]
        field_code = base_name.split('_')[0]

        possible_shp_names = [
            f"{field_code}.shp",
            f"{field_code}_boundary.shp",
            f"{field_code}_Boundary.shp",
            f"{field_code}_BOUNDARY.shp"
        ]

        matched_shp_path = None
        for shp_name in possible_shp_names:
            temp_path = os.path.join(shp_folder, shp_name)
            if os.path.exists(temp_path):
                matched_shp_path = temp_path
                break

        if matched_shp_path:
            print(f"✅ 매칭 성공: {filename} ↔ {os.path.basename(matched_shp_path)}")
            create_unet_dataset(tif_path, matched_shp_path, output_dir)
        else:
            print(f"❌ 매칭 실패: '{field_code}' 필지의 Shapefile을 찾을 수 없습니다.")

    print("\n🎉 모든 데이터 전처리가 완료되었습니다!")


if __name__ == "__main__":
    IMAGES_FOLDER = "raw_data/images"
    SHAPES_FOLDER = "raw_data/shapes"
    OUTPUT_FOLDER = "unet_dataset"  # 결과가 저장될 폴더

    process_all_folders(IMAGES_FOLDER, SHAPES_FOLDER, OUTPUT_FOLDER)