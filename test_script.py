import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os

files=['D:/rektorov_rad/Tokyo/desno107/merged6/Tokyo6_107.tif',
       'D:/rektorov_rad/Tokyo/lijevo108/merg6/Tokyo6_108.tif']

out='Tokyo6_1078.tif'

src_files_to_mosaic = []

for file in files:
    src = rasterio.open(file)
    src_files_to_mosaic.append(src)

mosaic, out_trans = merge(src_files_to_mosaic)

out_meta = src.meta.copy()

out_meta.update({"driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": "+proj=utm +zone=54 +datum=WGS84 +units=m +no_defs"
                })
with rasterio.open(out, "w", **out_meta) as dest:
                dest.write(mosaic)
