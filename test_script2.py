import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os
from osgeo import gdal
import pycrs
import fiona
from fiona.crs import from_epsg
from shapely.geometry import box
from rasterio.mask import mask

rastUlaz='D:/rektorov_rad/Tokyo/desno107/merged6/Tokyo6_107.tif'
rastIzlaz='Tokyo6_107clipped.tif'
poligon='D:/rektorov_rad/Tokyo/desno107/rolypoly/poligon107.shp'

def GetClippedRaster(ulaz, izlaz, poligon):
    with fiona.open(poligon, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]    
    
    with rasterio.open(ulaz) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform,
                     "crs": "+proj=utm +zone=54 +datum=WGS84 +units=m +no_defs"})
    
    with rasterio.open(izlaz, "w", **out_meta) as dest:
        dest.write(out_image)

GetClippedRaster(rastUlaz, rastIzlaz, poligon)


##files=['D:/rektorov_rad/Tokyo/desno107/merged6/Tokyo6_107.tif',
##       'D:/rektorov_rad/Tokyo/lijevo108/merg6/Tokyo6_108.tif']
##
##out='Tokyo6_1078.tif'
##
##src_files_to_mosaic = []
##
##for file in files:
##    src = rasterio.open(file)
##    src_files_to_mosaic.append(src)
##
##mosaic, out_trans = merge(src_files_to_mosaic)
##
##out_meta = src.meta.copy()
##
##out_meta.update({"driver": "GTiff",
##                "height": mosaic.shape[1],
##                "width": mosaic.shape[2],
##                "transform": out_trans,
##                "crs": "+proj=utm +zone=54 +datum=WGS84 +units=m +no_defs"
##                })
##with rasterio.open(out, "w", **out_meta) as dest:
##                dest.write(mosaic)
