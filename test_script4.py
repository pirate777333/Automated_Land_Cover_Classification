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
from skimage import exposure

##files=['D:/rektorov_rad/Tokyo/desno107/sirovo/LC08_L1TP_107035_20200819_20200823_01_T1_B2.TIF',
##       'D:/rektorov_rad/Tokyo/desno107/sirovo/LC08_L1TP_107035_20200819_20200823_01_T1_B3.TIF',
##       'D:/rektorov_rad/Tokyo/desno107/sirovo/LC08_L1TP_107035_20200819_20200823_01_T1_B4.TIF']
##
file_list=['exposure2.tif','exposure3.tif','exposure4.tif']
##
##driverTiff=gdal.GetDriverByName('GTiff')
##
##b=0
##for file in files:
##    file_ds=gdal.Open(file)
##    band=file_ds.GetRasterBand(1).ReadAsArray().flatten()
##    band=band*1.0
##    img=exposure.rescale_intensity(band)
##    img=img.reshape(file_ds.RasterYSize,file_ds.RasterXSize)
##    out_ds=driverTiff.Create(files_out[b],file_ds.RasterXSize,
##                             file_ds.RasterYSize,1, gdal.GDT_Float32)
##    out_ds.SetGeoTransform(file_ds.GetGeoTransform())
##    out_ds.SetProjection(file_ds.GetProjectionRef())
##    out_ds.GetRasterBand(1).WriteArray(img)
##    out_ds=None
##    b+=1


##file_list = ['D:/rektorov_rad/Tokyo/desno107/atmosfKor/RT_LC08_L1TP_107035_20200819_20200823_01_T1_B2.TIF',
##             'D:/rektorov_rad/Tokyo/desno107/atmosfKor/RT_LC08_L1TP_107035_20200819_20200823_01_T1_B3.TIF',
##             'D:/rektorov_rad/Tokyo/desno107/atmosfKor/RT_LC08_L1TP_107035_20200819_20200823_01_T1_B4.TIF']
##
# Read metadata of first file
with rasterio.open(file_list[0]) as src0:
    meta = src0.meta

#Update meta to reflect the number of layers
meta.update(count = len(file_list))

#Read each layer and write it to stack
with rasterio.open('stacked1.tif', 'w', **meta) as dst:
    for id, layer in enumerate(file_list, start=1):
        with rasterio.open(layer) as src1:
            dst.write_band(id, src1.read(1))



##rastUlaz='D:/rektorov_rad/Tokyo/desno107/merged6/Tokyo6_107.tif'
##rastIzlaz='Tokyo6_107clipped.tif'
##poligon='D:/rektorov_rad/Tokyo/desno107/rolypoly/poligon107.shp'
##
##def GetClippedRaster(ulaz, izlaz, poligon):
##    with fiona.open(poligon, "r") as shapefile:
##        shapes = [feature["geometry"] for feature in shapefile]    
##    
##    with rasterio.open(ulaz) as src:
##        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
##        out_meta = src.meta
##
##    out_meta.update({"driver": "GTiff",
##                     "height": out_image.shape[1],
##                     "width": out_image.shape[2],
##                     "transform": out_transform,
##                     "crs": "+proj=utm +zone=54 +datum=WGS84 +units=m +no_defs"})
##    
##    with rasterio.open(izlaz, "w", **out_meta) as dest:
##        dest.write(out_image)
##
##GetClippedRaster(rastUlaz, rastIzlaz, poligon)


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
