import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

import os
import math
from osgeo import gdal
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import pycrs
import fiona
from shapely.geometry import Point, Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import shape
from skimage.util import img_as_float
from skimage.util import img_as_int
import xarray as xr
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import time
import scipy
from skimage import exposure
from skimage.segmentation import quickshift
from skimage.segmentation import slic
from osgeo import ogr
import geopandas as gp
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm

pocetak = time.time()

# PUT DO FILE-A, MERGED, 6 KANALA, OBAVLJENA ATMOSFERSKA KOREKCIJA
merged_raster = "D:/diplomski_rad/slike/rijeka/merg/merged7_cl_rt.tif"

# PUT DO FILE-A, 2 I 4 SU OBLACI I SJENE
masked_raster = "D:/diplomski_rad/slike/rijeka/sirovo/LC08_L1TP_190028_20200808_20200821_01_T1_MTLFmask_co.tif"


# INFO O RASTERU
def getInfo(rasterInfo):
    ds = gdal.Open(rasterInfo)
    print('RASTER INFO:')
    print("Rezolucija:")
    print(ds.RasterXSize, ds.RasterYSize)
    Xvelicina=ds.RasterXSize
    Yvelicina=ds.RasterYSize
    print("Projekcija:")
    print(ds.GetProjection())
    print("Početne koordinate i veličina čelije:")
    print(ds.GetGeoTransform())
    print("Broj kanala:")
    print(ds.RasterCount)

    ds = None
    return Xvelicina, Yvelicina

Xsize, Ysize = getInfo(merged_raster)

# OTVORI RASTER
def openRaster (fn, access=0):
    ds=gdal.Open(fn, access)
    if ds is None:
        print("Error")
    return ds

# UČITAJ BAND KAO ARRAY
def getRasterBand (fn, band=1, access=0):
    ds=openRaster(fn, access)
    band_ = ds.GetRasterBand(band).ReadAsArray()
    ds = None
    return band_


# KREIRAJ RASTER (DRIVER, DATA, NO DATA VALUES)
def createRasterFromTemplate(fn, ds, data, ndv=np.nan, driverFmt="GTiff"):
    driver=gdal.GetDriverByName(driverFmt)
    outds= driver.Create(fn, xsize=ds.RasterXSize,
                         ysize=ds.RasterYSize,
                         bands=1, eType=gdal.GDT_Float32)
    outds.SetGeoTransform(ds.GetGeoTransform())
    outds.SetProjection(ds.GetProjection())
    outds.GetRasterBand(1).SetNoDataValue(ndv)
    outds.GetRasterBand(1).WriteArray(data)
    outds=None
    ds=None

# MNDWI
def mndwi(greenband, swir2band, ndv=np.nan):
    greenband[greenband<0]=np.nan
    greenband[greenband>10000]=np.nan
    swir2band[swir2band<0]=np.nan
    swir2band[swir2band>10000]=np.nan    
    mndwiband = (greenband-swir2band)/(greenband+swir2band)
    mndwiband[np.isnan(mndwiband)]=ndv
    return mndwiband

# NDVI 
def ndvi(nirband, redband, ndv=np.nan):
    nirband[nirband<0]=np.nan
    nirband[nirband>10000]=np.nan
    redband[redband<0]=np.nan
    redband[redband>10000]=np.nan
    ndviband = (nirband-redband)/(nirband+redband)
    ndviband[np.isnan(ndviband)]=ndv
    return ndviband

# NDBI
def ndbi(swir1band, nirband, ndv=np.nan):
    swir1band[swir1band<0]=np.nan
    swir1band[swir1band>10000]=np.nan
    nirband[nirband<0]=np.nan
    nirband[nirband>10000]=np.nan    
    ndbiband = (swir1band-nirband)/(swir1band+nirband)
    ndbiband[np.isnan(ndbiband)]=ndv
    return ndbiband

# BSI
def bsi(blueband, redband, swir1band, nirband, ndv=np.nan):
    blueband[blueband<0]=np.nan
    blueband[blueband>10000]=np.nan
    redband[redband<0]=np.nan
    redband[redband>10000]=np.nan
    swir1band[swir1band<0]=np.nan
    swir1band[swir1band>10000]=np.nan
    nirband[nirband<0]=np.nan
    nirband[nirband>10000]=np.nan
    bsiband = ((redband+swir1band)-(nirband+blueband))/((redband+swir1band)+(nirband+blueband))
    bsiband[np.isnan(bsiband)]=ndv
    return bsiband

# KREIRANJE FOLDERA
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
                print("Error "+directory)

# FOLDER ZA SPEKTRALNE INDEKSE
folder_Si='SpecInd'
createFolder('./'+folder_Si+'/')

mjesto_za_mndwi = "./"+folder_Si+"/mndwi_id.tif"

# MNDWI

greenband=getRasterBand(merged_raster, 2)
swir2band=getRasterBand(merged_raster, 6)
mndwiband=mndwi(greenband, swir2band)
createRasterFromTemplate(mjesto_za_mndwi, gdal.Open(merged_raster), mndwiband)

mjesto_za_ndvi = "./"+folder_Si+"/ndvi_id.tif"

# NDVI

redband=getRasterBand(merged_raster, 3)
nirband=getRasterBand(merged_raster, 4)
ndviband=ndvi(nirband, redband)
createRasterFromTemplate(mjesto_za_ndvi, gdal.Open(merged_raster), ndviband)

mjesto_za_ndbi = "./"+folder_Si+"/ndbi_id.tif"

# NDBI

swir1band=getRasterBand(merged_raster, 5)
nirband=getRasterBand(merged_raster, 4)
ndbiband=mndwi(swir1band, nirband)
createRasterFromTemplate(mjesto_za_ndbi, gdal.Open(merged_raster), ndbiband)

mjesto_za_bsi = "./"+folder_Si+"/bsi_id.tif"

# BSI
blueband=getRasterBand(merged_raster, 1)
redband=getRasterBand(merged_raster, 3)
swir1band=getRasterBand(merged_raster, 5)
nirband=getRasterBand(merged_raster, 4)
bsiband=bsi(blueband, redband, swir1band, nirband)
createRasterFromTemplate(mjesto_za_bsi, gdal.Open(merged_raster), bsiband)


# FOLDER ZA MASKU I MASKIRANJE
folder_Ma='Masking_Rasters'
createFolder('./'+folder_Ma+'/')

mjesto_za_BB = "./"+folder_Ma+"/poligon_BB.shp"

def GetPoligon(ulaz, izlaz, epsg_broj=32633):
    ds = gdal.Open(ulaz)
    geoTransform = ds.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * ds.RasterXSize
    miny = maxy + geoTransform[5] * ds.RasterYSize

    df = gp.GeoDataFrame( [['box', Point(minx, maxy)], 
                           ['box', Point(maxx, maxy)], 
                           ['box', Point(maxx,miny)], 
                           ['box', Point(minx,miny)]],  
                         columns = ['shape_id', 'geometry'], 
                         geometry='geometry')

    df['geometry'] = df['geometry'].apply(lambda x: x.coords[0])

    df = df.groupby('shape_id')['geometry'].apply(lambda x: Polygon(x.tolist())).reset_index()

    df = gp.GeoDataFrame(df, geometry = 'geometry')
    df = df.set_crs(epsg=epsg_broj)
    df.to_file(izlaz)
    ds = None

GetPoligon(mjesto_za_mndwi, mjesto_za_BB)

mjesto_za_ClipMask = "./"+folder_Ma+"/ClippedMask.tif"

def GetClippedRaster(ulaz, izlaz, poligon):
    with fiona.open(poligon, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]    
    
    with rasterio.open(ulaz) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    
    with rasterio.open(izlaz, "w", **out_meta) as dest:
        dest.write(out_image)

GetClippedRaster(masked_raster, mjesto_za_ClipMask, mjesto_za_BB)

def kopirajRaster(ulaz, izlaz):
    ds = gdal.Open(ulaz)
    driver_tiff = gdal.GetDriverByName('GTiff')
    ds_copy = driver_tiff.CreateCopy(izlaz, ds, strict=0)
    ds_copy=None
    ds=None

mjesto_za_mndwi_copy = "./"+folder_Ma+"/mndwi_copy.tif"
mjesto_za_ndvi_copy = "./"+folder_Ma+"/ndvi_copy.tif"
mjesto_za_ndbi_copy = "./"+folder_Ma+"/ndbi_copy.tif"
mjesto_za_bsi_copy = "./"+folder_Ma+"/bsi_copy.tif"
mjesto_za_ClipMask_copy = "./"+folder_Ma+"/ClippedMask_copy.tif"

kopirajRaster(mjesto_za_mndwi, mjesto_za_mndwi_copy)
kopirajRaster(mjesto_za_ndvi, mjesto_za_ndvi_copy)
kopirajRaster(mjesto_za_ndbi, mjesto_za_ndbi_copy)
kopirajRaster(mjesto_za_bsi, mjesto_za_bsi_copy)
kopirajRaster(mjesto_za_ClipMask, mjesto_za_ClipMask_copy)


def dajArray(ulaz):
    ds2 = gdal.Open(ulaz)
    band2 = ds2.GetRasterBand(1).ReadAsArray()
    ds2=None
    return band2.flatten()


prazni_niz = np.full((Xsize, Ysize),-9999,dtype=np.float32).flatten()
maska_niz = dajArray(mjesto_za_ClipMask_copy)

def zamaskiraj(specID, nizmaske, prazniniz, Xd=667, Yd=667):
    dataID = gdal.Open(specID, 1)
    dataID_b = dataID.GetRasterBand(1).ReadAsArray()
    dataID_b = dataID_b.flatten()

    maska=np.logical_or(nizmaske==2,nizmaske==4)
    maskirani=xr.where(maska,prazniniz,dataID_b)

    maskirani = maskirani.reshape(Xd,Yd)

    dataID.GetRasterBand(1).WriteArray(maskirani)

    dataID = None

zamaskiraj(mjesto_za_mndwi_copy, maska_niz, prazni_niz)
zamaskiraj(mjesto_za_ndvi_copy, maska_niz, prazni_niz)
zamaskiraj(mjesto_za_ndbi_copy, maska_niz, prazni_niz)
zamaskiraj(mjesto_za_bsi_copy, maska_niz, prazni_niz)


# FOLDER ZA KLASIFIKACIJU
folder_Cl='Classifying_Rasters'
createFolder('./'+folder_Cl+'/')

VODApath = "./"+folder_Cl+"/VODA.tif"
HVpath = "./"+folder_Cl+"/VISOKAVEG.tif"
LVpath = "./"+folder_Cl+"/NISKAVEG.tif"
BUpath = "./"+folder_Cl+"/IZGR.tif"
BSpath = "./"+folder_Cl+"/GLTL.tif"

def GetWATERThresholdRaster(SIDWATERPath):
    SIDds = gdal.Open(SIDWATERPath)
    bandds = SIDds.GetRasterBand(1).ReadAsArray().flatten()

    bandwaterds = np.where((bandds>0),1,0)

    return bandwaterds.reshape(SIDds.RasterXSize,SIDds.RasterYSize)

waterdata = GetWATERThresholdRaster(mjesto_za_mndwi_copy)

createRasterFromTemplate(VODApath, gdal.Open(mjesto_za_mndwi_copy), waterdata)

def GetVEGThresholdRaster(SIDVEGPath):
    SIDds = gdal.Open(SIDVEGPath)
    bandds = SIDds.GetRasterBand(1).ReadAsArray().flatten()

    bandHVds = np.where((bandds>0.80)&(bandds<0.85),1,0)
    bandLVds = np.where(((bandds>0.65)&(bandds<0.75)),1,0)

    return bandHVds.reshape(SIDds.RasterXSize,SIDds.RasterYSize), bandLVds.reshape(SIDds.RasterXSize,SIDds.RasterYSize)

HVdata, LVdata = GetVEGThresholdRaster(mjesto_za_ndvi_copy)

createRasterFromTemplate(HVpath, gdal.Open(mjesto_za_ndvi_copy), HVdata)
createRasterFromTemplate(LVpath, gdal.Open(mjesto_za_ndvi_copy), LVdata)

def GetBUBSThresholdRaster(SIDBUBSPath):
    SIDds = gdal.Open(SIDBUBSPath)
    bandds = SIDds.GetRasterBand(1).ReadAsArray().flatten()

    bandBUds = np.where((bandds>-0.1)&(bandds<0.0),1,0)
    bandBSds = np.where((bandds>0.1),1,0)

    return bandBUds.reshape(SIDds.RasterXSize,SIDds.RasterYSize), bandBSds.reshape(SIDds.RasterXSize,SIDds.RasterYSize)

BUdata, BSdata = GetBUBSThresholdRaster(mjesto_za_bsi_copy)

createRasterFromTemplate(BUpath, gdal.Open(mjesto_za_bsi_copy), BUdata)
createRasterFromTemplate(BSpath, gdal.Open(mjesto_za_bsi_copy), BSdata)

# FOLDER ZA POLIGONIZACIJU
folder_Pol='Polygon_Rasters'
createFolder('./'+folder_Pol+'/')

VODAPLGNpath = "./"+folder_Pol+"/VODAPLGN.shp"
HVPLGNpath = "./"+folder_Pol+"/VISOKAVEGPLGN.shp"
LVPLGNpath = "./"+folder_Pol+"/NISKAVEGPLGN.shp"
BUPLGNpath = "./"+folder_Pol+"/IZGRPLGN.shp"
BSPLGNpath = "./"+folder_Pol+"/GLTLPLGN.shp"
PTSpath = "./"+folder_Pol+"/GeneratedPts.shp"

def PoligonizirajRaster(RasterZaPoligon, PoligoniziranoPath):
    msk = None
    with rasterio.Env():
        with rasterio.open(RasterZaPoligon) as src:
            image = src.read(1)
            results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) 
            in enumerate(
                shapes(image, mask=msk, transform=src.transform)))

    geoms = list(results)
    gpd_polygonized_raster  = gpd.GeoDataFrame.from_features(geoms)
    gpd_polygonized_raster2 = gpd_polygonized_raster.drop(gpd_polygonized_raster[gpd_polygonized_raster.raster_val == 0.0].index)
    gpd_polygonized_raster2=gpd_polygonized_raster2[gpd_polygonized_raster2.is_valid]
    gpd_polygonized_raster2.set_crs(epsg=32634, inplace=True)
    gpd_polygonized_raster2['area']=gpd_polygonized_raster2['geometry'].area
    gpd_polygonized_raster2=gpd_polygonized_raster2[gpd_polygonized_raster2['area']>901]
    print(gpd_polygonized_raster2.shape)

    gpd_polygonized_raster2.to_file(PoligoniziranoPath, driver='ESRI Shapefile')

PoligonizirajRaster(VODApath, VODAPLGNpath)
PoligonizirajRaster(HVpath, HVPLGNpath)
PoligonizirajRaster(LVpath, LVPLGNpath)
PoligonizirajRaster(BUpath, BUPLGNpath)
PoligonizirajRaster(BSpath, BSPLGNpath)

VODAPLGN2path = "./"+folder_Pol+"/VODAPLGN2.shp"
HVPLGN2path = "./"+folder_Pol+"/VISOKAVEGPLGN2.shp"
LVPLGN2path = "./"+folder_Pol+"/NISKAVEGPLGN2.shp"
BUPLGN2path = "./"+folder_Pol+"/IZGRPLGN2.shp"
BSPLGN2path = "./"+folder_Pol+"/GLTLPLGN2.shp"

def explode(indata, savingdata):
    indf = gpd.GeoDataFrame.from_file(indata)
    outdf = gpd.GeoDataFrame(columns=indf.columns)
    for idx, row in indf.iterrows():
        if type(row.geometry) == Polygon:
            outdf = outdf.append(row,ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gpd.GeoDataFrame(columns=indf.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs,ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom,'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf,ignore_index=True)
    outdf=outdf[outdf.is_valid]
    outdf.set_crs(epsg=32633, inplace=True)
    outdf['area']=outdf['geometry'].area
    outdf=outdf[outdf['area']>901]
    print(outdf.shape)
    outdf.to_file(savingdata, driver='ESRI Shapefile')

##############################################################################

PTSKLASSpath = "./"+folder_Pol+"/PTSklass.shp"

def AddKlass(filex, klassLabel):
    gdf=gpd.read_file(filex)
    gdfgb=gdf.groupby('raster_val', as_index=False).sum()
    totalarea=gdfgb.iloc[0,1]

    l=[]
    klass=klassLabel

    if gdf.shape[0]<150:
        gdf2=gdf.nlargest(gdf.shape[0], ['area']).reset_index(drop=True)
        for idx,row in gdf2.iterrows():
            if sum(l)>totalarea*0.5:
                a=len(l)
                break
            else:
                l.append(gdf2.iloc[idx,1])
    
        gdf3=gdf2.iloc[0:a,:]
        brtocaka=int(150/gdf3.shape[0])
        listatocaka=[]
        
        if brtocaka==1:
            gdf4=gpd.GeoDataFrame(columns=['geometry','klass'])
            gdf4['geometry']=gdf3['geometry'].representative_point().buffer(60)
            gdf4['klass']=klass
            gdf4=gdf4[['geometry','klass']]
            gdf4.set_crs(epsg=32633, inplace=True, allow_override=True)
            return gdf4
        else:
            for i,r in gdf3.iterrows():
                minx, miny, maxx, maxy = r['geometry'].bounds
                b=0
                while b<brtocaka:
                    p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                    if r['geometry'].contains(p):
                       listatocaka.append(p.buffer(60))
                       b+=1
            gdf4=gpd.GeoDataFrame(columns=['geometry','klass'])
            gdf4['geometry']=listatocaka
            gdf4['klass']=klass
            gdf4.set_crs(epsg=32633, inplace=True, allow_override=True)
            return gdf4

    if gdf.shape[0]>150:
        gdf2=gdf.nlargest(150, ['area']).reset_index(drop=True)    
        gdfgb2=gdf2.groupby('raster_val', as_index=False).sum()
        totalarea2=gdfgb2.iloc[0,1]

        if totalarea2/totalarea > 0.5:
            for idx,row in gdf2.iterrows():
                if sum(l)>totalarea*0.5:
                    a=len(l)
                    break
                else:
                    l.append(gdf2.iloc[idx,1])
                    
            gdf3=gdf2.iloc[0:a,:]
            brtocaka=int(150/gdf3.shape[0])
            listatocaka=[]
            if brtocaka==1:
                gdf4=gpd.GeoDataFrame(columns=['geometry','klass'])
                gdf4['geometry']=gdf3['geometry'].representative_point().buffer(60)
                gdf4['klass']=klass
                gdf4=gdf4[['geometry','klass']]
                gdf4.set_crs(epsg=32633, inplace=True, allow_override=True)
                return gdf4

            else:
                for i,r in gdf3.iterrows():
                    minx, miny, maxx, maxy = r['geometry'].bounds
                    b=0
                    while b<brtocaka:
                        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                        if r['geometry'].contains(p):
                           listatocaka.append(p.buffer(60))
                           b+=1
                gdf4=gpd.GeoDataFrame(columns=['geometry','klass'])
                gdf4['geometry']=listatocaka
                gdf4['klass']=klass
                gdf4.set_crs(epsg=32633, inplace=True, allow_override=True)
                return gdf4

        else:
            gdf4=gpd.GeoDataFrame(columns=['geometry','klass'])
            gdf4['geometry']=gdf2['geometry'].representative_point().buffer(60)
            gdf4['klass']=klass
            gdf4=gdf4[['geometry','klass']]
            gdf4.set_crs(epsg=32633, inplace=True, allow_override=True)
            return gdf4

gdftotal=gpd.GeoDataFrame(columns=['geometry','klass'])
    
gg1=AddKlass(VODAPLGNpath, 1)
gg2=AddKlass(BSPLGNpath, 2)
gg3=AddKlass(LVPLGNpath, 3)
gg4=AddKlass(HVPLGNpath, 4)
gg5=AddKlass(BUPLGNpath, 5)

gdftotal=gdftotal.append(gg1)
gdftotal=gdftotal.append(gg2)
gdftotal=gdftotal.append(gg3)
gdftotal=gdftotal.append(gg4)
gdftotal=gdftotal.append(gg5)

print(gdftotal.shape)
gdftotal.set_crs(epsg=32633, inplace=True)
gdftotal.to_file(PTSKLASSpath, driver='ESRI Shapefile')

########################################################################################################

# FILE
RasterFile = "D:/diplomski_rad/slike/rijeka/merg/merged4.tif"
SegmentFile = "segmented3.tif"
Training_file = PTSKLASSpath
Output_klasifikacija = 'classifiedRI3.tif'
Testing_file = "D:/diplomski_rad/slike/rijeka/shps/test_points2.shp"

# DRIVER ZA SPREMANJE OSTALIH TIFOVA
driverTiff = gdal.GetDriverByName('GTiff')

# UCITAVANJE PODATAKA RASTERA
RasterFileDataset = gdal.Open(RasterFile)

# BROJ KANALA
brojKanala = RasterFileDataset.RasterCount

bandData = []

# INFO
print('RASTER INFO:')
print('BANDS: ', RasterFileDataset.RasterCount)
print('ROWS: ', RasterFileDataset.RasterYSize)
print('COLUMNS: ', RasterFileDataset.RasterXSize)
print("Projekcija:")
print(RasterFileDataset.GetProjection())
print("Početne koordinate i veličina čelije:")
print(RasterFileDataset.GetGeoTransform())

def zamaskirajMultiBand(rfds, brk, bd, nizmaske, prazniniz):
    maska=np.logical_or(nizmaske==2,nizmaske==4)
    
    for i in range(1, brk+1):
        kanal_i = rfds.GetRasterBand(i).ReadAsArray().flatten()
        zamaskiranikanal_i = xr.where(maska,prazni_niz,kanal_i)
        bd.append(zamaskiranikanal_i.reshape(667,667))

    return bd

MultiBandDATA = zamaskirajMultiBand(RasterFileDataset, brojKanala, bandData, maska_niz, prazni_niz)

# SPREMANJE SVIH PODATAKA U 1 VARIJABLU, STACKING
MultiBandDATA = np.dstack(MultiBandDATA)
bandDataDD=MultiBandDATA.astype('double')

segments = slic(bandDataDD, n_segments=50000, compactness = 0.1)

def segment_features(segment_pixels):
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:,b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1: # varijanca je NaN > 0
            band_stats[3] = 0.0

        features += band_stats
            
    return features

# DAJ ID SVAKOG SEGMENTA
segment_ids = np.unique(segments)

objects = []
object_ids = []

for id_ in segment_ids:
    segment_pixels = bandDataDD[segments == id_]
    object_features = segment_features(segment_pixels)
    objects.append(object_features)
    object_ids.append(id_)

# SPREMANJE U RASTER
segmentsData = driverTiff.Create(SegmentFile, RasterFileDataset.RasterXSize,
                                 RasterFileDataset.RasterYSize, 1,
                                 gdal.GDT_Float32)
segmentsData.SetGeoTransform(RasterFileDataset.GetGeoTransform())
segmentsData.SetProjection(RasterFileDataset.GetProjectionRef())
segmentsData.GetRasterBand(1).WriteArray(segments)
segmentsData = None

# RASTERIZACIJA TRAIN PODATAKA
trainFile = Training_file
trainDataset = ogr.Open(trainFile)
lyr = trainDataset.GetLayer()
driver = gdal.GetDriverByName('MEM')

targetDataset = driver.Create('', RasterFileDataset.RasterXSize,
                              RasterFileDataset.RasterYSize, 1,
                              gdal.GDT_UInt16)

targetDataset.SetGeoTransform(RasterFileDataset.GetGeoTransform())
targetDataset.SetProjection(RasterFileDataset.GetProjection())
options = ['ATTRIBUTE=klass']
gdal.RasterizeLayer(targetDataset, [1], lyr, options=options)

ground_truth = targetDataset.GetRasterBand(1).ReadAsArray()

classes = np.unique(ground_truth)[1:]

segments_per_class = {}

for c in classes:
    segments_of_class = segments[ground_truth == c]
    segments_per_class[c] = set(segments_of_class)

intersection = set()
accumulation = set()

for class_segments in segments_per_class.values():
    intersection |= accumulation.intersection(class_segments)
    accumulation |= class_segments

#assert len(intersection) == 0, 'Segment(s) represent multiple classes'

train_img = np.copy(segments)
threshold = train_img.max()+1

for k in classes:
    class_label = threshold + k
    for s_id in segments_per_class[k]:
        train_img[train_img == s_id] = class_label

train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold

training_objects = []
training_labels = []

for klass in classes:
    class_train_objects = [v for i,v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
    training_labels += [klass] * len(class_train_objects)
    training_objects += class_train_objects

training_objects = np.array(training_objects)
training_labels = np.array(training_labels)
objects = np.array(objects)

model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(6, activation='softmax'),    
    ])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(training_objects, training_labels, epochs=20, batch_size=32)

pred = model.predict(objects)
predicted = np.argmax(pred, axis=1)

clf = np.copy(segments)
for sid, kl in zip(segment_ids, predicted):
    clf[clf == sid] = kl

mask = np.sum(bandDataDD, axis=2)
mask[mask > 0.0] = 1.0
mask[mask == 0.0] = -1.0

clf = np.multiply(clf, mask)
clf[clf<0] = -9999.0

clfDataset = driverTiff.Create(Output_klasifikacija,
                               RasterFileDataset.RasterXSize,RasterFileDataset.RasterYSize,
                               1, gdal.GDT_Float32)

clfDataset.SetGeoTransform(RasterFileDataset.GetGeoTransform())
clfDataset.SetProjection(RasterFileDataset.GetProjection())
clfDataset.GetRasterBand(1).SetNoDataValue(-9999.0)
clfDataset.GetRasterBand(1).WriteArray(clf)
clfDataset=None

trainFile = Testing_file
trainDataset = ogr.Open(trainFile)
lyr = trainDataset.GetLayer()
driver = gdal.GetDriverByName('MEM')

targetDataset = driver.Create('', RasterFileDataset.RasterXSize,
                              RasterFileDataset.RasterYSize, 1,
                              gdal.GDT_UInt16)

targetDataset.SetGeoTransform(RasterFileDataset.GetGeoTransform())
targetDataset.SetProjection(RasterFileDataset.GetProjection())
options = ['ATTRIBUTE=class']
gdal.RasterizeLayer(targetDataset, [1], lyr, options=options)

data = targetDataset.GetRasterBand(1).ReadAsArray()

predDs = gdal.Open(Output_klasifikacija)
pred=predDs.GetRasterBand(1).ReadAsArray()

idx = np.nonzero(data)

cm = metrics.confusion_matrix(data[idx], pred[idx])

print(cm)

acc = cm.diagonal() / cm.sum(axis=0)
print(acc)

total_acc = metrics.accuracy_score(data[idx], pred[idx])
print(total_acc)

class_report = metrics.classification_report(data[idx], pred[idx])
print(class_report)

print('TIME: ',time.time()-pocetak,' sec ')
