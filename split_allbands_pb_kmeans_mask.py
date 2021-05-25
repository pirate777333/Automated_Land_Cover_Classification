import os
import math
from osgeo import gdal
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import pycrs
import fiona
import geopandas as gp
from shapely.geometry import Point, Polygon
from skimage.util import img_as_float
from skimage.util import img_as_int
import xarray as xr
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

pocetak = time.time()

# PUT DO FILE-A, MERGED, 6 KANALA, OBAVLJENA ATMOSFERSKA KOREKCIJA
merged_raster = "D:/rektorov_rad/mergano/merged_2to7.tif"

# BEZ MASKIRANJA

# PUT DO FILE-A, 2 I 4 SU OBLACI I SJENE
masked_raster = "D:/rektorov_rad/sirovo/LC08_L1TP_189030_20200630_20200708_01_T1_MTLFmask.dat"

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
    bk = ds.RasterCount

    ds = None
    return Xvelicina, Yvelicina, bk

Xsize, Ysize, BrKanala = getInfo(merged_raster)

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

# NDVI (FILTRIRANJE VRIJEDNOSTI)
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

def zamaskiraj(specID, nizmaske, prazniniz, Xd=Xsize, Yd=Ysize):
    dataID = gdal.Open(specID, 1)
    dataID_b = dataID.GetRasterBand(1).ReadAsArray()
    dataID_b = dataID_b.flatten()

    maska=np.logical_or(nizmaske==2,nizmaske==4)
    maskirani=xr.where(maska,prazniniz,dataID_b)

    maskirani = maskirani.reshape(Yd,Xd)

    dataID.GetRasterBand(1).WriteArray(maskirani)

    dataID = None

zamaskiraj(mjesto_za_mndwi_copy, maska_niz, prazni_niz)
zamaskiraj(mjesto_za_ndvi_copy, maska_niz, prazni_niz)
zamaskiraj(mjesto_za_ndbi_copy, maska_niz, prazni_niz)
zamaskiraj(mjesto_za_bsi_copy, maska_niz, prazni_niz)

def zamaskirajMultiBand(multiband, nizmaske, prazniniz):
    otvori = gdal.Open(multiband)
    brkanala = otvori.RasterCount

    maska=np.logical_or(nizmaske==2,nizmaske==4)

    data = np.empty((otvori.RasterXSize*otvori.RasterYSize, brkanala+1))

    for i in range(1, brkanala+1):
        kanal_i = otvori.GetRasterBand(i).ReadAsArray().flatten()
        zamaskiranikanal_i = xr.where(maska,prazni_niz,kanal_i)
        data[:, i-1] = zamaskiranikanal_i

    return data

MultiBandDATA = zamaskirajMultiBand(merged_raster, maska_niz, prazni_niz)

##def DajMultibandData(multiband):
##    otvori = gdal.Open(multiband)
##    brkanala = otvori.RasterCount
##
##    data = np.empty((otvori.RasterXSize*otvori.RasterYSize, brkanala+1))
##
##    for i in range(1, brkanala+1):
##        kanal_i = otvori.GetRasterBand(i).ReadAsArray().flatten()
##        data[:, i-1] = kanal_i
##
##    return data
##
##MultiBandDATA = DajMultibandData(merged_raster)

folder_Cl='Classifying_Rasters'
createFolder('./'+folder_Cl+'/')

mjesto_za_KlasifikacijuMNDWI = "./"+folder_Cl+"/MNDWIklasificirano1.tif"
mjesto_za_KlasifikacijuNDVI = "./"+folder_Cl+"/NDVIklasificirano1.tif"
mjesto_za_KlasifikacijuNDBI = "./"+folder_Cl+"/NDBIklasificirano1.tif"
mjesto_za_KlasifikacijuBSI = "./"+folder_Cl+"/BSIklasificirano1.tif"

def KMeansKlasifikacija(MultiBandNiz, SIDXRasterUlaz, KLSRasterIzlaz, BrojClustera):
    driverTiff = gdal.GetDriverByName('GTiff')
    SIDXraster = gdal.Open(SIDXRasterUlaz)
    SIDXrasterNiz = SIDXraster.GetRasterBand(1).ReadAsArray().flatten()

    MultiBandNiz[:,-1] = SIDXrasterNiz

    km = KMeans(n_clusters=BrojClustera, random_state=42)
    km.fit(MultiBandNiz)
    km.predict(MultiBandNiz)
    izlaz = km.labels_.reshape((SIDXraster.RasterYSize, SIDXraster.RasterXSize))
    result = driverTiff.Create(KLSRasterIzlaz,
                               SIDXraster.RasterXSize,
                               SIDXraster.RasterYSize, 1,
                               gdal.GDT_Float32)

    result.SetGeoTransform(SIDXraster.GetGeoTransform())
    result.SetProjection(SIDXraster.GetProjection())
    result.GetRasterBand(1).SetNoDataValue(-9999.0)
    result.GetRasterBand(1).WriteArray(izlaz)
    result = None
    SIDXraster = None

def groupby(a, b):
    sidx = b.argsort(kind='mergesort')
    a_sorted = a[sidx]
    b_sorted = b[sidx]

    cut_idx = np.flatnonzero(np.r_[True,b_sorted[1:] != b_sorted[:-1],True])

    out = [np.mean(a_sorted[i:j]) for i,j in zip(cut_idx[:-1],cut_idx[1:])]
    return out # grupira ih i izbaci mean, grupira po b, izbaci mean za a, grupe odgovaraju indeksu (indeks==klasa)

KMeansKlasifikacija(MultiBandDATA, mjesto_za_mndwi_copy, mjesto_za_KlasifikacijuMNDWI, 4)

def RasporediKlasificirano(MultiBandNiz2, mndwi_original, mndwi_klasificirano, ndvi_original, ndbi_original, bsi_original, brojkanala=BrKanala, Xd=Xsize, Yd=Ysize):
    
    dataset_mndwi_original = gdal.Open(mndwi_original)
    dataset_mndwi_klasificirano = gdal.Open(mndwi_klasificirano)
    dataset_ndvi_original = gdal.Open(ndvi_original, 1)    
    dataset_ndbi_original = gdal.Open(ndbi_original, 1)
    dataset_bsi_original = gdal.Open(bsi_original, 1)

    band_dataset_mndwi_original = dataset_mndwi_original.GetRasterBand(1).ReadAsArray().flatten()
    band_dataset_mndwi_klasificirano = dataset_mndwi_klasificirano.GetRasterBand(1).ReadAsArray().flatten()
    band_dataset_ndvi_original = dataset_ndvi_original.GetRasterBand(1).ReadAsArray().flatten()    
    band_dataset_ndbi_original = dataset_ndbi_original.GetRasterBand(1).ReadAsArray().flatten()
    band_dataset_bsi_original = dataset_bsi_original.GetRasterBand(1).ReadAsArray().flatten()

    meduniz = groupby(band_dataset_mndwi_original, band_dataset_mndwi_klasificirano)
    print(meduniz)
    g=np.max(meduniz)
    if g>0:
        for i,j in enumerate(meduniz):
            if j == g:
                f=i    

        vodena_klasa = np.where((band_dataset_mndwi_klasificirano==f),1,-9999)
        ndvi_preclassified = np.where((band_dataset_mndwi_klasificirano==f),-9999,band_dataset_ndvi_original)    
        ndbi_preclassified = np.where((band_dataset_mndwi_klasificirano==f),-9999,band_dataset_ndbi_original)
        bsi_preclassified = np.where((band_dataset_mndwi_klasificirano==f),-9999,band_dataset_bsi_original)

        MultiBandNiz3=np.empty((Xd*Yd, brojkanala+1))
        for i in range(1, brojkanala+1):
            MultiBandNiz3[:,i-1] = np.where((band_dataset_mndwi_klasificirano==f),-9999, MultiBandNiz2[:,i-1])

        ndvi_preclassified_reshaped = ndvi_preclassified.reshape(Yd,Xd)
        dataset_ndvi_original.GetRasterBand(1).WriteArray(ndvi_preclassified_reshaped)
        ndbi_preclassified_reshaped = ndbi_preclassified.reshape(Yd,Xd)
        dataset_ndbi_original.GetRasterBand(1).WriteArray(ndbi_preclassified_reshaped)
        bsi_preclassified_reshaped = bsi_preclassified.reshape(Yd,Xd)
        dataset_bsi_original.GetRasterBand(1).WriteArray(bsi_preclassified_reshaped)

    else:
        vodena_klasa = np.where((band_dataset_mndwi_klasificirano==f),-9999,-9999)
        MultiBandNiz3=MultiBandNiz2

    dataset_mndwi_original = None
    dataset_mndwi_klasificirano = None
    dataset_ndvi_original = None
    dataset_ndbi_original = None
    dataset_bsi_original = None

    return vodena_klasa, MultiBandNiz3

voda, MultiBandDATA2 = RasporediKlasificirano(MultiBandDATA, mjesto_za_mndwi_copy,
                                              mjesto_za_KlasifikacijuMNDWI,
                                              mjesto_za_ndvi_copy,
                                              mjesto_za_ndbi_copy,
                                              mjesto_za_bsi_copy)

KMeansKlasifikacija(MultiBandDATA2, mjesto_za_ndvi_copy, mjesto_za_KlasifikacijuNDVI, 6)


def RasporediKlasificirano2(MultiBandNiz4, ndvi_original, ndvi_klasificirano, ndbi_original, bsi_original, brojkanala=BrKanala, Xd=Xsize, Yd=Ysize):

    dataset_ndvi_original1 = gdal.Open(ndvi_original)
    dataset_ndvi_klasificirano1 = gdal.Open(ndvi_klasificirano)
    dataset_ndbi_original1 = gdal.Open(ndbi_original, 1)    
    dataset_bsi_original1 = gdal.Open(bsi_original, 1)

    band_dataset_ndvi_original1 = dataset_ndvi_original1.GetRasterBand(1).ReadAsArray().flatten()
    band_dataset_ndvi_klasificirano1 = dataset_ndvi_klasificirano1.GetRasterBand(1).ReadAsArray().flatten()
    band_dataset_ndbi_original1 = dataset_ndbi_original1.GetRasterBand(1).ReadAsArray().flatten()
    band_dataset_bsi_original1 = dataset_bsi_original1.GetRasterBand(1).ReadAsArray().flatten()

    meduniz2 = groupby(band_dataset_ndvi_original1, band_dataset_ndvi_klasificirano1)
    print(meduniz2)
    
    mx=max(meduniz2[0],meduniz2[1]) 
    secondmax=min(meduniz2[0],meduniz2[1]) 
    n =len(meduniz2)
    for i in range(2,n): 
        if meduniz2[i]>mx: 
            secondmax=mx
            mx=meduniz2[i] 
        elif meduniz2[i]>secondmax and mx != meduniz2[i]: 
            secondmax=meduniz2[i]

    for i,j in enumerate(meduniz2):
        if j==mx:
            visoka1=i
        if j==secondmax:
            niska1=i    
        
    visoka_klasa = np.where((band_dataset_ndvi_klasificirano1==visoka1),4,-9999)
    niska_klasa = np.where((band_dataset_ndvi_klasificirano1==niska1),3,-9999)
    ndbi_preclassified = np.where((band_dataset_ndvi_klasificirano1==visoka1)|(band_dataset_ndvi_klasificirano1==niska1),-9999,band_dataset_ndbi_original1)
    bsi_preclassified = np.where((band_dataset_ndvi_klasificirano1==visoka1)|(band_dataset_ndvi_klasificirano1==niska1),-9999,band_dataset_bsi_original1)

    MultiBandNiz5=np.empty((Xd*Yd, brojkanala+1))
    for i in range(1, brojkanala+1):
        MultiBandNiz5[:,i-1] = np.where((band_dataset_ndvi_klasificirano1==visoka1)|(band_dataset_ndvi_klasificirano1==niska1),-9999, MultiBandNiz4[:,i-1])

    ndbi_preclassified_reshaped = ndbi_preclassified.reshape(Yd,Xd)
    dataset_ndbi_original1.GetRasterBand(1).WriteArray(ndbi_preclassified_reshaped)
    bsi_preclassified_reshaped = bsi_preclassified.reshape(Yd,Xd)
    dataset_bsi_original1.GetRasterBand(1).WriteArray(bsi_preclassified_reshaped)

    dataset_ndvi_original = None
    dataset_ndvi_klasificirano = None
    dataset_ndbi_original = None
    dataset_bsi_original = None

    return visoka_klasa, niska_klasa, MultiBandNiz5

visokav, niskav, MultiBandDATA3 = RasporediKlasificirano2(MultiBandDATA2,
                                                          mjesto_za_ndvi_copy,
                                                          mjesto_za_KlasifikacijuNDVI,
                                                          mjesto_za_ndbi_copy,
                                                          mjesto_za_bsi_copy)

KMeansKlasifikacija(MultiBandDATA3, mjesto_za_ndbi_copy, mjesto_za_KlasifikacijuNDBI, 6)

def Rasporediklasificirano3(MultiBandNiz6, ndbi_original, ndbi_klasificirano, bsi_original, brojkanala=BrKanala, Xd=Xsize, Yd=Ysize):

    dataset_ndbi_original = gdal.Open(ndbi_original)
    dataset_ndbi_klasificirano1 = gdal.Open(ndbi_klasificirano)
    dataset_bsi_original = gdal.Open(bsi_original, 1)

    band_dataset_ndbi_original = dataset_ndbi_original.GetRasterBand(1).ReadAsArray().flatten()
    band_dataset_ndbi_klasificirano1 = dataset_ndbi_klasificirano1.GetRasterBand(1).ReadAsArray().flatten()
    band_dataset_bsi_original = dataset_bsi_original.GetRasterBand(1).ReadAsArray().flatten()

    meduniz3 = groupby(band_dataset_ndbi_original, band_dataset_ndbi_klasificirano1)

    gp=np.max(meduniz3)
    print(meduniz3)

    for i,j in enumerate(meduniz3):
        if j == gp:
            fp=i
    
    gola_klasa = np.where((band_dataset_ndbi_klasificirano1==fp),2,-9999)
    
    bsi_preclassified = np.where((band_dataset_ndbi_klasificirano1==fp),-9999,band_dataset_bsi_original)

    MultiBandNiz7=np.empty((Xd*Yd, brojkanala+1))
    for i in range(1, brojkanala+1):
        MultiBandNiz7[:,i-1] = np.where((band_dataset_ndbi_klasificirano1==fp),-9999, MultiBandNiz6[:,i-1])

    bsi_preclassified_reshaped = bsi_preclassified.reshape(Yd,Xd)
    dataset_bsi_original.GetRasterBand(1).WriteArray(bsi_preclassified_reshaped)

    dataset_ndbi_original = None
    dataset_ndbi_klasificirano = None
    dataset_bsi_original = None

    return gola_klasa, MultiBandNiz7

gola1, MultiBandDATA4 = Rasporediklasificirano3(MultiBandDATA3, mjesto_za_ndbi_copy,
                                                mjesto_za_KlasifikacijuNDBI,
                                                mjesto_za_bsi_copy)

KMeansKlasifikacija(MultiBandDATA4, mjesto_za_bsi_copy, mjesto_za_KlasifikacijuBSI, 5)

def Rasporediklasificirano4(bsi_original, bsi_klasificirano, brojkanala=BrKanala, Xd=Xsize, Yd=Ysize):

    dataset_bsi_klasificirano = gdal.Open(bsi_klasificirano)
    dataset_bsi_original = gdal.Open(bsi_original)#, 1)

    band_dataset_bsi_klasificirano = dataset_bsi_klasificirano.GetRasterBand(1).ReadAsArray().flatten()
    band_dataset_bsi_original = dataset_bsi_original.GetRasterBand(1).ReadAsArray().flatten()

    meduniz4 = groupby(band_dataset_bsi_original, band_dataset_bsi_klasificirano)
    print(meduniz4)
    
    najveca = np.max(meduniz4)
    najmanja = np.min(meduniz4)


    for i,j in enumerate(meduniz4):
        if j == najveca:
            f=i
        if j == najmanja:
            h=i
    k=[]
    for i,j in enumerate(meduniz4):
        if j != najveca and j != najmanja:
            k.append(i)

    izgradena_klasa2 = np.where((band_dataset_bsi_klasificirano==k[0])|(band_dataset_bsi_klasificirano==k[1])|(band_dataset_bsi_klasificirano==k[2]),5,-9999)# |(band_dataset_bsi_klasificirano==k[2])
    tlo_klasa = np.where((band_dataset_bsi_klasificirano==f),2,-9999)

    dataset_bsi_klasificirano = None
    dataset_bsi_original = None

    return izgradena_klasa2, tlo_klasa

izgradena2, tlo = Rasporediklasificirano4(mjesto_za_bsi_copy, mjesto_za_KlasifikacijuBSI)

s1 = np.where((voda==-9999),visokav,voda)
s2 = np.where((s1==-9999),niskav,s1)
s3 = np.where((s2==-9999),gola1,s2)
s4 = np.where((s3==-9999),izgradena2,s3)
s5 = np.where((s4==-9999),tlo,s4)
s5 = s5.reshape(Ysize, Xsize)

mjesto_za_final_klasifikaciju = "./"+folder_Cl+"/FinalKlasificirano4665.tif"

createRasterFromTemplate(mjesto_za_final_klasifikaciju, gdal.Open(mjesto_za_KlasifikacijuBSI), s5) 

print('TIME: ', time.time()-pocetak, ' s')
