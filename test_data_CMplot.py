import numpy as np
from osgeo import gdal
from osgeo import ogr
import geopandas as gpd
import pandas as pd
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

RasterPath = "D:/diplomski_rad/slike/rijeka/merg/merged4.tif"
RasterFileDataset = gdal.Open(RasterPath)
nrows = RasterFileDataset.RasterYSize
ncols = RasterFileDataset.RasterXSize
driverTiff = gdal.GetDriverByName('GTiff')

#gdf = gpd.read_file("D:/diplomski_rad/slike/rijeka/shps/train_points2.shp")

# TEST PODACI
# D:/diplomski_rad/slike/rijeka/shps/test.shp

#print(gdf.head())

#trainFile = "D:/diplomski_rad/slike/rijeka/shps/test_points2.shp"

trainFile = "D:/diplomski_rad/slike/rijeka/shps/test.shp"

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

print(data.min(), data.max(), data.mean())

#predDs = gdal.Open('D:/diplomski_rad/Konacni_rezultati/Nadzirana_rezultati/RI/rf_class.tif')
#predDs = gdal.Open('D:/diplomski_rad/Konacni_rezultati/Nadzirana_rezultati/RI/svm_class.tif')
#predDs = gdal.Open('D:/diplomski_rad/Konacni_rezultati/PB_rezultati/FinalKlasificirano0989701.tif')
#predDs = gdal.Open('D:/diplomski_rad/Konacni_rezultati/OB_rezultati_RF/classified4044.tif')
#predDs = gdal.Open('D:/diplomski_rad/Konacni_rezultati/OB_rezultati_NN/classifiedRI2.tif')

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

sn.heatmap(cm, annot=True, annot_kws={"size": 16}, vmin=0.0, vmax=100.0)
bottom, top = plt.ylim()
plt.ylim(bottom+0.5, top-0.5)
plt.show()
