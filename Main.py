from numba import cuda
import numpy as np
import Dm3Reader3 as dm3
import ImageSupport as imsup

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

imgData = dm3.ReadDm3File('img1.dm3')
imgMatrix = imsup.PrepareImageMatrix(imgData, 1024)
img = imsup.Image(1024,1024, imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'])
img.amPh.am = np.sqrt(imgMatrix)

imsup.RotateImage(img, 137)