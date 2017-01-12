from numba import cuda
import numpy as np
import Constants as const
import Dm3Reader3 as dm3
import ImageSupport as imsup
import CrossCorr as cc

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

imgData = dm3.ReadDm3File('holo1.dm3')
imgMatrix = imsup.PrepareImageMatrix(imgData, const.dimSize)
img1 = imsup.Image(const.dimSize, const.dimSize, imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'])
img1.amPh.am = np.sqrt(imgMatrix).astype(np.float32)

imgData = dm3.ReadDm3File('holo2.dm3')
imgMatrix = imsup.PrepareImageMatrix(imgData, const.dimSize)
img2 = imsup.Image(const.dimSize, const.dimSize, imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'])
img2.amPh.am = np.sqrt(imgMatrix).astype(np.float32)

imgBigger = imsup.MagnifyImage(img1, 2)
imsup.DisplayAmpImage(imgBigger)

# mcfBest, img2Rot = cc.MaximizeMCFvsRotation(img1, img2, 1, 359)
# shift = cc.GetShift(mcfBest)
# img2RotShifted = cc.ShiftImage(img2Rot, shift)
# cropCoords = imsup.DetermineCropCoords(img2RotShifted.width, img2RotShifted.height, shift)
# squareCoords = imsup.MakeSquareCoords(cropCoords)
# print(squareCoords)
# img1Cropped = imsup.CropImageROICoords(img1, squareCoords)
# img2RotShiftedCropped = imsup.CropImageROICoords(img2RotShifted, squareCoords)
# imsup.SaveAmpImage(img1Cropped, 'holo1.png')
# imsup.SaveAmpImage(img2RotShiftedCropped, 'holo2Aligned.png')