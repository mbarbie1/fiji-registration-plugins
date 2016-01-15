# @File(label = "Unregistered image stack (first slice is reference)", style = "file") filePath
# @File(label = "Directory to save registered stack", style = "directory") saveDir
# @Float(label = "Pixel size in the plane", value=1.0) pixelSize
# @Integer(label = "target slice index", value=2) targetId
# @Integer(label = "largest slice index to process", value=5) maxSlice
# @Boolean(label = "Scale the stack?", value=0) Scale
# @Integer(label = "Registration scale in pixels long side of the image", value=1000) longSide
# @Boolean(label = "Register the stack?", value=0) Reg
# @Boolean(label = "Translation registration", value=0) translationReg
# @Boolean(label = "Rotation registration", value=0) rotationReg
# @Integer(label = "Stepsize in degrees", value=10) rotationStep
# @Boolean(label = "SIFT registration", value=0) siftReg
# @Boolean(label = "BunwarpJ registration", value=0) bunwarpjReg
# @Boolean(label = "Calculate errors", value=0) calculateError

# SCRIPT: Registration_v4.py
#
#	This script does the basic registration
#
# INPUT
#	filePath:		file path to image stack of unregistered slices
#	saveDir:		Directory to save the registered stack (output)
#	step:			stepsize of the rotation of the target image to optimize similarity with the ref image
#
# OUTPUT
#	regStack		Stack containing registered slices saved in saveDir and named 'registered.tif'
#
# AUTHOR
#	Michael Barbier


# LIBRARIES
#
from ij.gui import Wand
from ij import IJ
from ij import ImagePlus, ImageStack
from ij.process import ImageProcessor, FloatProcessor
from ij.process import ImageStatistics as IS
from ij.process import ShortBlitter
from ij.plugin.filter import Binary
from math import sqrt
from jarray import zeros
from java.awt import Rectangle
from ij.gui import Roi
from ij.plugin import CanvasResizer
#from loci.apps.SlideScannerImport
import SIFT_ExtractPointRoi
#from mpicbg.ij import *
import os
from bunwarpj import Transformation, bUnwarpJ_
from ij.plugin.filter import EDM
from ij.plugin.filter import GaussianBlur

from Michael.external.hausdorff import Hausdorff_Distance

# -----------------------------------------------------------------------------------------------------------------------
# IO FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------------

# WRITE CSV FILE
import csv
from ij.measure import ResultsTable

def writeCSV(filePath, results, header):
	""" Write a table as an csv file """
	rt = ResultsTable()
	for i in range(len(results[1])): 
		rt.incrementCounter()
		for j in range(len(results)):
			rt.addValue(str(header[j]), results[j][i])
	rt.show("Results")
	rt.saveAs(filePath); 


# -----------------------------------------------------------------------------------------------------------------------
# TRANSFORMATIONS
# -----------------------------------------------------------------------------------------------------------------------


def scale(ip, s):
	""" Scale the image with the parameter scale = s """
	imp = ImagePlus('scale',ip)
	IJ.run(imp, "Scale...", "x="+str(s)+" y="+str(s)+" interpolation=Bilinear average");
	ip = imp.getProcessor()
	w = ip.width
	h = ip.height
	cd = CanvasResizer()
	ip = cd.expandImage(ip, int(round(s*w)), int(round(s*h)), -int(round((1-s)/2*w)), -int(round((1-s)/2*h)) )
	return ip

def scaleLongSide(ip, longSide):
	""" Scale the image with respect to the longSide parameter (new size along the long side of the image should equal the longSide parameter) """
	w = ip.width
	h = ip.height
	l = max(w,h)
	s = float(longSide)/l
	imp = ImagePlus('scaleLongSide',ip)
	IJ.run(imp, "Scale...", "x="+str(s)+" y="+str(s)+" interpolation=Bilinear average");
	ip = imp.getProcessor()
	cd = CanvasResizer()
	ip = cd.expandImage(ip, int(round(s*w)), int(round(s*h)), -int(round((1-s)/2*w)), -int(round((1-s)/2*h)) )
	return ip, s

def translation(ref, target):
	return target

def rotation(ref, target):
	return target


# -----------------------------------------------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------------



def maskArea(ip):
	""" compute the number of pixels of an image mask (pixels == 1) """
	stats = ip.getStatistics()
	return stats.area * stats.mean / ip.maxValue()

def area(ip):
	""" compute the number of pixels of an image """
	stats = ip.getStatistics()
	return stats.area

def sumOfSquares(ip):
	""" compute the sum of the squares of the pixels of an image """
	ipTemp = ip.duplicate()
	ipTemp = ipTemp.convertToShort(0)
	ipTemp.sqr()
	stats = ipTemp.getStatistics()
	return stats.mean * stats.area

def sumOfProduct(ip1,ip2):
	""" compute the sum of the product image of two images """
	ip1Temp = ip1.duplicate()
	ip1Temp = ip1Temp.convertToShort(0)
	ip2Temp = ip2.duplicate()
	ip2Temp = ip2Temp.convertToShort(0)
	ip1Temp.copyBits(ip2Temp, 0, 0, ShortBlitter.MULTIPLY)
	stats = ip1Temp.getStatistics()
	return stats.mean * stats.area

def mask(ip, valueA, valueB):
	""" Mask the image with a threshold value ranging between valueA and valueB"""
	ip.setThreshold(valueA, valueB, ImageProcessor.NO_LUT_UPDATE)
	imp = ImagePlus('Mask', ip)
	IJ.run(imp, "Convert to Mask", "")
	
	return imp

def maskIntersection(ip1,ip2):
	""" Intersection of 2 images by their masks excluding zero pixels  """
	ip1Temp = ip1.duplicate()
	ip1Temp = ip1Temp.convertToShort(0)
	ip2Temp = ip2.duplicate()
	ip2Temp = ip2Temp.convertToShort(0)
	imp1 = mask(ip1Temp, 1, ip1Temp.maxValue())
	imp2 = mask(ip2Temp, 1, ip2Temp.maxValue())
	#imp1.show()
	#imp2.show()
	ip1Temp = imp1.getProcessor()
	ip2Temp = imp2.getProcessor()
	ip1Temp.copyBits(imp2.getProcessor(), 0, 0, ShortBlitter.AND)
	return ip1Temp

def maskUnion(ip1,ip2):
	""" Intersection of 2 images by their masks excluding zero pixels  """
	ip1Temp = ip1.duplicate()
	ip1Temp = ip1Temp.convertToShort(0)
	ip2Temp = ip2.duplicate()
	ip2Temp = ip2Temp.convertToShort(0)
	imp1 = mask(ip1Temp, 1, ip1Temp.maxValue())
	imp2 = mask(ip2Temp, 1, ip2Temp.maxValue())
	#imp1.show()
	#imp2.show()
	ip1Temp = imp1.getProcessor()
	ip2Temp = imp2.getProcessor()
	ip1Temp.copyBits(imp2.getProcessor(), 0, 0, ShortBlitter.OR)
	return ip1Temp

def minus(ip1,ip2):
	""" compute the subtraction image of two images """
	ip1Temp = ip1.duplicate()
	ip1Temp = ip1Temp.convertToShort(0)
	ip2Temp = ip2.duplicate()
	ip2Temp = ip2Temp.convertToShort(0)
	ip1Temp.copyBits(ip2Temp, 0, 0, ShortBlitter.SUBTRACT)
	return ip1Temp

def test_mask():
	""" Test of the function: mask(ip, valueA, valueB) """
	imp = IJ.openImage(filePathTest.getAbsolutePath())
	ip = imp.getProcessor().duplicate()
	valueA = 0
	valueB = 1
	mask(ip, valueA, valueB).show()

#test_mask()

def test_maskIntersection():
	""" Test of the function: maskIntersection(ip1, ip2) """
	# load the input stack as an ImagePlus
	imp = IJ.openImage(filePath.getAbsolutePath())
	stack = imp.getStack()
	sizeZ = imp.getStackSize()
	ip1 = stack.getProcessor(1).duplicate()
	ip2 = stack.getProcessor(2).duplicate()
	ImagePlus('ip1',ip1).show()
	ImagePlus('ip2',ip2).show()
	ImagePlus('intersection',maskIntersection(ip1,ip2)).show()

#test_maskIntersection()

def test_maskArea():
	""" Test of the function maskArea """

	# 8bit test
	imp = IJ.openImage(filePathTest.getAbsolutePath())
	ip = imp.getProcessor().duplicate()
	impMask = mask( ip, 1, ip.maxValue() )
	impMask.show()
	outMaskArea = maskArea(impMask.getProcessor())
	outArea = area(impMask.getProcessor())
	print(outMaskArea)
	print(outArea)
	print(outMaskArea/outArea)

	# 16bit test
	imp = IJ.openImage(filePath.getAbsolutePath())
	stack = imp.getStack()
	ip = stack.getProcessor(1).duplicate()
	sizeZ = imp.getStackSize()
	impMask = mask( ip, 1, ip.maxValue() )
	#impMask.show()
	outMaskArea = maskArea( impMask.getProcessor() )
	outArea = area( impMask.getProcessor() )
	print(outMaskArea)
	print(outArea)
	print(outMaskArea/outArea)

#test_maskArea()


# -----------------------------------------------------------------------------------------------------------------------
# SIMILARITY MEASURE
# -----------------------------------------------------------------------------------------------------------------------


def CC(ip1,ip2):
	""" The crossCorrelation (unnormalized) is defined as CC(X,Y) = sum_i(Xi*Yi) / sqrt( sum_i(Xi)^2 * sum_i(Yi)^2 )  """
	denom = sqrt( sumOfSquares(ip1) * sumOfSquares(ip2) )
	num = sumOfProduct(ip1,ip2)
	return num/denom

def NCC(ip1,ip2):
	""" The normalized crossCorrelation is defined as NCC(X,Y) = sum_i((Xi-X)*(Yi-Y)) / sqrt( sum_i(Xi-X)^2 * sum_i(Yi-Y)^2 )  """
	stats1 = ip1.getStatistics()
	stats2 = ip2.getStatistics()
	ipd1 = ip1.duplicate()
	ipd2 = ip2.duplicate()
	ipd1.subtract(stats1.mean)
	ipd2.subtract(stats2.mean)
	denom = sqrt( sumOfSquares( ipd1 ) * sumOfSquares( ipd2 ) )
	num = sumOfProduct(ipd1,ipd2)
	return num/denom

def MSE_nonEmpty(ip1,ip2):
	""" The MSE (= Mean Square Error) is defined as MSE(X,Y) = sum_i(Yi-Xi)^2 / N, Here we take into account only occupied pixels (only pixels where at least one of the images is nonzero) """
	return sumOfSquares( minus(ip1,ip2) ) / maskArea( maskUnion(ip1,ip2) )

def MSE(ip1,ip2):
	""" The MSE (= Mean Square Error) is defined as MSE(X,Y) = sum_i(Yi-Xi)^2 / N """
	return sumOfSquares( minus(ip1,ip2) ) / area(ip1)

def RMSE(ip1,ip2):
	""" The RMSE (= Root Mean Square Error) is defined as RMSE(X,Y)  = sqrt[ sum_i(Yi-Xi)^2 / N ]  """
	return sqrt( MSE_nonEmpty(ip1,ip2) )

def NRMSE(ip1,ip2):
	""" The normalized RMSE (= Root Mean Square Error) is defined as NRMSE(X,Y)  = sqrt[ sum_i(Yi-Xi)^2 / N ] / ( max(Yi) - min(Yi) )  """
	stats = ip1.getStatistics()
	return RMSE(ip1,ip2) / (stats.max-stats.min)

def CVRMSE(ip1,ip2):
	""" The normalized RMSE (= Root Mean Square Error) is defined as CVRMSE(X,Y)  = sqrt[ sum_i(Yi-Xi)^2 / N ] / mean(Yi) )  """
	stats = ip1.getStatistics()
	return RMSE(ip1,ip2) / stats.mean

def measureError(ip1,ip2):
	""" Measure the error between the reference image and the target image """
	stats = ip1.getStatistics()
	cc = CC(ip1,ip2)
	ncc = NCC(ip1,ip2)
	mse_roi = MSE_nonEmpty(ip1,ip2)
	mse = sumOfSquares( minus(ip1,ip2) ) / area(ip1)
	rmse = sqrt( mse )
	n_rmse = rmse / (stats.max-stats.min)
	cv_rmse = rmse / stats.mean
	return cc, ncc, mse, mse_roi, rmse, n_rmse, cv_rmse

def cleanMask(impMask):
	""" Retain only the largest mask """
	IJ.run(impMask, 'Dilate','')
	IJ.run(impMask,"Fill Holes",'')
	IJ.run(impMask,"Erode",'')
	IJ.run(impMask,"Erode",'')
	IJ.run(impMask,"Erode",'')
	IJ.run(impMask,"Erode",'')
	IJ.run(impMask,"Erode",'')
	IJ.run(impMask, 'Dilate','')
	IJ.run(impMask, 'Dilate','')
	IJ.run(impMask, 'Dilate','')
	IJ.run(impMask, 'Dilate','')

def VOP( ip1, ip2 ):
	""" The VOP (= Volume Overlap Percentage, or SI = Similarity Index) is defined as VOPj  =  2 * V( intersection(Aj,Bj) ) / ( V(Aj) + V(Bj) ) with V() the volume function, label j, and labeled pixels in image A and B. """

	mask1 = ip1
	mask2 = ip2
	MaskArea1 = maskArea( mask1 )
	MaskArea2 = maskArea( mask2 )

	mI = maskIntersection( mask1, mask2 )
	MaskAreaI = maskArea(mI)
	si = 2 * MaskAreaI / ( MaskArea1 + MaskArea2)
	si1 = MaskAreaI / MaskArea1
	si2 = MaskAreaI / MaskArea2

	return si, si1, si2

def maskIP(ip):
	gb = GaussianBlur()
	sigma = ip.getWidth()/100
#	print(sigma)
	accuracy = 0.01
#	ImagePlus('ip',ip.duplicate()).show()
	gb.blurGaussian(ip, sigma, sigma, accuracy)
#	ImagePlus('ipblur',ip.duplicate()).show()

	t = round( ip.getAutoThreshold() )
	impMask = mask( ip, t, ip.maxValue() )
	IJ.run(impMask,"Fill Holes",'')
	#cleanMask(impMask)
	return impMask.getProcessor()

def test_maskIP():
	imp = IJ.openImage(filePath.getAbsolutePath())
	stack = imp.getStack()
	ip1 = stack.getProcessor(1).duplicate()
	ip2 = stack.getProcessor(2).duplicate()

	mask1 = maskIP(ip1)
	mask2 = maskIP(ip2)
	ImagePlus('mask1',mask1).show()
	ImagePlus('mask2',mask2).show()

#test_maskIP()

def maskOutline(ip):
	imp = ImagePlus('mask',ip)
	IJ.run(imp, "Outline", "")
	return imp

def test_VOP():
	imp = IJ.openImage(filePath.getAbsolutePath())
	stack = imp.getStack()
	ip1 = stack.getProcessor(1).duplicate()
	ip2 = stack.getProcessor(2).duplicate()
	mask1 = maskIP(ip1)
	mask2 = maskIP(ip2)
	[si, s1, s2] = VOP( mask1, mask2 )
	print(si, s1, s2)

#test_VOP()
#test_HD()

def hdist( ip1, ip2 ):

	mask1 = ImagePlus('mask1',ip1)
	mask2 = ImagePlus('mask2',ip2)
	# make outlines from the masks
	IJ.run(mask1, "Outline", "")
	IJ.run(mask2, "Outline", "")
	hd = Hausdorff_Distance()
	hd.exec( mask1, mask2 )

	return hd.getHausdorffDistance(), hd.getAveragedHausdorffDistance()

def test_HD():

	imp = IJ.openImage(filePath.getAbsolutePath())
	stack = imp.getStack()
	ip1 = stack.getProcessor(1).duplicate()
	ip2 = stack.getProcessor(2).duplicate()
	[ip1, s] = scaleLongSide(ip1, longSide)
	if (Scale == 1):
		ip2 = scale(ip2, s)
	maskA = impMask1.duplicate()
	maskB = impMask2.duplicate()

#	IJ.run(maskA, "Invert", "")
#	IJ.run(maskB, "Invert", "")
	IJ.run(maskA, "Outline", "")
	IJ.run(maskB, "Outline", "")
#	IJ.run(maskA, "Invert", "")
#	IJ.run(maskB, "Invert", "")
	maskA.show()
	maskB.show()

	impMask1.show()
	MaskArea1 = maskArea( impMask1.getProcessor() )
	impMask2.show()
	MaskArea2 = maskArea( impMask2.getProcessor() )
	hd = Hausdorff_Distance()
	hd.exec( maskA, maskB )
	print(hd.getAveragedHausdorffDistance())
	print(hd.getHausdorffDistance())
	
# -----------------------------------------------------------------------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------------------------------------------------------------------

def run():
	""" Loads target images, calculates errors with the ref """

	# load the input stack as an ImagePlus
	imp = IJ.openImage(filePath.getAbsolutePath())
	stack = imp.getStack()
	sizeZ = imp.getStackSize()

	# Copy the reference and target slice
	refId = 1
	ref = stack.getProcessor(refId).duplicate()

	

	sizeZ = min(sizeZ, maxSlice)
	if (calculateError == 1):
		eCorrelation = zeros(sizeZ, 'f')
		eNCC = zeros(sizeZ, 'f')
		eMSE = zeros(sizeZ, 'f')
		eMSE_ROI = zeros(sizeZ, 'f')
		eRMSE = zeros(sizeZ, 'f')
		eNRMSE = zeros(sizeZ, 'f')
		eCVRMSE = zeros(sizeZ, 'f')
		si = zeros(sizeZ, 'f')
		si1 = zeros(sizeZ, 'f')
		si2 = zeros(sizeZ, 'f')
		hd = zeros(sizeZ, 'f')
		hda = zeros(sizeZ, 'f')
		eCVRMSE = zeros(sizeZ, 'f')
		for i in range(1, sizeZ+1):
			print(i)
			ip = stack.getProcessor(i).duplicate()
			#ImagePlus('test', ip).show()
			eCorrelation[i-1], eNCC[i-1], eMSE[i-1], eMSE_ROI[i-1], eRMSE[i-1], eNRMSE[i-1], eCVRMSE[i-1] = measureError( ref, ip )
			refMask = maskIP(ref)
			ipMask = maskIP(ip)
			si[i-1], si1[i-1], si2[i-1] = VOP( refMask, ipMask )
			hd[i-1], hda[i-1] = hdist( refMask, ipMask )
			hd[i-1] = hd[i-1] * pixelSize
			hda[i-1] = hda[i-1] * pixelSize
		errorFileName = 'error.txt'
		errorFilePath = os.path.join(saveDir.getAbsolutePath(), errorFileName)
		writeCSV( errorFilePath, [hda, hd, si, si1, si2, eCorrelation, eNCC, eMSE, eMSE_ROI, eRMSE,eNRMSE,eCVRMSE], ["MHD","HD_ref","SI", "SI_ref", "SI_target", "Correlation", "NCC","MSE","MSE_ROI","RMSE","N_RMSE","CV_RMSE"] )
		print('done')

run()
