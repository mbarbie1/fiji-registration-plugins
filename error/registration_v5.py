# @File(label = "Unregistered image stack (first slice is reference)", style = "file") filePath
# @File(label = "Directory to save registered stack", style = "directory") saveDir
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
	impMask.show()
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
	# denom = sqrt( sumOfSquares(ip1) * sumOfSquares(ip2) )
	# num = sumOfProduct( ip1, ip2 )
	# return num/denom

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
	return cc, mse, mse_roi, rmse, n_rmse, cv_rmse

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

	t1 = round(ip1.getAutoThreshold()/2.0)
	impMask1 = mask( ip1, t1, ip1.maxValue() )
	t2 = round(ip2.getAutoThreshold()/2.0)
	impMask2 = mask( ip2, t2, ip2.maxValue() )

	cleanMask(impMask1)
	cleanMask(impMask2)

	impMask1.show()
	MaskArea1 = maskArea( impMask1.getProcessor() )
	impMask2.show()
	MaskArea2 = maskArea( impMask2.getProcessor() )

	mI = maskIntersection( impMask1.getProcessor(), impMask2.getProcessor() )
	MaskAreaI = maskArea(mI)
	si = 2 * MaskAreaI / ( MaskArea1 + MaskArea2)
	si1 = MaskAreaI / MaskArea1
	si2 = MaskAreaI / MaskArea2

	return si, si1, si2


def test_VOP2():
	imp = IJ.openImage(filePath.getAbsolutePath())
	stack = imp.getStack()
	ip1 = stack.getProcessor(1).duplicate()
	ip2 = stack.getProcessor(2).duplicate()
	[si, s1, s2] = VOP( ip1, ip2 )
	print(si, s1, s2)
	return

def test_VOP():
	# 16bit test
	imp = IJ.openImage(filePath.getAbsolutePath())
	stack = imp.getStack()
	ip1 = stack.getProcessor(1).duplicate()
	ip2 = stack.getProcessor(2).duplicate()
	sizeZ = imp.getStackSize()
	t1 = ip1.getAutoThreshold()/2
	impMask1 = mask( ip1, t1, ip1.maxValue() )
	t2 = ip2.getAutoThreshold()/2
	impMask2 = mask( ip2, t2, ip2.maxValue() )

	cleanMask(impMask1)
	cleanMask(impMask2)

	impMask1.show()
	MaskArea1 = maskArea( impMask1.getProcessor() )
	impMask2.show()
	MaskArea2 = maskArea( impMask2.getProcessor() )

	w = Wand(impMask1.getProcessor())
	#print(w.allPoints())
	round(impMask1.getProcessor().getWidth()/2.0)
#	w.autoOutline( round(impMask1.getProcessor().getWidth()/2.0) , round( impMask1.getProcessor().getHeight()/2.0) )
#	print(w.xpoints())
#	print(w.ypoints())
	
	totArea = area( impMask1.getProcessor() )
	print("MaskArea1: ", MaskArea1)
	print("MaskArea2: ", MaskArea2)
	print("Total Area: ", totArea)
	mIntersection = maskIntersection( impMask1.getProcessor(), impMask2.getProcessor())
	ImagePlus('intersection', mIntersection).show()
	return mIntersection
	#VOP( impMask1, impMask2 )

#test_VOP()
test_VOP2()

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
# REGISTRATION FUNCTIONS
# -----------------------------------------------------------------------------------------------------------------------

def rotationSingle(ref,target,rotationStep):
	""" find the optimal rotation of the target image by computing the similarity with the reference image for a large amount of angles."""
	refTemp = ref.duplicate()
	ra = rotationStep;
	maxRot = -(ra/2)
	maxCorr = 0
	for j in xrange(0, 360-1, ra):
		angle = j-(ra/2)
		ip = target.duplicate()
		IJ.run(ImagePlus('rotated',ip), "Rotate...", "angle="+str(angle)+" interpolate")
		corrCoeff = CC(ref,ip)
		if (corrCoeff>maxCorr):
			maxCorr=corrCoeff
			maxRot=angle
			rotTarget=ip
		print( j, maxRot, corrCoeff, maxCorr)
	return maxRot, rotTarget

def siftFIJI(ref, reg):
	""" perform SIFT registration """
	tmp = reg.duplicate()
	imp = IJ.run(tmp, "Linear Stack Alignment with SIFT", "initial_gaussian_blur=1.6 steps_per_scale_octave=4 minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Affine interpolate");
	return imp

from java.util import ArrayList
from mpicbg.ij import SIFT
from mpicbg.imagefeatures import Feature
from mpicbg.imagefeatures import FloatArray2DSIFT

	#impRef.show()
	#impTarget.show()
#	print(t)
#	param = FloatArray2DSIFT.Param()
#	sift = SIFT(FloatArray2DSIFT(param))
#	features = ArrayList()
#	sift.extractFeatures(ref, features)
	#for feature in features:
	#	print("x: " + str(feature.location[0]) + ", y: " + str(feature.location[1]));

	#IJ.run(imp, "Extract SIFT Correspondences", "source_image=sift_source target_image=sift_target initial_gaussian_blur="+
	#str(initial_gaussian_blur) +" steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size="+
	#str(feature_descriptor_size) +
	#" feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 filter maximal_alignment_error=25 minimal_inlier_ratio=0.05 minimal_number_of_inliers=7 expected_transformation="+expected_transformation)

#	impRef.show()
#	impTarget.show()


def siftSingle(ref, target):
	""" perform SIFT registration for one image """
	impRef = ImagePlus('Sift_source', ref)
	impTarget = ImagePlus('Sift_target', target)
	expected_transformation = 'Similarity'
	initial_gaussian_blur = 2
	feature_descriptor_size = 8
	imp = ImagePlus('siftRefTest', ref)
	t = SIFT_ExtractPointRoi()
	t.exec( impRef, impTarget )
	roiRef = impRef.getRoi()
	roiTarget = impTarget.getRoi()
	return roiRef, roiTarget

def bunwarpjSingle(impRef, impTarget, landMarks, directTransfoName, inverseTransfoName):
	directName = 'direct_transformation'
	inverseName = 'inverse_transformation'
	impRef.setTitle('bunwarpj_source')
	impTarget.setTitle('bunwarpj_target')
	directFileName = os.path.join(saveDir.getAbsolutePath(), directName)
	inverseFileName = os.path.join(saveDir.getAbsolutePath(), inverseName)
		#imp = IJ.run(impRef, "bUnwarpJ", "source_image=bunwarpj_source target_image=bunwarpj_target registration=Accurate image_subsample_factor=0 initial_deformation=[Coarse] final_deformation=Fine divergence_weight=0 curl_weight=0 landmark_weight=0.5 image_weight=0.5 consistency_weight=10 stop_threshold=0.01 save_transformations save_direct_transformation=["+ directFileName +"] save_inverse_transformation=["+ inverseFileName +"]");
#		transfo = computeTransformationBatch(
# 			param ImagePlus targetImp input target image 
# 			param ImagePlus sourceImp input source image
# 			param ImageProcessor targetMskIP target mask 
# 			param ImageProcessor sourceMskIP source mask
# 			param int mode accuracy mode (0 - Fast, 1 - Accurate, 2 - Mono)
# 			param int img_subsamp_fact image subsampling factor (from 0 to 7, representing 2^0=1 to 2^7 = 128)
# 			param int min_scale_deformation (0 - Very Coarse, 1 - Coarse, 2 - Fine, 3 - Very Fine)
# 			param int max_scale_deformation (0 - Very Coarse, 1 - Coarse, 2 - Fine, 3 - Very Fine, 4 - Super Fine)
# 			param double divWeight divergence weight
# 			param double curlWeight curl weight
# 			param double landmarkWeight landmark weight
# 			param double imageWeight image similarity weight
# 			param double consistencyWeight consistency weight
# 			param double stopThreshold stopping threshold
# 			return results transformation object
	accuracy_mode = 1
	img_subsamp_fact = 0
	tmp = FloatProcessor(impTarget.getWidth(), impTarget.getHeight())
	tmp.setValue(1.0)
	tmp.fill()
	targetMskIp = tmp.duplicate()
	sourceMskIp = tmp.duplicate()
	accuracy_mode = 1
	img_subsamp_fact = 0
	min_scale_deformation = 1
	max_scale_deformation = 2
	divWeight = 0.0
	curlWeight = 0.0
	consistencyWeight = 10
	stopThreshold = 0.01
	if landMarks == 1:
		landmarkWeight = 0.5
		imageWeight = 0.5
		transfo = bUnwarpJ_.computeTransformationBatch(
			impTarget, impRef, targetMskIp, sourceMskIp, accuracy_mode, img_subsamp_fact, min_scale_deformation, max_scale_deformation, divWeight, curlWeight, landmarkWeight, imageWeight, consistencyWeight, stopThreshold)
	else:
		landmarkWeight = 0.0
		imageWeight = 1.0
		transfo = bUnwarpJ_.computeTransformationBatch(
			impTarget, impRef, targetMskIp, sourceMskIp, accuracy_mode, img_subsamp_fact, min_scale_deformation, max_scale_deformation, divWeight, curlWeight, landmarkWeight, imageWeight, consistencyWeight, stopThreshold)
	transfo.saveDirectTransformation(os.path.join(saveDir.getAbsolutePath(), directTransfoName))
	transfo.saveInverseTransformation(os.path.join(saveDir.getAbsolutePath(), inverseTransfoName))
	#print(transfo.getCoefficients())
	return transfo.getInverseResults().getStack().getProcessor(1).duplicate()

def bunwarpjFIJI(ref,reg):
	directName = 'direct_transformation'
	inverseName = 'inverse_transformation'
	regStack = reg.getStack()
	sizeZ = reg.getStackSize()
	source = ImagePlus('source', ref.duplicate() )
	for i in range(1, sizeZ):
		ip = regStack.getProcessor(i).duplicate()
		target = ImagePlus('target', ip)
		directFileName = os.path.join(saveDir.getAbsolutePath(), directName + '_' + str(i))
		inverseFileName = os.path.join(saveDir.getAbsolutePath(), inverseName + '_' + str(i))
		imp = IJ.run(source, "bUnwarpJ", "source_image=source target_image=target registration=Accurate image_subsample_factor=0 initial_deformation=[Very Coarse] final_deformation=Fine divergence_weight=0 curl_weight=0 landmark_weight=0 image_weight=1 consistency_weight=10 stop_threshold=0.01 save_transformations save_direct_transformation=["+ directFileName +"] save_inverse_transformation=["+ inverseFileName +"]");
	return imp




# -----------------------------------------------------------------------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------------------------------------------------------------------

def run():
	""" Loads an image stack which contains both reference and target images for the registration
		Scales the images to have their largest side equal to longSide
		Registration is performed:
			- translation (brute force optimization)
			- rotation (brute force optimization)
			- sift registration
			- bunwarpj registration
		Calculation of the errors by different methods """

	# load the input stack as an ImagePlus
	imp = IJ.openImage(filePath.getAbsolutePath())
	stack = imp.getStack()
	sizeZ = imp.getStackSize()

	LAND_MARKS = 0

	# Copy the reference and target slice
	refId = 1
	ref = stack.getProcessor(refId).duplicate()

	if (Scale == 1):
		[ref, s] = scaleLongSide(ref, longSide)

	sizeZ = min(sizeZ, maxSlice)
	stackReg = ImageStack(ref.getWidth(), ref.getHeight())
	stackReg.addSlice(ref)
# = stack.duplicate()
	for i in range(2, sizeZ+1):
		targetId = i
		target = stack.getProcessor(targetId).duplicate()

		# Scale the slices: scale the reference slice using the longSide parameter, and use the same scale for the target slice.
		if (Scale == 1):
			target = scale(target, s)
			#ImagePlus('Ref',ref).show()
			#ImagePlus('Target',target).show()
	
		if (Reg == 1):
	
			if (translationReg == 1):
				target = translation(ref, target)
	
			if (rotationReg == 1):
				[rot, target] = rotationSingle(ref,target,rotationStep)
	
			if (siftReg == 1):
				[roiRef, roiTarget] = siftSingle(ref, target)
				impTarget = ImagePlus('Target',target)
				impTarget.setRoi(roiTarget)
				#impTarget.show()
				impRef = ImagePlus('Ref',ref)
				impRef.setRoi(roiRef)
				#impRef.show()
				LAND_MARKS = 1

			if (bunwarpjReg == 1):
				target = bunwarpjSingle(impRef, impTarget, LAND_MARKS, 'direct_transfo_' + str(i) + '.txt', 'inverse_transfo_' + str(i) + '.txt')
				impTarget = ImagePlus('unwarpj_target', target)
				#impTarget.show()
				fileName = 'target_id' + str(targetId) + '.tif'
				IJ.saveAs(impTarget, "Tiff", os.path.join(saveDir.getAbsolutePath(), fileName))

			#stackReg.setProcessor(target.convertToShortProcessor(), i)
			stackReg.addSlice(target)

	if (calculateError == 1):
		eCorrelation = zeros(sizeZ, 'f')
		eMSE = zeros(sizeZ, 'f')
		eMSE_ROI = zeros(sizeZ, 'f')
		eRMSE = zeros(sizeZ, 'f')
		eNRMSE = zeros(sizeZ, 'f')
		eCVRMSE = zeros(sizeZ, 'f')
		si = zeros(sizeZ, 'f')
		si1 = zeros(sizeZ, 'f')
		si2 = zeros(sizeZ, 'f')
		eCVRMSE = zeros(sizeZ, 'f')
		for i in range(1, sizeZ+1):
			ip = stackReg.getProcessor(i).duplicate()
			#ip = stack.getProcessor(i).duplicate()
			#ImagePlus('test',ip).show()
			eCorrelation[i-1], eMSE[i-1], eMSE_ROI[i-1], eRMSE[i-1], eNRMSE[i-1], eCVRMSE[i-1] = measureError(ref,ip)
			si[i-1], si1[i-1], si2[i-1] = VOP( ref,ip )
		errorFileName = 'error.txt'
		errorFilePath = os.path.join(saveDir.getAbsolutePath(), errorFileName)
		writeCSV( errorFilePath, [si, si1, si2, eCorrelation,eMSE, eMSE_ROI, eRMSE,eNRMSE,eCVRMSE], ["si", "si1", "si2", "Correlation","MSE","MSE_ROI","RMSE","N_RMSE","CV_RMSE"] )

run()
