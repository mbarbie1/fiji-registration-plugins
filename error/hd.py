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

from ij import IJ, ImagePlus, ImageStack
from ij.process import ImageProcessor, FloatProcessor
from ij.process import ImageStatistics as IS
from ij.process import ShortBlitter
from math import sqrt
import os


def mask(ip, valueA, valueB):
	""" Mask the image with a threshold value ranging between valueA and valueB"""
	ip.setThreshold(valueA, valueB, ImageProcessor.NO_LUT_UPDATE)
	imp = ImagePlus('Mask', ip)
	IJ.run(imp, "Convert to Mask", "")
	return imp

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


def hd(mask1, mask2):
		
#		// Get outline of binary masks
	maskA = mask1.duplicate();
	maskB = mask2.duplicate();

	IJ.run(maskA, "Invert", "");
	IJ.run(maskB, "Invert", "");
	IJ.run(maskA, "Outline", "");
	IJ.run(maskB, "Outline", "");
	IJ.run(maskA, "Invert", "");
	IJ.run(maskB, "Invert", "");

	pixelsContourA = maskA.getStatistics().histogram[len(maskA.getStatistics().histogram) - 1];
	pixelsContourB = maskB.getStatistics().histogram[len(maskB.getStatistics().histogram) - 1];
		
#		// For every pixelA>0 (part of outlineA): store coordinates (n=pixelsContourB times) in coordA
	height = maskA.getHeight();
	width = maskA.getWidth();
	ip = maskA.getProcessor();

	coordA = [[[0 for k in xrange(2)] for j in xrange(pixelsContourB)] for i in xrange(pixelsContourA)]
	i = 0
	
	for y in range(height-1, 0, -1):
		for x in range(0, width):
			v = ip.getPixelValue(x, y)
			if (v > 0):
				for j in range(0, pixelsContourB):
					coordA[0][j][i] = x
				for j in range(0, pixelsContourB):
					coordA[1][j][i] = y
					print(y)
				i = i + 1
	print(coordA)

def test_HD():
	imp = IJ.openImage(filePath.getAbsolutePath())
	stack = imp.getStack()
	ip1 = stack.getProcessor(1).duplicate()
	ip2 = stack.getProcessor(2).duplicate()
	sizeZ = imp.getStackSize()
	t1 = round(ip1.getAutoThreshold()/2.0)
	impMask1 = mask( ip1, t1, ip1.maxValue() )
	t2 = round(ip2.getAutoThreshold()/2.0)
	impMask2 = mask( ip2, t2, ip2.maxValue() )

	cleanMask(impMask1)
	cleanMask(impMask2)

	impMask1.show()

	hd(impMask1, impMask2)

test_HD()