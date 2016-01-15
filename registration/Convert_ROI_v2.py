# @File(label = "svg file", style = "file") xmlPath
# @File(label = "Imagej ROI-file", style = "file") roiPath
# @File(label = "Image with imagej ROI to be converted", style = "file") filePath
# @File(label = "Directory to save the converted ROI", style = "directory") saveDir
# @Boolean(label = "Convert to Evotec ROI?", value=0) convertEvotec
# @Boolean(label = "Convert to svg", value=0) convertSvg

from ij import IJ
from ij.io import RoiDecoder, RoiEncoder
from ij.gui import PolygonRoi, Roi
from ij.plugin.frame import RoiManager
import math
import os
import zipfile
from javax.xml.xpath import XPathFactory, XPathConstants
from javax.xml.parsers import DocumentBuilder, DocumentBuilderFactory, ParserConfigurationException
from java.io import StringReader
from org.xml.sax import InputSource
from org.apache.batik.svggen import SVGGraphics2D

from java.io import IOException
from org.apache.batik.dom.svg import SAXSVGDocumentFactory
from org.apache.batik.util import XMLResourceDescriptor
from org.w3c.dom import Document
from org.apache.batik.swing import JSVGCanvas
from javax.swing import JPanel, JFrame
from java.awt.event import WindowAdapter

def readSVG(svgPath):
	#parser = XMLResourceDescriptor.getXMLParserClassName()
	#f = SAXSVGDocumentFactory(parser)
	#print(svgPath)
	#doc = f.createDocument(svgPath.toURI().toString())
	#print(doc)
	svgCanvas = JSVGCanvas()
	svgCanvas.setURI(svgPath.toURI().toString())
	f = JFrame('SVG image', size=(1000, 1000), locationRelativeTo=None)
	p = JPanel()
	p.add(svgCanvas)
	f.getContentPane().add(p)
	f.setVisible(True)
	#f.addWindowListener(WindowAdapter())

def roiToPath(roi):
	text = ""
	if roi is not None:
		# Get ROI points
		polygon = roi.getPolygon()
		n_points = polygon.npoints
		x = polygon.xpoints
		y = polygon.ypoints
		text = text + "<path d='m"
		x0 = 0
		y0 = 0
		for i in range(0, len(x)):
			text = text + " " + str(x[i] - x0) + "," + str(y[i] - y0)
			x0 = x[i]
			y0 = y[i]
		text = text + " z' />"
	return(text)

def loadXMLFromString(xml):
	db = DocumentBuilderFactory.newInstance().newDocumentBuilder()
	inputSource = InputSource(StringReader(xml));
	return(db.parse(inputSource))

def loadXMLFromFile(xml):
	db = DocumentBuilderFactory.newInstance().newDocumentBuilder()
	return(db.parse(xml))

def svgToRois(xmlPath):
	doc = loadXMLFromFile(xmlPath)
	xpath = XPathFactory.newInstance().newXPath();
	expression = "//path[@structure_id=\"8\"]/@d"
	print("svg:xpath")
	print(xpath.evaluate(expression, doc))
	return(xpath.evaluate(expression, doc))

import re

def pathCoordsToRoi(coordString):
	print(coordString)
	#m = re.search('(?<=c)\w+', coordString)
	#m.group(0)
	l = re.split('c',coordString)
	x1 = []
	x2 = []
	y1 = []
	y2 = []
	text1 = ""
	text2 = ""
	print("Print coordinates")
	for c in l:
		text1 = text1 + c
		text1 = text1 + "\n"
#		if c[0] == 'c':
#			print('coordinate')
#			ll = re.split(',|[-]',c)
#			for cc in ll:
#				print(cc)
#				text2 = text2 + cc + " "
#				text2 = text2 + "\n"
#		else:
#			print('not a coordinate')
#			ll = re.split(',|[-]',c)
#			for cc in ll:
#				print(cc)
#				text2 = text2 + cc + " "
#				text2 = text2 + "\n"
		print(c)
	#print(m.group(0))
	print("END coordinates")
	return(text1,text2)

def pathToRoi(path):
	doc = loadXMLFromString(path)
	xpath = XPathFactory.newInstance().newXPath();
	expression = "//path[@structure_id=\"8\"]/@d"
	print("xpath")
	print(xpath.evaluate(expression, doc))
	nodes = xpath.evaluate(expression, doc, XPathConstants.NODESET)
	nodeString = xpath.evaluate(expression, doc, XPathConstants.STRING)
	print("nodeString:")
	print(nodeString)
	#NodeList nodeList = (NodeList) xPath.compile(expression).evaluate(xmlDocument, XPathConstants.NODESET);
	print(nodes.length)
	for i in range(0, nodes.length):
		print(nodes.item(i).getNodeName())
	#print(nodes.item(1).getFirstChild().getNodeValue()) 

	return("")
#	# width image from path
#	w = 
#	# height image from path
#	h = 
#	# new roi
#	roi = 
#	
#	# Get ROI points
#	polygon = roi.getPolygon()
#	n_points = polygon.npoints
#	x = polygon.xpoints
#	y = polygon.ypoints
#	text = text + "<path d='m"
#	x0 = 0
#	y0 = 0
#	for i in range(0, len(x)):
#		text = text + " " + str(x[i] - x0) + "," + str(y[i] - y0)
#		x0 = x[i]
#		y0 = y[i]
#	text = text + " z' />"
#
#	return(roi)

def roiToEvotecRegion(roi, w, h, Xlt, Ylt, Xrb, Yrb):
	text = "["
	typeRegion = "IN-spline"
	headerRegion = "[" + typeRegion + ";"+ str(w) + "," + str(h) + ";" + str(Xlt) + "," + str(Ylt) + ";" +str(Xrb)+","+str(Yrb) + "]"
	text = text + headerRegion
	if roi is not None:
		# Get ROI points
		polygon = roi.getPolygon()
		n_points = polygon.npoints
		x = polygon.xpoints
		y = polygon.ypoints
		for i in range(0, len(x)):
			text = text + str(x[i]) + "," + str(y[i])
			if i < (len(x)-1):
				text = text + ";"
		text = text + "]"
	return(text)

def EvotecRegionToRoi(roi, w, h, Xlt, Ylt, Xrb, Yrb):
	text = "["
	typeRegion = "IN-spline"
	headerRegion = "[" + typeRegion + ";"+ str(w) + "," + str(h) + ";" + str(Xlt) + "," + str(Ylt) + ";" +str(Xrb)+","+str(Yrb) + "]"
	text = text + headerRegion
	if roi is not None:
		# Get ROI points
		polygon = roi.getPolygon()
		n_points = polygon.npoints
		x = polygon.xpoints
		y = polygon.ypoints
		for i in range(0, len(x)):
			text = text + str(x[i]) + "," + str(y[i])
			if i < (len(x)-1):
				text = text + ";"
		text = text + "]"
	return(text)

def roisToSVG(rois, w, h):
	text = "<svg width=\""+ str(w) +"\" height=\""+ str(h) +"\"><g>"
	j = 0
	for roi in rois:
		j = j + 1
		print(j)
		text = text + roiToPath(roi)
	text = text + "</g></svg>"
	return(text)

def roisFromImagePlusOverlay(imp):
	w = imp.getWidth()
	h = imp.getHeight()
	rois = imp.getOverlay().toArray()
	return(rois, w, h)

def roisFromFile(filePath):
	""" Get rois ROI[] from an roi file encoded by imagej """
	rois = []
	isZip = zipfile.is_zipfile(filePath)
	if isZip:
		print("It is a zip-file")
		zf = zipfile.ZipFile(filePath, 'r')
		print zf.namelist()
		for filename in zf.namelist():
			try:
				roiFile = zf.read(filename)
				rd = RoiDecoder(roiFile)
				roi = rd.getRoi()
				rois.append(roi)
			except KeyError:
				print 'ERROR: Did not find %s in zip file' % filename
	else:
		rd = RoiDecoder(filePath)
		rois.append(rd.getRoi())
	return(rois)

def writeTextFile(text, saveDirectory, ext, fileName):
	""" Write a text file """
	filePath = os.path.join(saveDirectory.getAbsolutePath(), fileName + '.' + ext)
	try:
		f = open(filePath,'w')
		f.write(text)
	finally:
		f.close()

def run():
	""" Loads an image which contains ROIs, converts the ROIs
		- to Evotec ROIs as an xml file, and/or
		- an svg image 

		TODO: error with reading imagej Roi zip files	"""

### Converting: svg --> ROI
	path = r'<svg width="14976" height="7616" xmlns="http://www.w3.org/2000/svg"><path id="1334669" parent_id="1334668" order="0" structure_id="8" d="M9467.215,5791.594   c82.746,153.9-39.498,279.637-145.297,374.731c-100.779,90.594-181.132,201.04-275.882,297.501   c-194.346,197.852-446.452,346.122-715.326,415.339c-197.956,50.952-375.339,57.515-574.739,10.292   c-163.894-38.817-238.032,23.184-374.617,96.708c-63.755,34.327-258.936,135.235-273.279,1.944   c-7.254-67.693,85.139-157.956,131.84-198.115c-75.233-21.777-179.232,63.946-259.551,73.706   c-107.278,13.026-218.306-4.761-326.646,7.151c-117.271,12.89-237.231,34.839-352.385,60.732   c-172.447,38.776-302.694,170.854-484.64,176.229c-184.149,5.439-355.878-49.775-514.428-140.981   c-109.53-63.024-206.267-143.981-286.415-241.022c-90.444-109.505-186.259-225.456-306.194-304.368   c-114.881-75.587-263.275-101.035-395.37-130.314c-210.799-46.727-428.626-63.547-643.334-35.954   c-211.209,27.146-412.149,76.014-626.047,80.668c-189.976,4.136-361.405-45.697-540.055-107.974   c-313.864-109.416-592.335-297.609-864.036-483.233c-168.544-115.146-305.872-300.74-461.981-435.467   c-170.078-146.796-346.666-292.801-464.156-487.593c-205.159-340.133,168.338-865.886,403.364-1091.214   c334.837-321.021,795.461-620.786,1270.673-652.714c203.342-13.663,451.505,36.074,586.741,199.88   c-133.226-249.919-140.221-425.897,39.538-633.947c504.592-584.002,1198.894-907.949,1923.306-1131.087   c860.56-265.083,1703.133-608.751,2591.13-775.18c388.287-72.771,785.628-112.479,1180.868-92.317   c160.411,8.182,304.905,36.956,471.539,62.145c196.454,29.697,414.285,60.732,588.465,139.16   c178.249,80.277,465.341,346.427,298.729,563.811c206.914-5.754,415.695,85.503,596.221,178.214   c62.759,32.229,130.516,63.325,201.264,71.23c112.855,12.605,207.703-84.423,312.921-110.243   c136.787-33.571,266.943-22.791,402.228,10.816c112.344-107.092,319.473-73.14,456.787-44.478   c353.834,73.854,692.672,90.21,1005.524,291.487c193.171,124.282,431.306,278.311,558.702,473.092   c42.081,64.333,61.586,130.708,109.786,189.1c132.437,160.473,311.515,311.912,333.839,533.115   c8.01,79.385-5.186,96.765,36.088,168.686c102.621,178.82,152.508,340.341,170.857,549.14   c14.027,159.611,124.551,239.911,158.223,409.915c21.908,110.606,49.25,226.635,47.013,324.01   c-3.304,143.779-49.269,419.993-174.464,441.982c96.516,21.854,220.154,37.618,304.754,90.247   c199.037,123.819,336.75,329.159,266.479,560.134c-103.856,341.34-337.229,610.193-560.424,880.842   c-183.617,222.651-410.791,430.394-677.031,550.955c-680.221,308.015-1326.973-5.006-2024.568,11.531   c-299.873,7.105-517.482-16.721-786.898-142.876c-118.914-55.688-215.523-84.235-345.721-105.013   c-251.304-40.1-443.702-203.442-577.426-413.314c-54.688-85.833-50.135-193.289-129.498-265.938   c-70.75-64.768-44.326-194.624,20.599-266.901c50.116-55.776,127.097-135.258,24.871-184.88   c-72.841-35.353-159.812-19.962-234.234,2.568c-42.647,12.912-99.439,29.135-126.929,67.372   C9404.609,5727.341,9444.971,5750.238,9467.215,5791.594z" style="stroke:black;fill:#bfdae3"/><path id="171453751" parent_id="1140195" order="5" structure_id="667" d="M2815.212,2659.0	c-41.795,92.794-60.946,187.111-62.889,281.487l449.915,18.282l-1.688-4.649L2815.212,2659.989z" style="stroke:black;fill:#268f45"/></svg>'
	node = pathToRoi(path)
	print(node)

	svgToRois(xmlPath)


### Converting: ImagePlus overlay --> svg

	# load the ImagePlus which contains an overlay
	imp = IJ.openImage(filePath.getAbsolutePath())
	# Extract the ROI list
	[rois, w, h] = roisFromImagePlusOverlay(imp)
	# Convert the ROIs to an svg string
	text = roisToSVG(rois, w, h)
	# Save the svg file
	writeTextFile(text, saveDir, 'svg', 'rois')

### Converting: ROI file --> svg

	# load a ROI file
	rois = roisFromFile(roiPath.getAbsolutePath())
	text = roisToSVG(rois, w, h)
	# Save the svg file
	writeTextFile(text, saveDir, 'svg', 'roisFromFile')

### Converting: ImagePlus overlay --> svg

### Converting: Evotec region --> ROI

### Converting: ROI --> Evotec region
	# Extract the ROI list
	[rois, w, h] = roisFromImagePlusOverlay(imp)
	# Convert the ROIs to an svg string
	Xlt = 0;
	Ylt = 0;
	Xrb = w;
	Yrb = h;
	text = roiToEvotecRegion(rois[0], w, h, Xlt, Ylt, Xrb, Yrb)
	# Save the svg file
	writeTextFile(text, saveDir, 'txt', 'rois_Evotec')

	[text1, text2] = pathCoordsToRoi(svgToRois(xmlPath))
	writeTextFile(text1, saveDir, 'txt', 'coords1')
	writeTextFile(text2, saveDir, 'txt', 'coords2')


run()

readSVG(xmlPath)
