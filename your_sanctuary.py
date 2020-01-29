from bisect import bisect_left
import os
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import time
import picamera
import picamera.array
import random
import concurrent.futures as cf
import pickle


RED = 0
GREEN = 1
BLUE = 2
filter = [[0, 1], [1, 2]] #R, G, G, B


def DPCM_encode(bayerData, w, h):
	'''
	Lossyly compress a frame of CFA/Bayer data (RGGB) to half of its original size using a differential compression algorithm.

	1. iterating over each Bayer-pixel, generate an intensity prediction
	   (average of previously generated values (predictions + quantized differences) for nearby (above and to the left) like-color pixels (when they exist))
	2. subtract this prediction from the true intensity at the current location
	2. quantize this difference to the nearest value in delta_table, storing the index of the quantized value
	   (note: since len(delta_table) = 16 = 2^4, every index can be stored in 4 bits - HALF of the original 8-bit intensity value)
	3. construct new 8-bit bytes by compounding pairs of 4-bit nybbles (left pixel, then right)

	4. return the resultant array of bytes, having length w*h/2 and being comprised of nybble-pairs representing indexes into delta_table
	'''
	delta_table = [-144, -110, -77, -53, -35, -21, -11, -3, 2, 10, 20, 34, 52, 76, 110, 144]
	tempLineRed = [0x80] * w
	tempLineGreen = [0x80] * w
	tempLineBlue = [0x80] * w
	differences = [] # hold indexes into delta_table, whose values represent differences

	for y in range(h):
		if y % 2 == 0: #even row
			for x in range(0, w, 2):
				#LEFT PIXEL (RED)
				if x == 0:
					prediction = tempLineRed[0]
				else:
					prediction = (tempLineRed[x] + tempLineRed[x-2]) // 2
				dif = bayerData[y*w + x] - prediction
				quantized_dif = quantize(delta_table, dif)
				leftNybble = quantized_dif << 4
				tempLineRed[x] = clamp(prediction + delta_table[quantized_dif])

				#RIGHT PIXEL (GREEN)
				if x == 0:
					prediction = tempLineGreen[0]
				else:
					prediction = (tempLineGreen[x] + tempLineGreen[x-2]) // 2

				dif = bayerData[y*w + x + 1] - prediction
				quantized_dif = quantize(delta_table, dif)
				rightNybble = quantized_dif
				tempLineGreen[x] = clamp(prediction + delta_table[quantized_dif])

				#Create nybble-pair
				nybblePair = leftNybble | rightNybble
				#APPEND NYBBLE-PAIR
				differences.append(nybblePair)

		else:	#odd row
			for x in range(0, w, 2):
				#LEFT PIXEL (GREEN)
				if x == 0:
					prediction = tempLineGreen[0]
				else:
					prediction = (tempLineGreen[x] + tempLineGreen[x-2]) // 2

				dif = bayerData[y*w + x] - prediction

				quantized_dif = quantize(delta_table, dif)
				leftNybble = quantized_dif << 4
				tempLineGreen[x] = clamp(prediction + delta_table[quantized_dif])

				#RIGHT PIXEL (BLUE)
				if x == 0:
					prediction = tempLineBlue[0]
				else:
					prediction = (tempLineBlue[x] + tempLineBlue[x-2]) // 2

				dif = bayerData[y*w + x + 1] - prediction
				quantized_dif = quantize(delta_table, dif)
				rightNybble = quantized_dif
				tempLineBlue[x] = clamp(prediction + delta_table[quantized_dif])

				#Create nybble-pair
				nybblePair = leftNybble | rightNybble
				#APPEND NYBBLE-PAIR
				differences.append(nybblePair)

	return differences


def DPCM_decode(differenceIndexes, w, h):
	'''
	Decompress a compressed CFA/Bayer data frame back to its original size (2x compressed size).

	1. split bytes into left and right nybbles (each representing a delta_table index)
	2. iterating over each location in the resultant frame, generate an intensity prediction
	   (average of previously generated values (predictions + quantized differences) for nearby (above and to the left) like-color pixels (when they exist))
	3. add the delta_table value at the index held by the current nybble to the generated intensity prediction
	4. clamp this value to the range 0-255 (a positive 8-bit number)

	5. return the resultant array of bytes, having length w*h and being comprised of 8-bit values representing raw CFA/Bayer data (RGGB)
	'''
	delta_table = [-144, -110, -77, -53, -35, -21, -11, -3, 2, 10, 20, 34, 52, 76, 110, 144]
	halfw = w//2
	tempLineRed = [0x80] * w
	tempLineGreen = [0x80] * w
	tempLineBlue = [0x80] * w

	uncomp = [] # * w * h

	for y in range(h):
		if y % 2 == 0:
			for x in range(halfw): #so for each one of these steps, handle a nybble-pair
				Nybblepair = differenceIndexes[y*w//2 + x]
				leftNybble = Nybblepair >> 4
				rightNybble = Nybblepair & 0x0f

				#LEFT PIXEL (RED)
				if x == 0:
					prediction = tempLineRed[0]
				else:
					prediction = (tempLineRed[2*x] + tempLineRed[2*(x-1)]) //2
				dif = delta_table[leftNybble]
				predictionPlusDif = prediction + dif
				clamped = clamp(predictionPlusDif)
				tempLineRed[2*x] = clamped
				uncomp.append(clamped)

				#RIGHT PIXEL (GREEN)
				if x == 0:
					prediction = tempLineGreen[0]
				else:
					prediction = (tempLineGreen[2*x] + tempLineGreen[2*(x-1)]) //2
				dif = delta_table[rightNybble]
				predictionPlusDif = prediction + dif
				clamped = clamp(predictionPlusDif)
				tempLineGreen[2*x] = clamped
				uncomp.append(clamped)

		else: #y (row) is an ODD number
			for x in range(halfw):
				Nybblepair = differenceIndexes[y*w//2 + x]
				leftNybble = Nybblepair >> 4
				rightNybble = Nybblepair & 0x0f

				#LEFT PIXEL (GREEN)
				if x == 0:
					prediction = tempLineGreen[0]
				else:
					prediction = (tempLineGreen[2*x] + tempLineGreen[2*(x-1)]) // 2
				dif = delta_table[leftNybble]
				predictionPlusDif = prediction + dif
				clamped = clamp(predictionPlusDif)
				tempLineGreen[2*x] = clamped
				uncomp.append(clamped)

				#RIGHT PIXEL (BLUE)
				if x == 0:
					prediction = tempLineBlue[0]
				else:
					prediction = (tempLineBlue[2*x] + tempLineBlue[2*(x-1)]) //2
				dif = delta_table[rightNybble]
				predictionPlusDif = prediction + dif
				clamped = clamp(predictionPlusDif)
				tempLineBlue[2*x] = clamped
				uncomp.append(clamped)

	return uncomp


def quantize(myList, val):
	'''
	Quantize 'val' to the nearest value in myList. Used in DPCM_encode.
	'''
	#assumes myList is sorted.
	pos = bisect_left(myList, val)
	if pos == 0:
		return 0
	if pos == len(myList):
		return len(myList) - 1
	before = myList[pos - 1]
	after = myList[pos]
	if after - val < val - before:
	   return pos
	else:
	   return pos - 1


def demosaicRed(bayerData, w, h):
	'''
	Use known red values to interpolate missing red values.
	'''
	redChannel = [] #*w*h
	finalTrueRed = h*w - 2
	for y in range(0, h-2, 2): #handle everything but final two ROWS (of RG, BG)
		yw = y*w
		ywabove = y*w
		ywbelow = (y+2)*w
		for x in range(1, w-2, 2):
			redChannel.append(bayerData[yw + x-1])
			redChannel.append((bayerData[yw + x-1] + bayerData[yw + x+1]) //2)
		redChannel.append(bayerData[finalTrueRed])
		redChannel.append(bayerData[finalTrueRed])
		for x in range(1, w, 2):
			gval = (bayerData[ywabove + x-1] + bayerData[ywbelow + x-1]) //2
			redChannel.append(gval)
			bval = (redChannel[ywabove + x] + gval) // 2
			redChannel.append(bval)
	for x in range(1, w-2, 2):
		redChannel.append(bayerData[yw + x-1])
		redChannel.append((bayerData[yw + x-1] + bayerData[yw + x+1]) //2)
	redChannel.append(bayerData[finalTrueRed])
	redChannel.append(bayerData[finalTrueRed])
	for x in range(w):
		redChannel.append(redChannel[(h-2)*w + x])
	return(redChannel)


def demosaicBlue(bayerData, w, h):
	'''
	Use known blue values to interpolate missing blue values.
	'''
	blueChannel = []
	GBLine = []
	GBLine.append(bayerData[w+1])
	GBLine.append(bayerData[w+1])
	for x in range(2, w, 2):
		GBLine.append((bayerData[w + x + 1] + bayerData[w + x - 1]) // 2)
		GBLine.append(bayerData[w + x + 1])
	blueChannel.extend(GBLine)
	blueChannel.extend(GBLine)
	for y in range(2, h, 2):
		yw = y*w
		ywabove = (y-1)*w
		ywbelow = (y+1)*w
		GBLine = []
		GBLine.append(bayerData[yw])
		GBLine.append(bayerData[yw])
		for x in range(2, w, 2):
			GBLine.append((bayerData[ywbelow + x + 1] + bayerData[ywbelow + x - 1]) // 2)
			GBLine.append(bayerData[ywbelow + x + 1])
		for x in range(w):
			blueChannel.append((blueChannel[ywabove + x] + GBLine[x]) // 2)
		blueChannel.extend(GBLine)
	return blueChannel


def demosaicRG_Green(bayerData, w, h):
	'''
	Use known even row green values to interpolate missing even row green values.
	'''
	RG_greenChannel = []
	RG_greenChannel.append((bayerData[1] + bayerData[w]) // 2)
	RG_greenChannel.append(bayerData[1])
	for x in range(2, w, 2):
		RG_greenChannel.append((bayerData[x-1] + bayerData[x+1] + bayerData[w + x]) // 3)
		RG_greenChannel.append(bayerData[x-1])
	RG_greenChannel.extend([0b00000000] * w)
	for y in range(2, h, 2):
		yw = y*w
		ywabove = (y-1)*w
		ywbelow = (y+1)*w
		RG_greenChannel.append((bayerData[ywabove] + bayerData[ywbelow] + bayerData[yw +1]) // 3)
		RG_greenChannel.append(bayerData[yw])
		for x in range(2, w, 2):
			RG_greenChannel.append((bayerData[ywabove + x] + bayerData[ywbelow + x] + bayerData[yw + x - 1] + bayerData[yw + x + 1]) // 4)
			RG_greenChannel.append(bayerData[yw + x + 1])
		RG_greenChannel.extend([0b00000000] * w)
	return RG_greenChannel


def demosaicGB_Green(bayerData, w, h):
	'''
	Use known odd row green values to interpolate missing odd row green values.
	'''
	GB_greenChannel = []
	hwm1 = h*(w-1)
	hwm2 = h*(w-2)
	GB_greenChannel.extend([0b00000000] * w)
	for y in range(1, h-2, 2):
		yw = y*w
		ywabove = (y-1)*w
		ywbelow = (y+1)*w
		for x in range(1, w-2, 2):
			GB_greenChannel.append(bayerData[yw + x-1])
			GB_greenChannel.append((bayerData[yw + x-1] + bayerData[yw + x + 1] + bayerData[ywabove + x] + bayerData[ywbelow + x]) // 4)
		GB_greenChannel.append(bayerData[yw + w -2])
		GB_greenChannel.append(bayerData[yw + w -2])
		GB_greenChannel.extend([0b00000000] * w)
	GB_greenChannel.append(bayerData[hwm1])
	GB_greenChannel.append((bayerData[hwm1] + bayerData[hwm1 + 2] + bayerData[hwm2 + 1]) // 3)
	for x in range(2, w-2, 2):
		GB_greenChannel.append(0b00000000)
		GB_greenChannel.append(0b00000000)
	GB_greenChannel.append(bayerData[h*w - 2])
	GB_greenChannel.append((bayerData[h*w - 2] + bayerData[hwm1 - 1]) // 2)
	return GB_greenChannel


def expand(bayerData, w, h):
	'''
	Inflate raw Bayer/CFA data to raw RGB format (3 bytes per pixel, 1 for each color).
	Only used for testing.
	'''
	global filter
	expandedData = [0] * (len(bayerData) * 3)
	for y in range(h):
		for x in range(w):
			colorOffset = filter[y%2][x%2]
			i = (y * w + x)
			expandedData[(i*3)+colorOffset] = bayerData[i]
	return expandedData


def clamp(x):
	'''
	Clamp arbitrary values to an 8-bit range (used in DPCM_encode and DPCM_decode).
	'''
	if x > 255:
		return 255
	elif x < 0:
		return 0
	else:
		return x


def REmosaicArbitraryImage(RGBdata, w, h):
	'''
	Given an arbitrary RGB image, discard 2/3 of input data to create artificial CFA/Bayer data (used for testing).
	'''
	global filter
	RED = 0
	GREEN = 1
	BLUE = 2
	output = []

	for y in range(h):
		for x in range(w*3):
			currentRGBcolor = x % 3
			desiredColor = filter[y%2][x%2]
			if currentRGBcolor == desiredColor:
				output.append(RGBdata[3*y*w + x])
	return output


def REmosaicPiCam(RGBdata, w, h):
	'''
	Given a PiCamera RGB frame, discard 2/3 of input data to create artificial CFA/Bayer data.
	'''
	output = []
	for y in range(0, h, 2):
		for x in range(0, w, 2): #RG row
			output.append(int(RGBdata[y][x][0])) #R
			output.append(int(RGBdata[y][x+1][1])) #G

		for x in range(0, w, 2): #GB row
			output.append(int(RGBdata[y][x][1])) #G
			output.append(int(RGBdata[y][x][2])) #B

	return output


def postprocess(RGBdata, w, h):
	'''
	Normalize brightness and contrast to make images 'punchier' (unused).
	Adapted from libgphoto2.
	'''
	red_min = 255
	red_max = 0
	blue_min = 255
	blue_max = 0
	green_min = 255
	green_max = 0

	for y in range(h):
		for x in range(w):
			#determine min and max for each color
			red_min = min(red_min, RGBdata[y*w*3+(x*3) + RED])
			red_max = max(red_max, RGBdata[y*w*3+(x*3) + RED])
			green_min = min(green_min, RGBdata[y*w*3+(x*3) + GREEN])
			green_max = max(green_max, RGBdata[y*w*3+(x*3) + GREEN])
			blue_min = min(blue_min, RGBdata[y*w*3+(x*3) + BLUE])
			blue_max = max(blue_max, RGBdata[y*w*3+(x*3) + BLUE])

	mini = min(min(red_min, blue_min), green_min)

	maxi = max(max(red_max, blue_max), green_max)
	amplify = float(255.0)/(maxi-mini)
	for i in range(w*h*3):
		RGBdata[i] = int(min(amplify * (RGBdata[i] - mini), 255))
	return RGBdata


def demosaic(executor, bayerData, w, h):
	'''
	Convert a frame of raw Bayer/CFA data to RGB data.
	Paralellized for a (very) small performance increase.
	'''
	redFuture = executor.submit(demosaicRed, bayerData, w, h)
	RG_greenFuture = executor.submit(demosaicRG_Green, bayerData, w, h)
	GB_greenFuture = executor.submit(demosaicGB_Green, bayerData, w, h)
	blueFuture = executor.submit(demosaicBlue, bayerData, w, h)

	while(redFuture.done() == False or blueFuture.done() == False or RG_greenFuture.done() == False or GB_greenFuture.done() == False):
		pass

	demoR = redFuture.result()
	demoRG = RG_greenFuture.result()
	demoGB = GB_greenFuture.result()
	demoB = blueFuture.result()
	out = []

	for y in range(0, h, 2):
		yw = y*w
		y1w = (y+1)*w
		for x in range(w):
			ywpx = yw + x
			out.append(demoR[ywpx])
			out.append(demoRG[ywpx])
			out.append(demoB[ywpx])
		for x in range(w):
			y1wpx = y1w + x
			out.append(demoR[y1wpx])
			out.append(demoGB[y1wpx])
			out.append(demoB[y1wpx])

	return out

if __name__ == "__main__":
	'''
	1. Initialize camera
	2. Unpickle your_sanctuary data to a list

	3. Begin main loop...
	   a. Capture a single RGB frame from PiCamera
	   b. (Lossyly) remosaic that frame to raw Bayer/CFA data
	   c. DPCM compress that data
	   d. Randomly select a location in your_sanctuary representing the start of a compressed frame
	      (to later be decompressed, demosaiced, and displayed on-screen)
	   e. Clip a contiguous chunk of compressed data (between 0 and w*h/2 bytes) from the most recently compressed frame (see c above)
	   f. Write that clip to your_sanctuary (specifically, within the boundaries of the frame that has been selected for display),
	      overwriting the existing data at that location
	   g. Introduce error into topmost row and leftmost column to encourage colorful, streaking artifacts
	   h. Decompress the (now modified) frame (of length w*h/2) beginning at the location randomly selected in (d) and looping around the end of your_sanctuary
	          Note: This is where the 'glitch' effect occurs. As prediction deltas are no longer in their original context,
			  and since the algorithm depends on previously calculated values, 'errors' manifest as colorful, streaking artifacts
			  traveling down and to the right.
	   i. Demosaic that frame of decompressed Bayer/CFA data to raw RGB data
	   j. Display that RGB data as an image on-screen
	'''

	w = 720
	h = 480
	ys_size = 100 # number of "full, encoded images" ys could hold
	camera_warmup_time = 1.5

	print("Initializing camera...")
	camera = picamera.PiCamera()
	camera.resolution = (w, h)
	cameraStream = picamera.array.PiRGBArray(camera)

	try:
		import camera_settings #local file
		camera.brightness = camera_settings.brightness
		print("Set camera brightness to " + str(camera_settings.brightness) + ".")
		camera.contrast = camera_settings.contrast
		print("Set camera contrast to " + str(camera_settings.contrast) + ".")

	except:
		print("Failed to load camera settings from file.")
		print("Creating default settings file...")
		f = open("camera_settings.py", "w")
		f.write("brightness = 66\ncontrast = 51")
		f.close()
		try:
			import camera_settings #local file
			camera.brightness = camera_settings.brightness
			print("Set camera brightness to " + str(camera_settings.brightness) + ".")
			camera.contrast = camera_settings.contrast
			print("Set camera contrast to " + str(camera_settings.contrast) + ".")
		except:
			print("Failed to create camera settings file.")
			print("Exiting.")
			quit()

	print("Warming up camera (" + str(camera_warmup_time) + " second(s))...")
	camera.capture(cameraStream, 'rgb')
	cameraStream.flush() # fill cameraStream
	print("Test capture successful. (" + str(cameraStream.array.shape[1]) + " x " + str(cameraStream.array.shape[0]) + ")")

	try:
		with open('your_sanctuary.data', 'rb') as tf:
			your_sanctuary = pickle.load(tf)
			print("Loaded your_sanctuary bucket from file.")
	except:
		print("Unable to load your_sanctuary bucket from file.")
		your_sanctuary = []
		for i in range((h * w // 2) * ys_size):
			print("Initializing new your_sanctuary bucket {:.2f}".format(i/(h*w//2*ys_size)*100) + "% complete.",end ="\r")
			your_sanctuary.append(random.randrange(0, 0xFF))
		print("Initialized your_sanctuary bucket with random values.")

	print("Starting...\n")

	executor = cf.ProcessPoolExecutor()
	lastWriteTime = time.time()

	pygame.init()
	pygame.mouse.set_visible(False)
	screen = pygame.display.set_mode((w,h), pygame.FULLSCREEN)
	pygame.display.set_caption('YOUR SANCTUARY')

	while True:

		#check if we need to write to disk
		if(time.time() - lastWriteTime >= 3600): #once every hour
			print("Writing your_sanctuary bucket to file.")
			with open('tempfile', 'wb') as tf:
				pickle.dump(your_sanctuary, tf)
				tf.flush()
				os.fsync(tf.fileno())
				tf.close()
			os.rename('tempfile', 'your_sanctuary.data')
			lastWriteTime = time.time()

		truestart = time.time()
		cameraStream.truncate(0) #clear out camera stream
		camera.capture(cameraStream, 'rgb')
		cameraStream.flush()
		RGBcapture = cameraStream.array #3-dimensional list
		print('Captured: ' + str(time.time() - truestart))
		start = time.time()
		bayer = REmosaicPiCam(RGBcapture, w, h)
		print('Remosaiced: ' + str(time.time() - start))
		start = time.time()
		enc = DPCM_encode(bayer, w, h)
		print('Encoded: ' + str(time.time() - start))

		start = time.time()
		ysStartOffset = random.randrange(0, len(your_sanctuary))
		ysEndOffset = ysStartOffset + (w*h//2)


		clipStartIndex = random.randrange(0, (w*h//2))
		clipEndIndex = random.randrange(clipStartIndex, (w*h//2))

		#insert clip into ys
		for i in range(clipStartIndex, clipEndIndex):
			your_sanctuary[(ysStartOffset + i) % len(your_sanctuary)] = enc[i]

		chunkToDisplay = []
		for i in range(w*h//2):
			chunkToDisplay.append(your_sanctuary[(ysStartOffset + i) % len(your_sanctuary)])
		print('Build encoded glitch chunk: ' + str(time.time() - start))

		#add some top edge randomness
		for x in range(0, w//2, 3):
			chunkToDisplay[x] = random.randrange(0, 0xFF)

		#add some left edge randomness
		for y in range(40):
			randIndex = random.randrange(0, h)
			chunkToDisplay[(randIndex*w//2)] = random.randrange(0, 0xFF)

		start = time.time()
		bayerData = DPCM_decode(chunkToDisplay, w, h)
		print('Decoded: ' + str(time.time() - start))
		start = time.time()
		RGBglitch = demosaic(executor, bayerData, w, h)
		print('Demosaiced: ' + str(time.time() - start))

		print('Total: ' + str(time.time() - truestart))

		image = pygame.image.frombuffer(bytearray(RGBglitch), (w,h), 'RGB')
		screen.blit(image, (0,0))
		pygame.display.flip()

		#check for keyboard interrupt
		events = pygame.event.get()
		for event in events:
			if event.key == pygame.K_c:
				print("Saving your_sanctuary bucket to file.")

				with open('tempfile', 'wb') as tf:
					pickle.dump(your_sanctuary, tf)
					tf.flush()
					os.fsync(tf.fileno())
					tf.close()
				os.rename('tempfile', 'your_sanctuary.data')

				print("Exiting (Keyboard Interrupt).")
				pygame.quit()
				quit()
