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
import numpy as np
import pickle


RED = 0
GREEN = 1
BLUE = 2
filter = [[0, 1], [1, 2]] #R, G, G, B (horizontal first)


'''
_ REMOSAIC = PRETTY FAST
_ DEMOSAIC = FAST AND GOOD AND LOVELY NOW
_ DPCM_decode = LIGHTNING FAST
_ DPCM_encode = LOVELY HOW FAST THIS ONE IS


THOUGHTS:
	at top of loop
	check if it's been 5 mins
		if it has...
			write ys and write RGB OUT TO FILES in subprocess
		BREATHE TO MASK THIS.. who knows how long that'll take...


	need to prove clipglitch concept works by running in a loop and.. writing to winscp dir?
	before implementing breathing...!


	while frame isnt done
		breathe
	frame is now done..
		RGB we made needs to go into a slot (random?)
		ys needs to be overwritten with a new one we just made
			both of these should be atomic operations, since they're pre-built



	8 "frames"
	ping pong / breathing (with rate that changes every so often)



- every 5 minutes... ATOMIC WRITING https://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python

'''

def DPCM_encode(bayerData, w, h):
	delta_table = [-144, -110, -77, -53, -35, -21, -11, -3, 2, 10, 20, 34, 52, 76, 110, 144]
	tempLineRed = [0x80] * w
	tempLineGreen = [0x80] * w
	tempLineBlue = [0x80] * w
	differences = [] # * (w * h // 2)

	for y in range(h):
		if y % 2 == 0: #even row
			for x in range(0, w, 2): #go all the way across width, in chunks of 2
				#LEFT PIXEL (RED)
				if x == 0:
					prediction = tempLineRed[0] #just above...
				else:
					prediction = (tempLineRed[x] + tempLineRed[x-2]) // 2 #avg prediction = true prediction for this pixel

				dif = bayerData[y*w + x] - prediction #dif is actual - prediction
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
	redChannel = [] #*w*h
	finalTrueRed = h*w - 2
	for y in range(0, h-2, 2): #handle everything but final two ROWS (of RG, BG)
		yw = y*w
		#vars for use when determining GB row
		ywabove = y*w
		ywbelow = (y+2)*w

		for x in range(1, w-2, 2): #RG LINE
			redChannel.append(bayerData[yw + x-1]) #known val
			redChannel.append((bayerData[yw + x-1] + bayerData[yw + x+1]) //2) #not clamping.. because we know DPCM decode outputs come pre-clamped
		#final bayer cell
		redChannel.append(bayerData[finalTrueRed]) #append known red
		redChannel.append(bayerData[finalTrueRed]) #nearest neighbor for rightmost column red (fast/simple/probably not even on screen)

		for x in range(1, w, 2): #doesnt touch final bayer cell
			#do G (avg of known red pixels..)
			gval = (bayerData[ywabove + x-1] + bayerData[ywbelow + x-1]) //2
			redChannel.append(gval)
			#do B (avg of.... the R we just wrote for G bayerpixel, and... above (derived) val..? index into redChannel?
			bval = (redChannel[ywabove + x] + gval) // 2
			redChannel.append(bval)

	#do bottom RG line
	for x in range(1, w-2, 2): #go up to last column (can't average to right in there)
		redChannel.append(bayerData[yw + x-1]) #R
		redChannel.append((bayerData[yw + x-1] + bayerData[yw + x+1]) //2) #G
	redChannel.append(bayerData[finalTrueRed]) #true red
	redChannel.append(bayerData[finalTrueRed]) #nearest neighbor for green val

	#do bottom BG line... (just nearest neighbor)
	for x in range(w):
		redChannel.append(redChannel[(h-2)*w + x])

	return(redChannel)





def demosaicBlue(bayerData, w, h): #RG, GB*
	blueChannel = [] #*w*h

	GBLine = []
	GBLine.append(bayerData[w+1]) #do first GB-green (nearest neighbor to right)
	GBLine.append(bayerData[w+1]) #do first B (true)

	for x in range(2, w, 2): #handle whole first GB row, one bayer cell per loop
		GBLine.append((bayerData[w + x + 1] + bayerData[w + x - 1]) // 2) #avg left and right blues (for Gpixel)
		GBLine.append(bayerData[w + x + 1]) #write B (true)

	blueChannel.extend(GBLine)	#now, GB line is built.. nearest neighbor for very first RG line
	blueChannel.extend(GBLine)	#now append true GB line

	#RG lines are always encapsulated from here out
	for y in range(2, h, 2): #handle one ROW of cells (RG, GB)  in each loop
		yw = y*w #points to RG row
		ywabove = (y-1)*w
		ywbelow = (y+1)*w

		#start building GB row..
		GBLine = []
		GBLine.append(bayerData[yw]) #do first GB-green (nearest neighbor to right)
		GBLine.append(bayerData[yw]) #do first B (true)

		for x in range(2, w, 2): #finish building GB line
			GBLine.append((bayerData[ywbelow + x + 1] + bayerData[ywbelow + x - 1]) // 2) #avg left and right blues (for Gpixel)
			GBLine.append(bayerData[ywbelow + x + 1]) #write B (true)

		#we have a full GB line now.
		#get RG values (averages above and below), then append GBLine (greens, blues)
		for x in range(w): # y points to row, x points to start of cell in that row
			blueChannel.append((blueChannel[ywabove + x] + GBLine[x]) // 2) #average GB row with row that is written to blueChannel above to get RG row

		blueChannel.extend(GBLine)
	return blueChannel



def demosaicRG_Green(bayerData, w, h):
	RG_greenChannel = [] # * w #* (h//2)

	#do upper left corner (2 neighbors)
	RG_greenChannel.append((bayerData[1] + bayerData[w]) // 2)
	RG_greenChannel.append(bayerData[1])
	#do rest of first row (3 neigbors)
	for x in range(2, w, 2):
		RG_greenChannel.append((bayerData[x-1] + bayerData[x+1] + bayerData[w + x]) // 3) #approximate first row green (red b-pixels)
		RG_greenChannel.append(bayerData[x-1]) #append true green

	RG_greenChannel.extend([0b00000000] * w) # add empty row (GB)

	#do middle (4 neighbors... except for first col with 3)
	for y in range(2, h, 2):
		yw = y*w #points to starting index of missing RED
		ywabove = (y-1)*w
		ywbelow = (y+1)*w

		RG_greenChannel.append((bayerData[ywabove] + bayerData[ywbelow] + bayerData[yw +1]) // 3) #approximate first col green (red b-pixel)
		RG_greenChannel.append(bayerData[yw])

		for x in range(2, w, 2): #x points to missing red
			RG_greenChannel.append((bayerData[ywabove + x] + bayerData[ywbelow + x] + bayerData[yw + x - 1] + bayerData[yw + x + 1]) // 4)
			RG_greenChannel.append(bayerData[yw + x + 1])

		RG_greenChannel.extend([0b00000000] * w)

	return RG_greenChannel



def demosaicGB_Green(bayerData, w, h):
	GB_greenChannel = []
	hwm1 = h*(w-1)
	hwm2 = h*(w-2)

	GB_greenChannel.extend([0b00000000] * w)
	#go across... 4 neighbors until last column with 3
	for y in range(1, h-2, 2): #y points to GB row..
		yw = y*w
		ywabove = (y-1)*w
		ywbelow = (y+1)*w

		for x in range(1, w-2, 2): # 4 neighbors till last column
			#append true green
			GB_greenChannel.append(bayerData[yw + x-1])
			GB_greenChannel.append((bayerData[yw + x-1] + bayerData[yw + x + 1] + bayerData[ywabove + x] + bayerData[ywbelow + x]) // 4)

		#do last column
		GB_greenChannel.append(bayerData[yw + w -2]) #append true G
		GB_greenChannel.append(bayerData[yw + w -2]) #nearest neighbor for G val for Blue bayer pixel

		GB_greenChannel.extend([0b00000000] * w) # add empty row (RG)

	#handle last row
	#append true green (bottom left corner)
	GB_greenChannel.append(bayerData[hwm1])
	GB_greenChannel.append((bayerData[hwm1] + bayerData[hwm1 + 2] + bayerData[hwm2 + 1]) // 3) #left, right, up


	#bottom row.. 3 neighbors.. until last col with 2
	for x in range(2, w-2, 2): # x points to known green offset
		#GB_greenChannel.append(bayerData[hwm1 + x])
		#print(x)
		#GB_greenChannel.append((bayerData[hwm1 + x] + bayerData[hwm1 + x + 2] + bayerData[hwm2 + x]) // 3) #left, right, up
		GB_greenChannel.append(0b00000000)
		GB_greenChannel.append(0b00000000)
	#final known green
	GB_greenChannel.append(bayerData[h*w - 2])
	GB_greenChannel.append((bayerData[h*w - 2] + bayerData[hwm1 - 1]) // 2)

	return GB_greenChannel


def expand(bayerData, w, h):
	global filter
	expandedData = [0] * (len(bayerData) * 3)
	for y in range(h):
		for x in range(w):
			colorOffset = filter[y%2][x%2]
			i = (y * w + x) #current index in bayerData[]
			expandedData[(i*3)+colorOffset] = bayerData[i] #put into proper location in expandedData
	return expandedData


def clamp(x):
	if x > 255:
		return 255
	elif x < 0:
		return 0
	else:
		return x


def REmosaicArbitraryImage(RGBdata, w, h):
	global filter
	RED = 0
	GREEN = 1
	BLUE = 2
	output = []

	for y in range(h):
		for x in range(w*3):
			currentRGBcolor = x % 3
			desiredColor = filter[y%2][x%2] #(x%2) if (y%2 == 0) else (x%2 + 1)
			if currentRGBcolor == desiredColor:
				output.append(RGBdata[3*y*w + x]) #where y and w are widths in rgb data
	return output


def REmosaicPiCam(RGBdata, width, height):
	global filter
	output = []

	y = 0
	x = 0

	for row in RGBdata:
		for pixel in row:
			output.append(int(pixel[filter[y%2][x%2]]))
			x += 1
		y += 1

	return output


def postprocess(RGBdata, w, h):
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


#Now with parallelization!
def demosaic(executor, bayerData, w, h):
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

#REmosaic RGB data, then DPCM_encode it.
def makeEncodedFrame(RGB, w, h):
	bayer = REmosaicPiCam(RGB, w, h)
	encoded = DPCM_encode(bayer, w, h)

	return encoded

#unused.. but cool idea
def pingpong(index, direction, listLength):
	if index == listLength-1:
		return (listLength-2, 1)
	elif index == 0:
		return (1, 0)
	elif direction == 0:
		return (index+1, 0)
	elif direction == 1:
		return (index-1, 1)


def glitch(RGB, w, h, glitchmode):
	enc = makeEncodedFrame(RGB, w, h)
	if(glitchmode == 0):
		return clip(enc, w, h)
	elif(glitchmode == 1):
		return sandwich(enc, w, h)



#return one decoded-and-demosaiced RGB frame, the clip to be written to parent's global your_sanctuary (modding on end)
#...and the index where that starts
def clip(enc, w, h):
	innerExecutor = cf.ProcessPoolExecutor()

	#determine leftmost pixel (encoded) starting point in ys
	#also last pixel (bottom right) .. which is one full encoded image away
	ysStartOffset = random.randrange(0, len(your_sanctuary))
	ysEndOffset = ysStartOffset + (w*h//2) #wil have to use MOD with this to wrap around

	clipStartIndex = random.randrange(0, (w*h//2)) #in enc
	clipEndIndex = random.randrange(clipStartIndex, (w*h //2)) #safe.

	#put the clip into child's ys
	for i in range(clipStartIndex, clipEndIndex):
		your_sanctuary[(ysStartOffset + i) % len(your_sanctuary)] = enc[i]

	#build modified_ys_chunk_for_RGB
	modified_ys_chunk_for_RGB = []
	for i in range(ysStartOffset, ysEndOffset): #have to do a loop because of modding
		modified_ys_chunk_for_RGB.append(your_sanctuary[(ysStartOffset + i) % len(your_sanctuary)])

	#TOP ROW RANDOM
	for x in range(0, w//2, 5):
		modified_ys_chunk_for_RGB[x] = random.randrange(0, 0xFF)
	#LEFT COLUMN RANDOM
	for y in range(0, h, 5):
			modified_ys_chunk_for_RGB[(y*w//2)] = random.randrange(0, 0xFF)

	bayerData = DPCM_decode(modified_ys_chunk_for_RGB, w, h)
	RGBframe = demosaic(innerExecutor, bayerData, w, h)

	return bytearray(RGBframe), your_sanctuary


def sandwich(enc, w, h):
	innerExecutor = cf.ProcessPoolExecutor()

	ysStartOffset = random.randrange(0, len(your_sanctuary))
	ysEndOffset = ysStartOffset + (w*h//2) #might have to do some modding around

	sandwiched = [0] * len(enc)
	rotation = random.randrange(0,len(enc))
	enc = enc[-(rotation):] + enc[:-(rotation)]

	for y in range(0, h, 2):
		for x in range(0, w//2, 1):
			sandwiched[y*w//2 + x] = enc[y*w//2 + x]

	for y in range(1, h, 2):
		for x in range(0, w//2, 1):
			sandwiched[y*w//2 + x] = your_sanctuary[(ysStartOffset + y*w + x)%len(your_sanctuary)]

	#write back to ys
	for i in range(len(sandwiched)):
		your_sanctuary[(ysStartOffset + i)%len(your_sanctuary)] = sandwiched[i]

	bayerData = DPCM_decode(sandwiched, w, h)
	RGBframe = demosaic(innerExecutor, bayerData, w, h)

	return bytearray(RGBframe), your_sanctuary



#unused, but maybe worth revisiting someday
def breathePong(RGBframesList, w, h):
	#hold indexes of the elements we need to display
	RGBframeindexes = []
	pingpongindex = 0
	pingpongdirection = 0
	for i in range((len(RGBframesList)*2)-2):
		RGBframeindexes.append(pingpongindex)
		pingpongindex, pingpongdirection = pingpong(pingpongindex, pingpongdirection, len(RGBframesList))
	#hold sleep times for each of those references frames
	sleepTimes = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
	for i in range(len(RGBframeindexes)):
		frameIndex = RGBframeindexes[i]
		image = pygame.image.frombuffer(RGBframesList[frameIndex], (w,h), 'RGB')
		screen.blit(image, (0,0))
		pygame.display.flip()
		time.sleep(sleepTimes[i])


def breatheThru(RGBframesList, w, h, numBreaths):
	sleepTimes = [0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18]
	for breath in range(numBreaths):
		for i in range(len(RGBframesList)):
			image = pygame.image.frombuffer(RGBframesList[i], (w,h), 'RGB')
			screen.blit(image, (0,0))
			pygame.display.flip()
			time.sleep(sleepTimes[i])


if __name__ == "__main__":
	camera_warmup_time = 1.5
	w = 720
	h = 480
	glitchmode = 0 # 0 = clip, 1 = sandwich (not great)
	mood = 0 # if mood = 0, just show glitches as they're made rather than breathing

	print("Initializing camera...")
	camera = picamera.PiCamera()
	camera.resolution = (720, 480)
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
			quit()

	print("Warming up camera (" + str(camera_warmup_time) + " second(s))...")
	camera.capture(cameraStream, 'rgb')
	cameraStream.flush() # fill cameraStream
	print("Test capture successful. (" + str(cameraStream.array.shape[1]) + " x " + str(cameraStream.array.shape[0]) + ")")

	try:
		with open('your_sanctuary.RAW', 'rb') as tf:
			your_sanctuary = pickle.load(tf)
	except:
		print("Unable to load your_sanctuary bucket from file.")
		your_sanctuary = []
		for i in range((h * w // 2) * 10):
			your_sanctuary.append(random.randrange(0, 0xFF))

	try:
		with open('RGBframes.RAW', 'rb') as tf:
			RGBframesList = pickle.load(tf)
	except:
		print("Unable to load RGB frames from file.")
		RGBframesList = [bytearray([0b11111111] * (w * h * 3))] * 8

	#Hold completed RGB glitches
	#when writing breaths... always insert at 0, then... remove the one at 8 (9th element)

	executor = cf.ProcessPoolExecutor()

	lastWriteTime = time.time() #write every 5 minutes.
	lastMoodChange = time.time()

	print("Starting...\n")


	pygame.init()
	pygame.mouse.set_visible(False)

	screen = pygame.display.set_mode((w,h), pygame.FULLSCREEN)
	pygame.display.set_caption('YOUR SANCTUARY')



	while True:
		#if(time.time() - lastMoodChange >= 60): #if it's been 1 minute
		#	mood = not mood

		#check if we need to write to disk
		if(time.time() - lastWriteTime >= 300): #5 minutes = 5*60 = 300
			print("Writing your_sanctuary bucket to file.")
			with open('tempfile', 'wb') as tf:
				pickle.dump(your_sanctuary, tf)
			os.rename('tempfile', 'your_sanctuary.RAW')

			print("Writing RGB frames to file.")
			with open('tempfile', 'wb') as tf:
				pickle.dump(RGBframesList, tf)
			os.rename('tempfile', 'RGBframes.RAW')

		if mood == 0: #show em as they come
			''' IDEA.. WHAT IF WE INTRODUCED PAUSES RIGHT HERE IN THIS LOOP TO FEEL LIKE BREATHING..?'''
			''' NOTATE with font'''


			cameraStream.truncate(0)
			camera.capture(cameraStream, 'rgb')
			cameraStream.flush()
			RGBcapture = cameraStream.array #3-dimensional list
			RGBglitch, your_sanctuary = glitch(RGBcapture, w, h, glitchmode)

			#update these slots even though we aren't using them at the moment
			RGBframesList.insert(0, RGBglitch)
			RGBframesList.pop()

			image = pygame.image.frombuffer(bytearray(RGBglitch), (w,h), 'RGB')
			screen.blit(image, (0,0))
			pygame.display.flip()

			#check for keyboard interrupt
			events = pygame.event.get()
			for event in events:
				if event.key == pygame.K_c:
					print("Saving local variables to files.")

					with open('tempfile', 'wb') as tf:
						pickle.dump(your_sanctuary, tf)
					os.rename('tempfile', 'your_sanctuary.RAW')

					with open('tempfile', 'wb') as tf:
						pickle.dump(RGBframesList, tf)
					os.rename('tempfile', 'RGBframes.RAW')

					print("Quitting (Keyboard Interrupt).")
					pygame.quit()
					quit()

		#hold onto some capture frames, queue em up, insert newest, remove oldest
		else:
			#hideCaptureFuture = executor.submit(breatheThru, RGBframesList, w, h, 1)
			cameraStream.truncate(0)
			camera.capture(cameraStream, 'rgb')
			cameraStream.flush()
			RGBcapture = cameraStream.array #3-dimensional list

			#while(hideCaptureFuture.done() == False):
			#	pass #wait for these breaths to finish, so we don't hyperventilate
			#	time.sleep(0.18)

			makeGlitchFuture = executor.submit(glitch, RGBcapture, w, h, glitchmode)
			while(makeGlitchFuture.done() == False):
				breatheThru(RGBframesList,w,h,1)

			#Cool, we have a frame now. Mask some operations with more breath
			#hideWriteFuture = executor.submit(breatheThru, RGBframesList, w, h, 1)


			#during that last breath, move some stuff around
			RGBglitch, your_sanctuary = makeGlitchFuture.result()

			RGBframesList.insert(0, RGBglitch) #insert into beginning of frameslist
			RGBframesList.pop()	#remove last one

			events = pygame.event.get()
			for event in events:
				if event.key == pygame.K_c:
					print("Saving local variables to files.")

					with open('tempfile', 'wb') as tf:
						pickle.dump(your_sanctuary, tf)
					os.rename('tempfile', 'your_sanctuary.RAW')

					with open('tempfile', 'wb') as tf:
						pickle.dump(RGBframesList, tf)
					os.rename('tempfile', 'RGBframes.RAW')

					print("Quitting (Keyboard Interrupt).")
					pygame.quit()
					quit()

			#while(hideWriteFuture.done() == False):
			#	pass #wait for these breaths to finish, so we don't hyperventilate
			#time.sleep(0.18)



			'''
			# ATOMIC file writing
			tempfile = open("tempfile.data", "wb")
			for byte in RGBglitch:
				tempfile.write(byte.to_bytes(1, byteorder='big'))

			tempfile.flush()
			os.fsync(tempfile.fileno())
			tempfile.close()
			os.rename('tempfile.data', 'glitchin.data')
			'''
