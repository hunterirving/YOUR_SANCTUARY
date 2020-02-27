import random
import time
import math

'''
In the final version of Your Sanctuary, a codebook inherited from libgphoto2 was used.
Theoretically, an optimal codebook would be "tuned" to the camera that it operates with.
This script represents an (incomplete) attempt to generate such a codebook (for my purpose, it turned out to be not so important).
'''

codebookSize = 16 #16 = 2^4 = indexes representable with 4 bits (as opposed to 8 bits for raw RGB)
codebook = [] * codebookSize
numIterations = 100 #bigger is better (with diminishing returns)
completedIterations = 0
w = 720
h = 480
avgIterationTime = 0.00
start = time.time()
outputFilename = "codebook.py"

#simulate difference gathering
def simulateDifs(w, h):
	output = []
	for i in range(w * h):
		output.append(random.randrange(-256, 256)) #theoretical pre-quantized "differences-from-predictions". In reality should be: capture, predict, determine difs (without quantizing)
	return output


#Get an RGB frame, remosaic to Bayer data, generate predictions, get differences from true values, but don't quantize_difs.. instead return a size (w*h) 1-dimensional list of these pre-quantized "differences-from-predictions"
def getDifs():
	#TODO actually get real differences using camera
	pass

def writeToFile(outfileName, codebook):
	outFile = open(outfileName, "w")
	print("\nWriting to \"" + outfileName + "\".")
	outFile.write("codebook = " + str(codebook))
	print("Write complete.")

def main():
	global completedIterations

	for k in range(numIterations):
		iterationStart = time.time()
		difs = simulateDifs(720, 480)
		difs.sort() #arrange the differences in ascending order

		chunkSize = w * h // codebookSize
		tempCodeBook = []

		#break the (now ordered) data into chunks
		#determine the average difference for each chunk
		for i in range(codebookSize):
			accumulator = 0
			for j in range(chunkSize):
				accumulator += difs[i*chunkSize + j]
			tempCodeBook.append(accumulator // chunkSize)

		if completedIterations == 0:
			codebook = tempCodeBook
			avgIterationTime = (time.time() - iterationStart)

		else:
			for i in range(codebookSize):
				codebook[i] = ((codebook[i]*completedIterations) + tempCodeBook[i]) // (completedIterations + 1) #determine weighted averages for each chunk
				avgIterationTime = ((avgIterationTime * completedIterations) + (time.time() - iterationStart)) / (completedIterations + 1) #determine average iteration execution time

		completedIterations += 1

		print("Completed iteration " + str(completedIterations) + " of " + str(numIterations) + " (" + str(round(completedIterations/numIterations*100, 2)) + "% complete. Estimated time remaining: " + str(math.floor(avgIterationTime * (numIterations - completedIterations) / 60)) + " minutes, " + str(math.ceil(avgIterationTime * (numIterations - completedIterations) % 60)) + " seconds)")

	print("\nAfter " + str(numIterations) + " iterations... (" + str(math.floor((time.time() - start) / 60)) + " minutes, " + str(math.ceil((time.time() - start) % 60)) + " seconds)")
	print(codebook)

	writeToFile("codebook.py", codebook)
	quit()

main()
