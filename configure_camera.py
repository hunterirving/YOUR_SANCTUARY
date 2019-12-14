import picamera, sys
import pygame.locals
import pygame

brightness = 50
contrast = 0
camera = picamera.PiCamera()

camera.start_preview()

pygame.init()
windowSurfaceObj = pygame.display.set_mode((0,0),1,16)

while True:
	events = pygame.event.get()
	for event in events:
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_a:
				if brightness != 0:
					brightness -= 1
			elif event.key == pygame.K_d:
				if brightness != 100:
					brightness += 1
			elif event.key == pygame.K_w:
				if contrast != 100:
					contrast += 1
			elif event.key == pygame.K_s:
				if contrast != 0:
					contrast -= 1
			elif event.key == pygame.K_c:
				f = open("camera_settings.py", "w")
				f.write("brightness = " + str(brightness) + "\ncontrast = " + str(contrast))
				f.close()
				pygame.quit()
				quit()

	camera.brightness = brightness
	camera.contrast = contrast
	camera.annotate_text = "YOUR SANCTUARY\nBRIGHTNESS: " + str(brightness) + "\nCONTRAST: " + str(contrast)
