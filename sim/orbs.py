import pygame
import numpy as np

class universe():

	def __init__(self, xsize, ysize):
		self.xsize = xsize
		self.ysize = ysize
		self.orbs = []
		
	def add_orb(self, x0, y0, varx, vary, r):
		self.orbs.append(orb(x0, y0, varx, vary, r))
	
	def tick_orbs(self):
		for orb in self.orbs:
			orb.tick()
			if (orb.x + orb.radius) > self.xsize: 
				orb.x = self.xsize - orb.radius
				orb.accx = 0
				orb.velx = -orb.velx
			if (orb.x - orb.radius) < 0: 
				orb.x = orb.radius
				orb.accx = 0
				orb.velx = -orb.velx
			if (orb.y + orb.radius) > self.ysize: 
				orb.y = self.ysize - orb.radius
				orb.accy = 0
				orb.vely = -orb.vely
			if (orb.y - orb.radius) < 0: 
				orb.y = orb.radius
				orb.accy = 0
				orb.vely = -orb.vely
			
	def draw_orbs(self, window):
		for orb in self.orbs:
			pygame.draw.circle(window, orb.color, (round(orb.x-orb.radius), round(orb.y-orb.radius)), round(orb.radius))
	
	def run(self):
		run = True
		pygame.init()
		pygame.display.set_caption("orbs")
		window = pygame.display.set_mode((self.xsize, self.ysize))
		while run:
			pygame.time.delay(10)
			pygame.draw.rect(window, (0, 0, 0), (0, 0, self.xsize, self.ysize))
			self.tick_orbs()
			self.draw_orbs(window)
			pygame.display.update()
			for event in pygame.event.get():
				if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
					run = False

		pygame.quit()

class orb():

	def __init__(self, x0, y0, varx, vary, r):
		self.x = x0
		self.y = y0
		self.velx = 0
		self.vely = 0
		self.accx = 0
		self.accy = 0
		self.varx = varx
		self.vary = vary
		self.radius = r
		self.color = (round(255*np.random.rand()), round(255*np.random.rand()), round(255*np.random.rand()))
		
	def tick(self):
		self.random_acc()
		self.velx = self.velx + self.accx
		self.vely = self.vely + self.accy
		self.x = self.x + self.velx
		self.y = self.y + self.vely
		
	def random_acc(self):
		self.accx = np.random.normal(0, self.varx)
		self.accy = np.random.normal(0, self.vary)
		if self.accx < -0.5: self.accx = -0.5
		if self.accx > 0.5: self.accx = 0.5
		if self.accy < -0.5: self.accy = -0.5
		if self.accy > 0.5: self.accy = 0.5
	
if __name__ == "__main__":
	onett = universe(1000, 850)
	for i in np.arange(2000):
		onett.add_orb(500, 425, 0.01, 0.01, 5)
	onett.run()
		