import numpy as np
import matplotlib.pyplot as plt

from operator import add # Is there a better way?

class maze():

    # Initialize important Properties
    def __init__(self, dim):
        
        self.xsize = dim[0]
        self.ysize = dim[1]
        self.paths = []
        self.boundaries = []
        
        self.temp_len = 0
    

    def make_paths(self, turn_prob):
        starting_point = [0, np.random.randint(np.floor(self.ysize/4))]
        ending_point = [self.xsize-1, self.ysize-np.random.randint(np.floor(self.ysize/4))-1]
        occupied = np.zeros([self.xsize, self.ysize])
        occupied[starting_point[0], starting_point[1]] = 1
        
        direction = [0,1]
        pos = starting_point
        fails = 0
        done = False
        while(not done):
            save_dir = direction.copy()
            save_pos = pos.copy()
            
            # reset on failure
            if(fails > 10000):
                resets = np.ceil(len(self.paths)/4) # make input param

                for i in np.arange(resets):
                    idx = len(self.paths)-1
                    occupied[self.paths[idx][1][0], self.paths[idx][1][1]] = 0
                    self.paths.pop()
                
                idx = len(self.paths)-1
                pos = self.paths[idx][1]
                
                fails = 0
                print("Failure")
                continue
            
            left = np.random.binomial(1, turn_prob)
            right = np.random.binomial(1, turn_prob)
            if(left and right):
                continue
            
            elif(left):
                direction[0] = save_dir[1]
                direction[1] = save_dir[0]
                
                if(save_dir[1] == 0):
                    direction[1] = -1*direction[1]
                
            elif(right):
                direction[0] = save_dir[1]
                direction[1] = save_dir[0]
                
                if(save_dir[0] == 0):
                    direction[0] = -1*direction[0]
            
            pos = list(map(add, pos, direction))
            
            if(not self.check_legality(pos, occupied)):
                direction = save_dir.copy()
                pos = save_pos.copy()
                fails = fails + 1
                continue
            elif(self.is_trap(pos, occupied)):
                direction = save_dir.copy()
                pos = save_pos.copy()
                fails = fails + 1
                continue
            elif(self.is_once_removed_trap(pos, occupied)):
                direction = save_dir.copy()
                pos = save_pos.copy()
                fails = fails + 1
                continue
            else:
                occupied[pos[0], pos[1]] = 1
                self.paths.append([save_pos, pos])
            
            if pos == ending_point:
                done = True
                self.temp_len = len(self.paths)
        
        while(any(0 in sublist for sublist in occupied)):
            i = np.random.randint(self.xsize)
            j = np.random.randint(self.ysize)
            
            if occupied[i][j]:
                grow = self.check_growth([i,j], occupied)
                if grow:
                    idx = np.random.randint(len(grow))
                    ness = grow[idx]
                    occupied[ness[0], ness[1]] = 1
                    self.paths.append([[i,j], ness])
        
        print(self.paths)
        
    
    def check_growth(self, pos, occupied):
        moves = []
        
        north = list(map(add, pos, [0,1]))
        south = list(map(add, pos, [0,-1]))
        east = list(map(add, pos, [1,0]))
        west = list(map(add, pos, [-1,0]))
        
        head_north = self.check_legality(north, occupied)
        if(head_north): moves.append(north)
        
        head_south = self.check_legality(south, occupied)
        if(head_south): moves.append(south)
        
        head_east = self.check_legality(east, occupied)
        if(head_east): moves.append(east)
        
        head_west = self.check_legality(west, occupied)
        if(head_west): moves.append(west)
        
        return moves
        
    def check_legality(self, pos, occupied):
        if(pos[0] < 0 or pos[0] >= self.xsize or pos[1] < 0 or pos[1] >= self.ysize):
            return False
        elif(occupied[pos[0],pos[1]]):
            return False
        else:
            return True
        
    def is_trap(self, pos, occupied):
        north = list(map(add, pos, [0,1]))
        south = list(map(add, pos, [0,-1]))
        east = list(map(add, pos, [1,0]))
        west = list(map(add, pos, [-1,0]))
        
        head_north = self.check_legality(north, occupied)
        head_south = self.check_legality(south, occupied)
        head_east = self.check_legality(east, occupied)
        head_west = self.check_legality(west, occupied)
        
        if(head_north or head_south or head_east or head_west):
            return False
        else:
            return True
        
    def is_once_removed_trap(self, pos, occupied):
        north = list(map(add, pos, [0,1]))
        south = list(map(add, pos, [0,-1]))
        east = list(map(add, pos, [1,0]))
        west = list(map(add, pos, [-1,0]))
        
        trap_north = self.is_trap(north, occupied)
        trap_south = self.is_trap(south, occupied)
        trap_east = self.is_trap(east, occupied)
        trap_west = self.is_trap(west, occupied)
        
        if(trap_north or trap_south or trap_east or trap_west):
            return True
        else:
            return False

    def draw(self):
        idx = 0
        for path in self.paths:
            x = np.array(path)[:,0]
            y = np.array(path)[:,1]
            
            if idx < self.temp_len:
                plt.plot(x,y,'r')
            else:
                plt.plot(x,y,'k')
                
            idx = idx + 1
        
        plt.axis('off')
        plt.show()

if __name__ == "__main__":

    grid_size = [25,25]
    test = maze(grid_size)
    test.make_paths(0.25)
    test.draw()
    

    print("Success")
