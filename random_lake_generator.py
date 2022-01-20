import numpy as np
import cv2
import os

from utils import saveDict

class random_lake_generator:
    def __init__(self, iterations=8, dir_name="gen", save_grids=False):
        self.grid_size = 3
        self.iterations = iterations
        self.con = np.arange(1,10).reshape([3,3])
        self.con[1,1] = 0
        self.dir_name = dir_name
        self.save_grids = save_grids
        os.makedirs(self.dir_name, exist_ok=True)

    def iterate(self, start=0, loc=[1,1]):
        """
        Iteratively creates grids.
        """
        not_passed = True
        while not_passed:
            try:
                grids = []
                contours = []
                grids = self.generateGrids(grids, start = start, loc = loc)
                for i in range(start,self.iterations-1,1):
                    grids[i+1], cnt = self.populateGrid(grids[i],grids[i+1])
                    contours.append(cnt)
                contours.append(self.sortGrid(grids[i+1]))
                if np.linalg.norm(contours[-1][0]-contours[-1][-1]) < 1.5:
                    not_passed = False
            except Exception as e: print(e)
        return grids, contours

    def run(self):
        """
        Generates a main lake with a set of islands in it.
        """
        # Generate main lake
        print('Generating lake')
        self.main_grd, self.main_cnt = self.iterate()
        # Generate grade 1 islands, max: 2
        print('Generating large islands: maximum 2')
        locs = self.lookForEmptySpots([self.main_grd[1]], [self.main_cnt[1]], max_spots=2)
        self.g1_isl_grd = []
        self.g1_isl_cnt = []
        for loc in locs:
            tmp_grd, tmp_cnt = self.iterate(start=1, loc=loc)
            self.g1_isl_grd.append(tmp_grd)
            self.g1_isl_cnt.append(tmp_cnt)
        tmp_grd = None
        tmp_cnt = None 
        # Generate grade 2 islands, max: 4
        print('Generating medium islands: maximum 4')
        locs = self.lookForEmptySpots([self.main_grd[2]] +  [i[2] for i in self.g1_isl_grd],
                                            [self.main_cnt[2]] + [i[1] for i in self.g1_isl_cnt],
                                             max_spots = 4)
        self.g2_isl_grd = []
        self.g2_isl_cnt = []
        for loc in locs:
            tmp_grd, tmp_cnt = self.iterate(start=2, loc=loc)
            self.g2_isl_grd.append(tmp_grd)
            self.g2_isl_cnt.append(tmp_cnt)
        tmp_grd = None
        tmp_cnt = None

        # Generate grade 3 islands, max: 8
        print('Generating small islands: maximum 8')
        locs = self.lookForEmptySpots([self.main_grd[3]] +  [i[3] for i in self.g1_isl_grd] + [i[3] for i in self.g2_isl_grd],
                                            [self.main_cnt[3]] + [i[2] for i in self.g1_isl_cnt] + [i[1] for i in self.g2_isl_cnt],
                                             max_spots = 8)
        self.g3_isl_grd = []
        self.g3_isl_cnt = []
        for loc in locs:
            tmp_grd, tmp_cnt = self.iterate(start=3, loc=loc)
            self.g3_isl_grd.append(tmp_grd)
            self.g3_isl_cnt.append(tmp_cnt)
        tmp_grd = None
        tmp_cnt = None 
        
        # Aggregate
        dct_contours = {}
        dct_contours['main'] = [self.main_cnt[-1]]
        dct_contours['1st_grade'] = [i[-1] for i in self.g1_isl_cnt]
        dct_contours['2nd_grade'] = [i[-1] for i in self.g2_isl_cnt]
        dct_contours['3rd_grade'] = [i[-1] for i in self.g3_isl_cnt]
        saveDict(dct_contours, os.path.join(self.dir_name,'contours'))
        if self.save_grids:
            self.saveGrids()
        
    def saveGrids(self):
        """
        Merges and saves the grids.
        """
        for i in range(self.iterations):
            grd = self.main_grd[i]
            for j in [self.g1_isl_grd, self.g2_isl_grd, self.g3_isl_grd]:
                for k in j:
                    grd += k[i]
            cv2.imwrite(os.path.join(self.dir_name,"grid_step_"+str(i)+".png"),grd*255)

    def lookForEmptySpots(self, grids, contours, max_spots = 1):
        """
        Looks for spots to generate islands in.
        """
        kernel = np.ones((3,3))
        loc_list = []
        lcl_grd = np.sum(grids,axis=0)
        cnts = [np.array([[0,0],
                        [0,lcl_grd.shape[1]],
                        [lcl_grd.shape[0],lcl_grd.shape[1]],
                        [lcl_grd.shape[0],0]])]
        for contour in contours:
            cnts.append(np.flip(contour.reshape((-1,1,2)).astype(np.int32),-1))
                
        for i in range(max_spots):
            locations = (cv2.filter2D(np.abs(lcl_grd -1),-1,kernel) == 9)*1
            mask = np.zeros_like(lcl_grd)
            mask = cv2.drawContours(mask,cnts,-1, 1, -1)
            locations = locations*np.abs(mask -1)
            locs = np.argwhere(locations == 1)
            if locs.shape[0] == 0:
                break
            idx = np.random.choice(np.arange(locs.shape[0]))
            loc = locs[idx]
            loc_list.append(loc)
            lcl_grd[loc[0]-1:loc[0]+2,loc[1]-1:loc[1]+2] = 1
        return loc_list
        
    def generateGrids(self, grids, start = 0, loc = [1,1]):
        """
        Iteratively generates grids
        """
        grids = []
        for i in range(1,self.iterations+1):
            grids.append(np.zeros((self.grid_size**i,self.grid_size**i)))
        grids[start][loc[0]-1:loc[0]+2,loc[1]-1:loc[1]+2] = 1
        grids[start][loc[0],loc[1]] = 0
        return grids

    def buildBlockList(self, blocks, start_position):
        """
        Builds a sorted list of connected blocks.
        """
        # Instantiate block list
        sorted_blocks = [blocks[start_position]]
        blocks = blocks.copy()
        blocks = np.delete(blocks,start_position,axis=0)
        # Find blocks
        not_done = True
        while not_done:
            # Finds the closest blocks to the latest selected block
            distance_to_blocks = np.linalg.norm(blocks-sorted_blocks[-1],axis=1)
            # keep all blocks at a distance of at most 1 block (diagonal included)
            blocks_idx = np.argwhere(distance_to_blocks < 1.5) # 1.5 > sqrt(2)
            selected_blocks = blocks[blocks_idx]
            if selected_blocks.shape[0] == 0:
                not_done = False
                continue
            elif selected_blocks.shape[0] > 1:
                # Apply priority rule:
                # right = 4, bottom = 3, left  = 2, top = 1
                v = []
                for block in selected_blocks:
                    diff = np.squeeze(block) - sorted_blocks[-1]
                    v.append(diff[1] * 2 + diff[0] + (diff[0]==0)*2)
                i = np.argmax(v)
                selected_block = blocks_idx[i,0]
            else:
                selected_block = blocks_idx[0,0]
            # Save and delete
            sorted_blocks.append(blocks[selected_block])
            blocks = np.delete(blocks, selected_block, axis=0)
            if blocks.shape[0] == 0:
                not_done = False
        return sorted_blocks

    def analyzeUpperGrid(self, prev, cur, nxt):
        """
        Checks how the blocks are organized.
         
         1 2 3
         4 # 6
         7 8 9
        
        returns a list telling how the block is connected to other blocks.
        """
        prev_ = prev - cur
        nxt_ = nxt - cur
        p_mask = np.zeros((3,3))
        n_mask = np.zeros((3,3))
        p_mask[prev_[0]+1,prev_[1]+1] = 1
        n_mask[nxt_[0]+1,nxt_[1]+1] = 1
        p_con = self.con * p_mask
        n_con = self.con * n_mask

        return list(p_con[p_con!=0].flatten()) + list(n_con[n_con!=0].flatten())
                
    def sortGrid(self, grid):
        """
        Sorts the blocks in the grid. Each block adjacent in the list are adjacent on the grid.
        """
        blocks = np.argwhere(grid==1)
        start_position = np.argmin(np.linalg.norm(blocks,axis=1))
        block_list = self.buildBlockList(blocks, start_position)
        return np.array(block_list)
    
    def populateGrid(self, upper_grid, current_grid):
        """
        Creates new blocks in the list of blocks.    
        """
        block_list = self.sortGrid(upper_grid)
        is_first = True
        for i, block in enumerate(block_list):
            is_last = (len(block_list)-1)==i
            if is_first:
                prev = block_list[-1]
                nxt = block_list[i+1]
            elif is_last:
                prev = block_list[i-1]
                nxt = block_list[0]
            else:
                prev = block_list[i-1]
                nxt = block_list[i+1]
            logic = self.analyzeUpperGrid(prev, block, nxt)
            prev = current_grid[prev[0]*3:(prev[0]+1)*3,prev[1]*3:(prev[1]+1)*3]
            nxt = current_grid[nxt[0]*3:(nxt[0]+1)*3,nxt[1]*3:(nxt[1]+1)*3]
            nblock = self.generateBLock(logic, prev, nxt, is_first=is_first, is_last=is_last)
            current_grid[block[0]*3:(block[0]+1)*3, block[1]*3:(block[1]+1)*3] = nblock
            is_first = False
        return current_grid, np.array(block_list)
    
    def isAtInterface(self, p, interfaces):
        """
        Check if a block is at an interface.
        """
        done = [False]*len(interfaces)
        for i, inter in enumerate(interfaces):
            if inter == 1:
                if (p[0] == 0) and (p[1] == 0):
                    done[i] = True
            elif inter == 3:
                if (p[0] == 0) and (p[1] == 2):
                    done[i] = True
            elif inter == 7:
                if (p[0] == 2) and (p[1] == 0):
                    done[i] = True
            elif inter == 9:
                if (p[0] == 2) and (p[1] == 2):
                    done[i] = True
            elif inter == 2:
                if p[0]==0:# and np.sum(new_block[0,:]) == 1:
                    done[i] = True
            elif inter == 4:
                if p[1] ==0:#np.sum(new_block[:,0]) == 1:
                    done[i] = True
            elif inter == 6:
                if p[1]==2:#np.sum(new_block[:,-1]) == 1:
                    done[i] = True
            elif inter == 8:
                if p[0]==2:#np.sum(new_block[-1,:]) == 1:
                    done[i] = True
        return done
        

    def randomize(self, v, bmax=2, bmin=0):
        rand = np.round(np.random.rand()*2-1)
        rv = np.min([np.max([v + int(rand),bmin]),bmax])
        return rv

    def generateBLock(self, logic, prev, nxt, is_first=False, is_last=False):
        """
        Generates a new block.
        """
        new_block = np.zeros((3,3))
        # Run constraints
        done = False
        p1 = [0,0]
        p2 = [0,0]
        if   logic[0] == 1:
            new_block[0,0] = 1
            p1 = [0,0]
        elif logic[0] == 2:
            if is_first:
                idx = 1
                idx = self.randomize(idx)
            else:
                idx = int(np.argwhere(prev[-1,:]==1))
                idx = self.randomize(idx)
            new_block[0,idx] = 1
            p1 = [0,idx]
        elif logic[0] == 3:
            new_block[0,2] = 1
            p1 = [0,2]
        elif logic[0] == 4:
            if is_first:
                idx = 1
                idx = self.randomize(idx)
            else:
                idx = int(np.argwhere(prev[:,-1]==1))
                idx = self.randomize(idx)
            new_block[idx,0] = 1
            p1 = [idx,0]
        elif logic[0] == 5:
            raise ValueError('Connectivity 5 does not exist, this should not be possible')
        elif logic[0] == 6:
            if is_first:
                idx = 1
                idx = self.randomize(idx)
            else:
                idx = int(np.argwhere(prev[:,0]==1))
                idx = self.randomize(idx)
            new_block[idx,-1] = 1
            p1 = [idx,2]
        elif logic[0] == 7:
            new_block[2,0] = 1
            p1 = [2,0]
        elif logic[0] == 8:
            if is_first:
                idx = 1
                idx = self.randomize(idx)
            else:
                idx = int(np.argwhere(prev[0,:]==1))
                idx = self.randomize(idx)
            new_block[-1,idx] = 1
            p1 = [2,idx]
        elif logic[0] == 9:
            new_block[2,2] = 1
            p1 = [2,2]
        else:
            raise ValueError('Unknown connectivity')

        done = self.isAtInterface(p1, [logic[1]])[0]

        #print('is done: ', done)
        if done:
            pass 
        elif   logic[1] == 1:
            new_block[0,0] = 1
            p2 = [0,0]
        elif logic[1] == 2:
            if is_last:
                idx = int(np.argwhere(nxt[-1,:]==1))
            else:
                idx = 1
            cd = True
            while cd:
                idx = self.randomize(idx)
                p2 = [0,idx]
                cd = self.isAtInterface(p2,[logic[0]])[0]
            new_block[p2[0],p2[1]] = 1
        elif logic[1] == 3:
            new_block[0,2] = 1
            p2 = [0,2]
        elif logic[1] == 4:
            if is_last:
                idx = int(np.argwhere(nxt[:,-1]==1))
            else:
                idx = 1
            cd = True
            while cd:
                idx = self.randomize(idx)
                p2 = [idx,0]
                cd = self.isAtInterface(p2,[logic[0]])[0]
            new_block[p2[0],p2[1]] = 1
        elif logic[1] == 5:
            raise ValueError('Connectivity 5 does not exist, this should not be possible')
        elif logic[1] == 6:
            if is_last:
                idx = int(np.argwhere(nxt[:,0]==1))
            else:
                idx = 1
            cd = True
            while cd:
                idx = self.randomize(idx)
                p2 = [idx,2]
                cd = self.isAtInterface(p2,[logic[0]])[0]
            new_block[p2[0],p2[1]] = 1
        elif logic[1] == 7:
            new_block[2,0] = 1
            p2 = [2,0]
        elif logic[1] == 8:
            if is_last:
                idx = int(np.argwhere(nxt[0,:]==1))
            else:
                idx = 1
            cd = True
            while cd:
                idx = self.randomize(idx)
                p2 = [2,idx]
                cd = self.isAtInterface(p2,[logic[0]])[0]
            new_block[p2[0],p2[1]] = 1
        elif logic[1] == 9:
            new_block[2,2] = 1
            p2 = [2,2]
        else:
            raise ValueError('Unknown connectivity')

        if np.sum(new_block) == 1:
            pass
        elif np.sum(new_block) == 2:
            d = np.array(p1)-np.array(p2)
            #print('distance: ',d)
            if d[0] == 0:
                if d[1] == 0:
                    pass
                elif np.abs(d[1]) == 1:
                    pass
                elif np.abs(d[1]) == 2:
                    idx = p1[0]
                    idx = self.randomize(idx)
                    new_block[idx,1] = 1
                else:
                    raise ValueError('Distance is too large') 
            elif np.abs(d[0]) == 1:
                if np.abs(d[1]) == 0:
                    pass
                elif np.abs(d[1]) == 1:
                    pass
                elif np.abs(d[1]) == 2:
                    for i in [1,-1,0]:
                        pt = [np.min([np.max([p1[0] + i,0]),2]), 1]
                        cd = self.isAtInterface(pt, logic)
                        if (np.sum(cd) == 0) and (np.sum((np.array(pt) - np.array(p2))**2) <= 2) and (np.sum((np.array(pt) - np.array(p1))**2) <= 2):
                            p3 = pt
                    new_block[p3[0],p3[1]] = 1
                else:
                    raise ValueError('Distance is too large')
            elif np.abs(d[0]) == 2:
                if d[1] == 0:
                    idx = p1[1]
                    idx = self.randomize(idx)
                    new_block[1,idx] = 1
                elif np.abs(d[1]) == 1:
                    for i in [1,-1,0]:
                        pt = [1,np.min([np.max([p1[0] + i,0]),2])]
                        cd = self.isAtInterface(pt, logic)
                        if (np.sum(cd) == 0) and (np.sum((np.array(pt) - np.array(p2))**2) <= 2) and (np.sum((np.array(pt) - np.array(p1))**2) <= 2):
                            p3 = pt
                    new_block[p3[0],p3[1]] = 1
                    idx = p1[1]
                    #new_block[1,idx] = 1
                elif np.abs(d[1]) == 2:
                    npoints = int(round(np.random.rand()*2)+1)
                    if npoints == 1:
                        new_block[1,1] = 1
                    elif npoints > 1:
                        if ((logic[0] == 1) and  (logic[1] == 9)) or ((logic[1] == 1) and  (logic[0] == 9)):
                            if np.round(np.random.rand()) == 0:
                                new_block[0,1] = 1
                                new_block[0,2] = 1
                                new_block[1,2] = 1
                            else:
                                new_block[1,0] = 1
                                new_block[2,0] = 1
                                new_block[2,1] = 1
                        elif ((logic[0] == 7) and (logic[1] == 3)) or ((logic[1] == 7) and (logic[0] == 3)):
                            if np.round(np.random.rand()) == 0:
                                new_block[1,0] = 1
                                new_block[0,0] = 1
                                new_block[0,1] = 1
                            else:
                                new_block[2,1] = 1
                                new_block[2,2] = 1
                                new_block[1,2] = 1
                        else:
                            new_block[1,1] = 1
                    else:
                        raise ValueError('Should not be possible')
                else:
                    raise ValueError('Distance is too large')
            else:
                raise ValueError('Distance is too large')
        elif np.sum(new_block) == 3:
            pass
        elif np.sum(new_block) == 0:
            raise ValueError('No points found after applying constraints. This should not be possible.')
        else:
            raise ValueError('Too many points found after applying constraints. This should not be possible.')
    
        return new_block

if __name__ == "__main__":
    RLG = random_lake_generator(dir_name='test')
    RLG.run()
