import numpy as np
import cv2
import os

from utils import loadDict, makeGaussianKernel, randomSign

class DemGenerator:
    def __init__(self, cnt_path=None, save_path=None, shape=(3**8, 3**8), max_sat=100, min_sat=-100):
        self.contours_path = cnt_path
        self.save_path = save_path
        self.shape = shape
        self.max_sat = max_sat
        self.min_sat = min_sat
        self.max_morph_iter = 3
        self.morph_max = 1.3
        self.morph_min = 0.6

    def run(self):
        self.loadContours()
        mask = self.makeMask()
        np.savez_compressed(os.path.join(self.save_path,'mask.npz'),data=mask)
        distance = self.computeDistanceTransform()
        dem = distance*(mask == 1) - distance*(mask==0)
        mult = self.morph()
        dem_morphed = self.saturate(dem*mult)
        np.savez_compressed(os.path.join(self.save_path,'dem.npz'),data=dem_morphed)

    def morph(self):
        multiplier = np.ones(self.shape)
        k_sizes = []
        count = []
        for i in range(0,self.max_morph_iter,1):
            v = int(self.shape[0]/(2**i))
            if (v%2 == 0):
                v += 1
            k_sizes.append(v)
            count.append(2**(i+5))
        tmp = np.ones((self.shape[0]+int(k_sizes[0]+2),(self.shape[1]+int(k_sizes[0]+2))))
        for i, c in enumerate(count):
            ks = int(k_sizes[i]//2)
            kern = makeGaussianKernel(int(k_sizes[i]), 4)
            kern = kern
            for j in range(c):
                x = int(np.random.rand()*self.shape[0] + k_sizes[0]//2 + 1)
                y = int(np.random.rand()*self.shape[1] + k_sizes[0]//2 + 1)
                tmp[x-ks:x+ks+1,y-ks:y+ks+1] += kern*randomSign()
        multiplier = tmp[k_sizes[0]//2 +1:-k_sizes[0]//2 -1,k_sizes[0]//2 +1:-k_sizes[0]//2 -1]
        multiplier = (multiplier-np.min(multiplier))/(np.max(multiplier) - np.min(multiplier))*(self.morph_max-self.morph_min) + self.morph_min
        return multiplier

    def loadContours(self):
        contours_dict = loadDict(self.contours_path)
        self.contours = []
        for key in contours_dict.keys():
            self.contours += contours_dict[key]

    def makeMask(self):
        cnts = []
        for contour in self.contours:
            cnts.append(np.flip(contour.reshape((-1,1,2)).astype(np.int32),-1))
        mask = np.zeros(self.shape)

        mask = cv2.drawContours(mask,[cnts[0]],-1, 1, -1)
        mask = np.abs(mask - 1)
        for cnt in cnts[1:]:
            mask=cv2.drawContours(mask, [cnt],-1, 1, -1)
        return mask.astype(bool)

    def computeDistanceTransform(self):
        fused_contours = np.ones(self.shape)
        for contour in self.contours:
            cnt = np.flip(contour.reshape((-1,1,2)).astype(np.int32),-1)
            cv2.drawContours(fused_contours, [cnt], -1, 0)
        dist = cv2.distanceTransform(fused_contours.astype(np.uint8), cv2.DIST_L2, 5)
        return dist.astype(np.int32)

    def saturate(self, distance):
        distance[distance > self.max_sat] = self.max_sat
        distance[distance < self.min_sat] = self.min_sat
        return distance

if __name__ == "__main__":
    DEMG = DemGenerator("raw_generation/gen8/contours.pkl", "raw_generation/gen8")
    DEMG.run()