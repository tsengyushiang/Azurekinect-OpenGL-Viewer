import json
import numpy as np
import cv2

class InputJson:
    
    def readJson(self,fileName):
        with open(fileName, 'r',encoding="utf-8") as reader:
            jf = json.loads(reader.read())
        return jf
    
    def writeJSON(self,jsonf,jsonData):
        with open(jsonf, 'w+') as f:
            f.seek(0)
            # ascii for chinese 
            json.dump(jsonData, f,ensure_ascii=False)
            f.truncate()

    def __init__(self, filepath):
        self.data = self.readJson(filepath)

    @property
    def color(self):
        color = np.asarray(self.data["colormap_raw"], dtype=np.uint8).reshape((self.height,self.width,3))
        return color

    @property
    def depth(self):
        return np.array(self.data["depthmap_raw"]).reshape((self.height,self.width))

    @property
    def depthVisualize(self):
        quantizationDepth = np.array(self.data["depthmap_raw"]).reshape((self.height,self.width))*self.data["depthscale"]/5.0*255
        return  cv2.cvtColor(quantizationDepth.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    @property
    def culleddepthVisualize(self):
        depth = np.array(self.data["depthmap_raw"]).reshape((self.height,self.width))*self.data["depthscale"]/5.0*255
        depth[(self.cullingMask <= 0)] = 0
        return depth

    @property
    def cullingMask(self):
        maskIncolorCoord =np.asarray(self.data["yzCullingMask"], dtype=np.uint8).reshape((self.height,self.width))
        return maskIncolorCoord

    @property 
    def validDepthMask(self):
        mask = np.zeros((self.height,self.width),np.uint8)
        mask[ (self.depth > 0) & (self.cullingMask > 0)] = 1
        return mask

    @property
    def validDepthMask3(self):
        return np.repeat(self.validDepthMask[..., np.newaxis], 3, axis=-1)

    @property
    def biggestConnectedDepth(self):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(self.validDepthMask, connectivity=4)
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        mask = np.zeros((self.height,self.width),np.uint8)
        mask[output == max_label] = 255        
        return mask

    @property
    def contour(self):
        
        contours, hierarchy = cv2.findContours(self.biggestConnectedDepth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            print('failed to find contour')
            exit()

        output = np.zeros((self.height,self.width),np.uint8)
        # contours = max(contours, key = cv2.contourArea)
        output = cv2.drawContours(output, contours, -1, 128, 10)

        return output

    @property
    def width(self):
        return self.data["width"]

    @property
    def height(self):
        return self.data["height"]
    
    @property
    def depthscale(self):
        return  self.data["depthscale"]

def grabcut_batch_test(input,headSeg,outputdir):

    def blendMask(mask,color):
        b_channel, g_channel, r_channel = cv2.split(color)
        b_channel[mask==0] = 102
        g_channel[mask==0] = 255
        r_channel[mask==0] = 256
        return cv2.merge((b_channel, g_channel, r_channel))

    def saveImg(filename,img):
        cv2.imwrite(os.path.join(outputdir,filename),img)

    # force alpha to binary
    headSeg[headSeg!=0]=255

    saveImg('color.png',input.color)
    saveImg('depth.png',input.depthVisualize)

    zeros = np.zeros((input.height,input.width),np.uint8)
    ones = np.ones((input.height,input.width),np.uint8)*255

    floor_far_mask = np.copy(ones)
    floor_far_mask[input.cullingMask>0] = 0
    saveImg('floor_far_mask.png', cv2.merge((ones, zeros, zeros, floor_far_mask)))

    zeroDepth_mask = np.copy(ones)
    zeroDepth_mask[input.depth>0] = 0
    saveImg('zeroDepth_mask.png', cv2.merge((zeros, ones, zeros, zeroDepth_mask)))
    saveImg('additional_mask.png', cv2.merge((zeros, zeros, zeros, 255-headSeg)))
    saveImg('biggestConnectedDepth.png', cv2.merge((zeros, zeros, zeros, 255-input.biggestConnectedDepth)))
    
    headAndForegroundDepth = np.copy(zeros)
    headAndForegroundDepth[(headSeg>0) + (input.biggestConnectedDepth>0)] = 255
    saveImg('headAndForegroundDepth.png', cv2.merge((zeros, zeros, zeros, 255-input.biggestConnectedDepth)))

    color_depth = np.vstack((blendMask(input.biggestConnectedDepth,input.color),input.color))
    depthCull_head = np.vstack((blendMask(headAndForegroundDepth,input.color),blendMask(headSeg,input.color)))

    return np.hstack((color_depth,depthCull_head))

import os

def getExtFilesInfolder(dataDict,folder,ext,removsubstrForKey=[]):
    arr_txt = [x for x in os.listdir(folder) if x.endswith(ext)]
    for file in arr_txt:

        name = file
        for substr in removsubstrForKey:
            name = name.replace("{}".format(substr), "")

        if name not in dataDict:
            dataDict[name]={}
        dataDict[name][ext] = os.path.join(folder,file)

dataDict={}

dataExt = ".json"
getExtFilesInfolder(dataDict,"../app/2021-08-30-12-33-23-longhairtest",dataExt,[dataExt])

# background matting v2 mask
# maskExt = ".color.jpg"
# outputdir = "20210820-146frames-testing-data"
# getExtFilesInfolder(dataDict,"../app/20210820-146frames-testing-data/background-mattingv2-output/pha",maskExt,[maskExt])

# CDCL-humanpart seg mask
maskExt = ".color.png_head.jpg"
outputdir = "2021-08-30-12-33-23-longhairtest-CDCL+depthculling"
getExtFilesInfolder(dataDict,"../app/2021-08-30-12-33-23-longhairtest/CDCL-humanpartSeg-output",maskExt,[maskExt,"seg_"])

os.mkdir(outputdir)
fps = 3
out = None

for key in dataDict:
    outputfolder = os.path.join(outputdir,key)
    os.mkdir(outputfolder)

    print(dataDict[key][dataExt])
    print(dataDict[key][maskExt])

    input = InputJson(dataDict[key][dataExt])

    segmask = cv2.imread(dataDict[key][maskExt],0)

    videoframe = grabcut_batch_test(input,segmask,outputfolder)

    cv2.putText(videoframe, key, (10, 120), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
    if out is None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(outputdir,'output.avi'), fourcc, fps , (videoframe.shape[1],  videoframe.shape[0]))

    out.write(videoframe)


out.release()

