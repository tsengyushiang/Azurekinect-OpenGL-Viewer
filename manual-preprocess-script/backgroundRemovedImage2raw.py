import json

def readJson(fileName):
    with open(fileName, 'r',encoding="utf-8") as reader:
        jf = json.loads(reader.read())
    return jf
    
def writeJSON(jsonf,jsonData):
    with open(jsonf, 'w+') as f:
        f.seek(0)
        # ascii for chinese 
        json.dump(jsonData, f,ensure_ascii=False)
        f.truncate()


input_output_folder ="../app/azureKinect/"

inputJson = input_output_folder + "000065393712-front.json"
inputJsonRawImageKey = "colormap_raw"
inputBackgroundRemovedImage = input_output_folder + "000065393712-front-green.png"
outputJson = input_output_folder + "000065393712-front-green.json"

originJson = readJson(inputJson)
outputJsonData = originJson

import cv2

backgroundRemovedImage = cv2.imread(inputBackgroundRemovedImage)

outputJsonData[inputJsonRawImageKey] = backgroundRemovedImage.flatten().tolist()
writeJSON(outputJson,outputJsonData)
