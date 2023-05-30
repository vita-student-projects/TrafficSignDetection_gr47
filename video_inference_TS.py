import os
import json
from tqdm import tqdm
import operator

from test_yolo_data_cluster import YOLOCustom


# format json {

# "project" :  "project_name",
# "output": [
# { "frame":  1,
# "predictions": []
# },
# { "frame":  2,
# "predictions": [] 
# },

# { "frame":  5000,
# "predictions": []
# }
# ]

# }


if __name__ == "__main__":

    folder_path = "/work/vita/nmuenger_trinca/annotations/"

    folder = "video_frames/"
    dirs = os.listdir(folder_path + folder)

    model_path = "our_retrained_models/yolov8x_tsd.pt"
    model = YOLOCustom(model = model_path)
    
    #dirs = dirs[:2]
    output = []
    for img in tqdm(dirs):
        #img.replace(folder_path, "")
        #os.path.normpath(img).split(os.path.sep)
        #img = breakpoint()
        frame = int(img[-8:-4])
        #breakpoint()
        pred = model(folder + img)[0]
        boxes = []
        for box in pred.boxes:
            boxes.append({"cls" :str(pred.names[box.cls.item()]), 
                          "x"   :box.xywh[0][0].item(),
                          "y"   :box.xywh[0][1].item(),
                          "w"   :box.xywh[0][2].item(),
                          "h"   :box.xywh[0][3].item()})
        
        
        output.append({"frame":frame, "prediction":boxes})

    output_sorted = sorted(output, key=operator.itemgetter('frame'))
    #breakpoint()
    aDict = {"project": "TSD_group47", "output":output_sorted}
    jsonString = json.dumps(aDict)
    jsonFile = open("video_prediction_test1_cut.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()