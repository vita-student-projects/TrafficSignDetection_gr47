import argparse

from test_yolo_data_cluster import YOLOCustom

def write(preds, output_file):
    with open(output_file, "w+") as f:
        f.write("img_path,inference_time,class_id,class_name,x_box,y_box,w_box,h_box\n")
        for pred in preds:
            speed = pred.speed["inference"]
            for box in pred.boxes:
                f.write(str(pred.path)+","+str(speed)+","+str(box.cls.item())+","+str(pred.names[box.cls.item()])+","+
                        str(box.xywh[0][0].item())+","+str(box.xywh[0][1].item())+","+
                        str(box.xywh[0][2].item())+","+str(box.xywh[0][3].item())+"\n") #format wrong

if __name__ == "__main__":

    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='yolov8x_tsd.pt',
                            help='path of the model')
    parser.add_argument('--data_path', default="images/01/image.000461.jp2", #rest of path hardcoded in streamloader.py
                            help='path of the data')
    parser.add_argument('--output_file', default="our_predictions/prediction_000461_x.txt",
                        help='path to store the output predictions')
    args = parser.parse_args()


    #good image to test : images/00/image.003091.jp2
    #                     images/01/image.006873.jp2

    model_path = "our_retrained_models/" + args.model_path

    model = YOLOCustom(model = model_path)
    preds = model(args.data_path)
    write(preds, args.output_file)
    print(f"prediction done, stroed at {args.output_file}")
