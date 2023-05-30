import argparse
import matplotlib.pyplot as plt
import os
from test_yolo_data_cluster import YOLOCustom

if __name__ == "__main__":

    print("evaluating metrics")

    all_test_set = True
    if all_test_set:
        test_data_file = "tsr_dataset_test.yaml"  #rest of the path is hardcoded in stream_loader.py


        maps50 = []
        maps50_95 = []
        time = []

        #NANO
        model_path = "our_retrained_models/yolov8n_tsd.pt"
        model = YOLOCustom(model = model_path)
        metrics = model.val(data = test_data_file)
        print(f"Final results for yolov8 nano: mAP50 {metrics.results_dict['metrics/mAP50(B)']}, "+
            f"mAP50 {metrics.results_dict['metrics/mAP50-95(B)']} and inference time {metrics.speed['inference']}")
        maps50.append(metrics.results_dict['metrics/mAP50(B)'])
        maps50_95.append(metrics.results_dict['metrics/mAP50-95(B)'])
        time.append(metrics.speed['inference'])
        # Final results for yolov8 nano: mAP50 0.4432601368850605, mAP50 0.35808306560387837 and inference time 2.4061742004992466

        #MEDIUM
        model_path = "our_retrained_models/yolov8m_tsd.pt"
        model = YOLOCustom(model = model_path)
        metrics = model.val(data = test_data_file)
        print(f"Final results for yolov8 medium: mAP50 {metrics.results_dict['metrics/mAP50(B)']}, "+
            f"mAP50 {metrics.results_dict['metrics/mAP50-95(B)']} and inference time {metrics.speed['inference']}")
        maps50.append(metrics.results_dict['metrics/mAP50(B)'])
        maps50_95.append(metrics.results_dict['metrics/mAP50-95(B)'])
        time.append(metrics.speed['inference'])
        # Final results for yolov8 medium: mAP50 0.524872974614051, mAP50 0.4547125627884293 and inference time 5.599455987948984
        
        #LARGE
        model_path = "our_retrained_models/yolov8x_tsd.pt"
        model = YOLOCustom(model = model_path)
        metrics = model.val(data = test_data_file)
        print(f"Final results for yolov8 x-tra large: mAP50 {metrics.results_dict['metrics/mAP50(B)']}, "+
            f"mAP50 {metrics.results_dict['metrics/mAP50-95(B)']} and inference time {metrics.speed['inference']}")
        maps50.append(metrics.results_dict['metrics/mAP50(B)'])
        maps50_95.append(metrics.results_dict['metrics/mAP50-95(B)'])
        time.append(metrics.speed['inference'])
        # Final results for yolov8 x-tra large: mAP50 0.5595280276367262, mAP50 0.48758862706402817 and inference time 13.416062195496893

        plt.plot(time, maps50, label = "maps50")
        plt.plot(time, maps50_95, label = "maps50_95")
        plt.legend()
        plt.xlabel("Inference time [s]")
        plt.ylabel("mAP50 / mAP50-95")
        plt.title("Comparison of the models' complexity")

        labels = ["nano", "medium", "x-tra large"]
        for i, (x1,y1) in enumerate(zip(time,maps50_95)):

            label = labels[i]

            plt.annotate(label, # this is the text
                    (x1,y1), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(10,20), # distance from text to points (x,y)
                    ha='right')
        
        plt.savefig("our_predictions/random.png")
    
    else:
        test_image = "tsr_dataset_one_image.yaml"  #rest of the path is hardcoded in stream_loader.py

        print("here")
        time = []

        #NANO
        model_path = "our_retrained_models/yolov8n_tsd.pt"
        model = YOLOCustom(model = model_path)
        metrics = model.val(data = test_image)
        print(f"Final inference time for yolov8 nano {metrics.speed['inference']}")
        time.append(metrics.speed['inference'])

        #NANO
        model_path = "our_retrained_models/yolov8m_tsd.pt"
        model = YOLOCustom(model = model_path)
        metrics = model.val(data = test_image)
        print(f"Final inference time for yolov8 medium {metrics.speed['inference']}")
        time.append(metrics.speed['inference'])

        #NANO
        model_path = "our_retrained_models/yolov8x_tsd.pt"
        model = YOLOCustom(model = model_path)
        metrics = model.val(data = test_image)
        print(f"Final inference time for yolov8 x-tra large {metrics.speed['inference']}")
        time.append(metrics.speed['inference'])

        plt.bar(["nano", "medium", "x-tra large"], time)
        plt.savefig("our_predictions/bar.png")