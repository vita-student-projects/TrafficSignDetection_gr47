import argparse

from test_yolo_data_cluster import YOLOCustom


if __name__ == "__main__":

    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n.pt',
                        choices=('yolov8n.pt', 'yolov8m.pt', 'yolov8x.pt'),
                            help='size of model to use')
    parser.add_argument('--epochs', default=3, type=int,
                                 help='number of epochs')
    args = parser.parse_args()


    model = args.model
    print("loading yolo, model: ", model)

    model = YOLOCustom('trained_models/' + model) #pretrained

    model.train(data = "tsr_dataset.yaml", epochs=args.epochs)