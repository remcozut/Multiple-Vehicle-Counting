import torch
import torch, os, sys

def main():
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    from ultralytics import YOLO
    model  = YOLO("yolov8s.pt")

    data_config = r"c:\projects\python\Multiple-Vehicle-Counting\mybikes\data.yaml"

    model.train(data = data_config, epochs = 2, batch = 20, imgsz = [1024,2048],  device="cuda:0")

    model.save("yolov8_custom.pt")


if __name__ == "__main__":
    main()