from ultralytics import YOLO

model = YOLO("yolo11n-obb.yaml").load("yolo11n.pt")

results = model.train(data="nbid.yaml", epochs=2000, imgsz=640, device=0, patience=400, name="yolo11n_2000_ptd")

model = YOLO("yolo11x-obb.yaml").load("yolo11x.pt")

results = model.train(data="nbid.yaml", epochs=2000, imgsz=640, device=0, patience=400, name="yolo11x_2000_ptd")
