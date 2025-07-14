# doc-analysis-reprod

All datasets are available at `duo:/datasets3/lmlwojcik`

- YOLO reproduction

Uses the NBID YAML format

Training `nano` and `xtra` YOLOv11-obb models (available at ultralytics) with `yolo_train.py`

Testing with `yolo_test.py`

Results at `yolo_results/`

Requirements: `ultralytics, shapely`

- RTMDet reproduction

Uses the NBID DOTA format

We train `tiny` and `l` RTMDet-R models (available at `mmyolo/configs/rtmdet` on github)

To train, clone `mmyolo`, replace configs with the ones provided here and run `python3 tools/train.py configs/rtmdet/rotated/{config}`

To test, `get_iou_rtm.py`

Results at `preds/`

Note: at the time of development, we had to change the DOTA file in the mmyolo library because the custom classes for DOTA are not yet supported. We changed the class name for class 0 to `doc`

Requirements:
```
cd {HOME}
pip install -U -q openmim
mim install -q "mmengine>=0.6.0"
mim install -q "mmcv>=2.0.0rc4,<2.1.0"
mim install -q "mmdet>=3.0.0rc6,<3.1.0"
git clone https://github.com/open-mmlab/mmyolo.git
cd {HOME}/mmyolo
mim install -v -e .
```

- Jdeskew reproduction

Uses only the base document images, in `base_images/`

Evaluation with `run_doc.py`, details on `read_rg_datasets` on `llm-ocr-analysis` repository in this organization

Results at `warped_icip/`

Requirements: `jdeskew, opencv-python` 

