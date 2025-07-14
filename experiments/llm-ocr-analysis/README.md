# llm-ocr-analysis

- image.py:

Auxiliary functions for handling the NBID dataset

`read_rg_dataset` assumes dataset is in format:

\- `synthetic\`

\-- `base_images\`

\-- `base_labels\` - labels in the base NBID JSON format

For the warped dataset, `transform=True`

For the dataset warped with the predicted boxes, `prediction_box=True` and predicted boxes at `pred_dir="predicted_dir"`

To check the warped images, `save_in_aux=True` and `image_dir="aux_im"`

Predicted boxes must be in format `f,x1,x2,x3,x4,y1,y2,y3,y4` where the coordinates are normalized between 0 and 1.

- get_output_gemini.py

Main script for querying LLM to analyse the documents

Arguments `run_name`, `wait_time` and `read_rg_dataset` can be changed between each run

Results are saved in `./gemini_outputs/{run_name}` (directory must exist, script does not create it)

- analyse_output_gemini.py

Main script for analysing LLM output

Argument `do_ocr_gt` is whether the output is compared with the annotated text (`do_ocr_gt = True`) or with the text predicted by the model using the GT bounding boxes (`do_ocr_gr = False`)

Argument `run` corresponds to `run_name` from `get_output_gemini.py`

Results are gathered using the Levenshtein-based OCR metric (calculated in `ocr_iou`)

The dataset with the format and predictions can be found at `duo:/datasets3/lmlwojcik/nbid_analysis`
