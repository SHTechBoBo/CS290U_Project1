# Change Python Path

`export PYTHONPATH=$(pwd):$PYTHONPATH`

# Generate Dataset

`python yehw2024_project1/generate_dataset.py`

Original images are in `yehw2024_project1/original_imgs` folder.

You can change `bayer_pattern` to generate images in different bayer pattern.

# Save Result

`python yehw2024_project1/save_results.py dbsr_yhw`

The result will be shown in `yehw2024_project1/result/DBSR_yhw` folder.

# Visualize Result

`python yehw2024_project1/visualize_results.py dbsr_yhw`

The result will be shown in `yehw2024_project1/result/processed_imgs` folder.

# Compute Score

`python yehw2024_project1/compute_score.py dbsr_yhw`

# Object Detection

`python yehw2024_project1/yolov8.py`

The result will be shown in `yehw2024_project1/yolo` folder.