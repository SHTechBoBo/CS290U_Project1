# 导入相关库
from ultralytics import YOLO
import cv2

# 加载预训练的 YOLOv8 模型（segmentation）
model = YOLO('yolov8n-seg.pt')  # 你可以使用不同版本的模型，如 yolov8s-seg.pt



for i in range(20):
    dbsr_img = cv2.imread('./yehw2024_project1/result/processed_imgs/rggb/{:04d}/0.png'.format(i))
    lr_image = cv2.imread('./yehw2024_project1/result/processed_imgs/rggb/{:04d}/lr_0.png'.format(i))
    # 使用模型对图像进行预测（分割）
    dbsr_result = model(dbsr_img)
    lr_result = model(lr_image)
    # 绘制预测结果
    dbsr_annotated_image = dbsr_result[0].plot()  # 这会把检测结果直接绘制到图像上
    lr_annotated_image = lr_result[0].plot()

    cv2.imwrite('./yehw2024_project1/yolo/{:04d}_hr.png'.format(i), dbsr_annotated_image)
    cv2.imwrite('./yehw2024_project1/yolo/{:04d}_lr.png'.format(i), lr_annotated_image)