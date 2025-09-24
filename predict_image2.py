import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from grounding_dino.groundingdino.util.inference import load_model
from sam2.sam2_image_predictor import SAM2ImagePredictor

"""
Hyper parameters
"""
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def main():
    # 加载模型
    model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device="cpu"
    )
    predictor = SAM2ImagePredictor(model)

    # 加载图像
    image_path = "/root/work/filestorage/wwg/Grounded-SAM-2/notebooks/images/truck.jpg"  # 替换为你的图像路径
    image = load_image(image_path)
    image_array = np.array(image)
    #设置图像
    predictor.set_image(image_array)
    
    # 将图像转换为Tensor
    input_tensor = torch.tensor(image_array.transpose(2, 0, 1)).to(torch.float32).unsqueeze(0) / 255.0

    # 进行推理
    point_coords = np.array([[100, 100], [200, 200], [150, 150]])  # 示例点坐标
    point_labels = np.array([1, 1, 1])  # 示例点标签，1 表示前景
    #point_coords = np.array([[150, 150]])  # 示例点坐标
    #point_labels = np.array([1])  # 示例点标签，1 表示前景
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )

    # 显示结果
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        show_mask(mask, plt.gca())
        print(f"Mask {i + 1}, Score: {score:.3f}")
    plt.axis('off')
    plt.savefig('output.png')


if __name__ == "__main__":
    main()

