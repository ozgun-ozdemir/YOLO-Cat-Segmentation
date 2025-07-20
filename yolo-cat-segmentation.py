import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os  

# Function to add a label to the image
def put_label(img, label):
    h, w = img.shape[:2]
    labeled_img = np.ones((h + 40, w, 3), dtype=np.uint8) * 255  
    labeled_img[40:, :, :] = img  
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = (w - text_size[0]) // 2  
    text_y = 30 
    cv2.putText(labeled_img, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    return labeled_img

# Main segmentation function
def run_segmentation(image_path, model_path='yolo11x-seg.pt'):
    if not os.path.exists(image_path):  
        print(f"Error: The image file '{image_path}' was not found.")
        return 

    model = YOLO(model_path)

    image = cv2.imread(image_path)

    results = model.predict(image, conf=0.5)
    result = results[0]

    instance_image = result.plot()
    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    cat_class_id = 15  # The class ID for cat
    
    cat_masks = []
    for i, cls_id in enumerate(classes):
        if cls_id == cat_class_id: 
            cat_masks.append(masks[i])

    if len(cat_masks) > 0: 
        cat_mask_combined = np.zeros_like(cat_masks[0], dtype=bool)
        for cat_mask in cat_masks:
            cat_mask_combined |= cat_mask.astype(bool)

        cat_mask_uint8 = cat_mask_combined.astype(np.uint8) * 255
        cat_semantic_colored = np.stack([cat_mask_uint8] * 3, axis=-1) 

        size = (640, 640) 
        image_resized = cv2.resize(image, size)
        instance_resized = cv2.resize(instance_image, size)
        cat_semantic_resized = cv2.resize(cat_semantic_colored, size)

        # Label the images
        image_labeled = put_label(image_resized, "Original")
        instance_labeled = put_label(instance_resized, "Instance")
        cat_semantic_labeled = put_label(cat_semantic_resized, "Cat Semantic")

        combined = cv2.hconcat([image_labeled, instance_labeled, cat_semantic_labeled])

        output_path = "output_cat_only.jpg"
        cv2.imwrite(output_path, combined)

        # Display the images
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Instance Segmentation")
        plt.imshow(cv2.cvtColor(instance_resized, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Cat Semantic Segmentation")
        plt.imshow(cv2.cvtColor(cat_semantic_resized, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    else:
        print("No cats detected.")  

if __name__ == "__main__":
    image_path = "test.png"  # Path to your image
    run_segmentation(image_path)
