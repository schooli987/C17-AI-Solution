import cv2
import numpy as np
import tensorflow as tf

# Load your trained Teachable Machine model (TF Keras)
model = tf.keras.models.load_model("keras_model.h5")


# Labels (must match your model's labels.txt)
CLASS_NAMES = ["With disease", "Without disease"]

# Read an image file (replace 'leaf.jpg' with your file path)
image_path = "leaf4.png"
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not load image.")
else:
    # Preprocess the image for Teachable Machine
    img = cv2.resize(frame, (224, 224))  # model input size
    img_array = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    print(" Image loaded and preprocessed successfully!")
    
    cv2.imshow("Plant Leaf Disease Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()