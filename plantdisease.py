import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Path to your exported Teachable Machine TFLite model
MODEL_PATH = "model_unquant.tflite"   # replace with your .tflite file
CLASS_NAMES = ["with disease", "without disease"]  # update according to your model

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_leaf(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  # Teachable Machine default size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    img_array = img_array / 255.0  # normalize

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Handle binary vs categorical
    if predictions.shape[1] == 1:  # Binary classification (sigmoid)
        score = predictions[0][0]
        result = CLASS_NAMES[1] if score > 0.5 else CLASS_NAMES[0]
    else:  # Categorical (softmax)
        index = np.argmax(predictions[0])
        result = CLASS_NAMES[index]

    print(f"Prediction: {result}")

# Example usage
img_path = "leaf2.jpg"   # replace with your test image
predict_leaf(img_path)
