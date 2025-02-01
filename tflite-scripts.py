import tensorflow as tf
import numpy as np

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="tflite_models/optimized_vRG_70epc_100001-1487003016146-0-1487003016393.weights.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example input data (adjust according to model input shape)
input_data = np.array(np.random.random(input_details[0]['shape']), dtype=np.float32)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run the model
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
