import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
import tensorflow_model_optimization as tfmot
import os

directory = 'tflite_models'
if not os.path.exists(directory):
    os.makedirs(directory)

# Define the model architecture (same as used for training)
def create_model(x_shape, y_shape):
    n_timesteps, n_channels, n_outputs = (x_shape[1], x_shape[2], y_shape[1])

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=21, activation='relu', input_shape=(n_timesteps, n_channels)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=64, kernel_size=21, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.1))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(n_outputs, activation='linear'))

    model.summary()

    # Use RMSprop optimizer with a learning rate of 0.001
    rmsprop = RMSprop(learning_rate=0.001, rho=0.9)
    model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['mae'])

    return model

# Initialize model
model = create_model((1000, 256, 2), (1000, 1))

# Load weights
weights_files = [
    "vRG_70epc_100001-1487003016146-0-1487003016393.weights.h5",
    "vRG_70epc_100002-1487006911581-0-1487006911849.weights.h5",
    "vRG_70epc_100003-1487010212332-0-1487010212594.weights.h5",
    "vRG_70epc_100004-1487016703619-0-1487016703979.weights.h5",
    "vRG_70epc_100005-1487019992000-0-1487019992346.weights.h5",
    "vRG_70epc_100006-1487023971109-0-1487023971496.weights.h5"
]

# Loop to load each set of weights and apply optimizations
for weight_file in weights_files:
    print(f"Loading weights: {weight_file}")
    model.load_weights(weight_file)

    # TODO: prune_low_magnitude is not working with the current model architecture
    # TODO: ver sobre o que vai considerar para tflite
    # TODO: calcular tempo de teste para cada modelo - no PC e microcontrolador

    # ---
    # TODO: neural architecture search

    # Optional: Apply pruning (Uncomment to use pruning)
    # pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    #     initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=1000
    # )

    # Apply pruning to the model (if needed)
    # pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

    # Optional: Apply quantization and adjust TFLite conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)  # Use pruned_model here if pruning is applied

    # Enable SELECT_TF_OPS and disable lowering tensor list ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save the optimized model and the TFLite model
    optimized_model_filename = f"optimized_models/optimized_{weight_file.split('.')[0]}.h5"
    model.save(optimized_model_filename)  # Save the model with weights (this is optional)

    tflite_model_filename = f"tflite_models/optimized_{os.path.splitext(weight_file)[0]}.tflite"
    with open(tflite_model_filename, 'wb') as f:
        f.write(tflite_model)

    print(f"Optimized model saved as: {optimized_model_filename}")
    print(f"TFLite model saved as: {tflite_model_filename}")
