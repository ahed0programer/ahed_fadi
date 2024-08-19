from tensorflow.keras.models import load_model
import tensorflow


model = load_model("models/paper_classification_model.h5")

# Save the model in the SavedModel format
model.save("models/paper_classification_model.keras")

# Load the model from the SavedModel format
loaded_model = tensorflow.keras.models.load_model("models/paper_classification_model.keras")


converter = tensorflow.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
