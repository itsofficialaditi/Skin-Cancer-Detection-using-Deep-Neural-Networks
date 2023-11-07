import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import load_model

# F1 Metric definition
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# Paths
metadata_path = "C:\\Users\\aryan\\OneDrive\\Desktop\\employability speedrun\\SkinCancerPrediction\\HAM10000_metadata.csv"
images_dir = "C:\\Users\\aryan\\OneDrive\\Desktop\\employability speedrun\\SkinCancerPrediction\\archive\\HAM10000_images\\"

metadata = pd.read_csv(metadata_path)
metadata["path"] = images_dir + metadata["image_id"] + ".jpg"

# Splitting the data
train_data, test_data = train_test_split(metadata, test_size=0.2, random_state=42)

# Data augmentation
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_data, x_col="path", y_col="dx", target_size=(224, 224), batch_size=32, class_mode="categorical", shuffle=False)

model = load_model('best_model.h5', custom_objects={'f1_metric': f1_metric})

# Evaluation
results = model.evaluate(test_generator)
loss, accuracy, f1 = results[0], results[1], results[2]
print("Test loss:", loss)
print("Test accuracy:", accuracy)
print("Test F1 Score:", f1)

# Saving metrics to "resnet_metrics.txt"
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

classification_rep = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
confusion_mat = confusion_matrix(y_true, y_pred_classes)

with open("resnet_metrics.txt", "w") as file:
    file.write("Test loss: {}\n".format(loss))
    file.write("Test accuracy: {}\n".format(accuracy))
    file.write("Test F1 Score: {}\n".format(f1))
    file.write("\nClassification Report:\n")
    file.write(classification_rep)
    file.write("\nConfusion Matrix:\n")
    file.write(str(confusion_mat))
