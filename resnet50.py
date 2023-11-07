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

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs detected. Running on CPU.")

# Paths
metadata_path = "C:\\Users\\aryan\\OneDrive\\Desktop\\employability speedrun\\SkinCancerPrediction\\HAM10000_metadata.csv"
images_dir = "C:\\Users\\aryan\\OneDrive\\Desktop\\employability speedrun\\SkinCancerPrediction\\archive\\HAM10000_images\\"

metadata = pd.read_csv(metadata_path)
metadata["path"] = images_dir + metadata["image_id"] + ".jpg"

# Splitting the data
train_data, test_data = train_test_split(metadata, test_size=0.2, random_state=42)

# Data augmentation
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, 
                                   horizontal_flip=True, 
                                   vertical_flip=True, 
                                   rotation_range=180, 
                                   zoom_range=0.1, 
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                   shear_range=0.1, 
                                   brightness_range=[0.5, 1.5])
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_data, x_col="path", y_col="dx", target_size=(224, 224), batch_size=32, class_mode="categorical")
test_generator = test_datagen.flow_from_dataframe(dataframe=test_data, x_col="path", y_col="dx", target_size=(224, 224), batch_size=32, class_mode="categorical", shuffle=False)

# Model: Load if exists or create a new one
try:
    model = load_model('res_model.h5', custom_objects={'f1_metric': f1_metric})
    print("Loaded saved model")
except:
    from keras.applications.resnet import ResNet50
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D, Dropout
    
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy', f1_metric])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model_checkpoint = ModelCheckpoint('res_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Training the model
history = model.fit(train_generator, validation_data=test_generator, epochs=30, verbose=1, callbacks=[reduce_lr, early_stopping, model_checkpoint])

# Evaluation
results = model.evaluate(test_generator)
loss, accuracy = results[0], results[1]
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# Saving metrics to "resnet_metrics.txt"
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

classification_rep = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
confusion_mat = confusion_matrix(y_true, y_pred_classes)

with open("resnet_metrics.txt", "w") as file:
    file.write("Test loss: {}\n".format(loss))
    file.write("Test accuracy: {}\n".format(accuracy))
    file.write("\nClassification Report:\n")
    file.write(classification_rep)
    file.write("\nConfusion Matrix:\n")
    file.write(str(confusion_mat))