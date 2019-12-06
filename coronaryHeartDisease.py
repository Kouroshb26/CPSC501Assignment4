import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split

data = pd.read_csv("heart.csv")
data = data.drop("row.names",1)
# print(data.describe())    #Describe the data

feature_columns = []
for header in ["sbp","tobacco","ldl","adiposity","typea","obesity","alcohol","age"]:
  feature_columns.append(tf.feature_column.numeric_column(header))


data["famhist"] = data["famhist"].apply(str)
famhist = tf.feature_column.categorical_column_with_vocabulary_list("famhist",["Present","Absent"])
feature_columns.append(tf.feature_column.indicator_column(famhist))

def create_dataset(dataframe, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('chd')
  return tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)) \
          .shuffle(buffer_size=len(dataframe)) \
          .batch(batch_size)

train, test = train_test_split(data, test_size=0.1, random_state=42)

train_ds = create_dataset(train)
test_ds = create_dataset(test)

model = tf.keras.models.Sequential([
  tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
  tf.keras.layers.Dense(units=128, activation='relu'),
  tf.keras.layers.Dropout(rate=0.2),
  tf.keras.layers.Dense(units=128, activation='relu'),
  tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, epochs=100, use_multiprocessing= True, verbose =2 )

print("--Save model--")
model.save("coronaryHeartDisease.h5")

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(test_ds,verbose = 2 )
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")
