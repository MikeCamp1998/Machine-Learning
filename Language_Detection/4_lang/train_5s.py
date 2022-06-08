from utils import *

np.random.seed(1)  # set seed for reproducibility
gc.collect()       # garbage collection

input_shape = (20, 216, 1)

model = Sequential()
model.add(Conv2D(32,(7, 7), activation='relu', padding='valid', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
model.add(Conv2D(64,(5,5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
model.add(Conv2D(128,(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
model.add(Conv2D(256,(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
model.add(Conv2D(512,(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(4, activation='softmax'))

adam = tf.keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])  #sparse is used when labels are 0, 1, 2 instead of [1, 0, 0] etc.
model.summary()

X_train = np.load('datasets/5s/X_train.npy')
y_train = np.load('datasets/5s/Y_train.npy')
X_val = np.load('datasets/5s/X_val.npy')
y_val = np.load('datasets/5s/Y_val.npy')
X_test = np.load('datasets/5s/X_test.npy')
y_test = np.load('datasets/5s/Y_test.npy')
#y_key = np.load('datasets/3s/Y_key.npy')

batch_size = 64
num_epochs = 100

X_train, y_train = shuffle(X_train, y_train)  # This fixes an issue where validation accuracy was stuck at 0.33 when using generator

NUM_TRAIN_EXAMPLES = X_train.shape[0]
NUM_TEST_EXAMPLES = X_val.shape[0]
NUM_STEPS = NUM_TRAIN_EXAMPLES // batch_size
NUM_VAL_STEPS = NUM_TEST_EXAMPLES // batch_size
train_generator = batch_generator(X_train, y_train, batch_size=batch_size, num_steps=NUM_STEPS)
#test_generator = mel_generator(X_test, y_test, batch_size=540, num_steps=1)

history = model.fit(
          x=train_generator, 
          steps_per_epoch = NUM_STEPS, 
          epochs=num_epochs, 
          validation_data=(X_val, y_val),
          verbose=1
          )

# history = model.fit(X_train, 
#           y_train, 
#           batch_size=batch_size, 
#           epochs=num_epochs, 
#           validation_data=(X_test, y_test),
#           verbose=1)

model.save('model_5s_{}_epochs.h5'.format(num_epochs))
pd.DataFrame.from_dict(history.history).to_csv('history_5s_{}_epochs.csv'.format(num_epochs),index=False)

model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
y_pred1 = y_pred.argmax(axis = 1)

# Needs to be single values classes for confusion matrix
# y_test1 = y_test.argmax(axis = 1)

cm = confusion_matrix(y_test, y_pred1)
print(np.around(cm/cm.sum(axis=1, keepdims=True)*100,1))
print(classification_report(y_test, y_pred1))

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("Training Complete, results shown above")
