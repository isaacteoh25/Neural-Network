##############@@@@@@@@####################
## STEP 1: DATA IMPORT & PRE-PROCESSING
##############@@@@@@@@####################
# Importing the libraries
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split as tts, train_test_split, KFold, cross_val_score
# Importing the Keras Labraries & Packages
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import TruncatedNormal
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import style
import matplotlib.pyplot as plt
style.use('ggplot')
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
############################
# 1.2: Import dataset
############################


# File dataset
test = 'dataset/CCS516_Assignment 2_Data.xlsx'

df = pd.read_excel(test, sheet_name='DataCCS516')
df.drop(['%Skor ', 'Input B', 'Jenis Pemeriksaan', 'Nama Pemeriksaan'], 1, inplace=True)
df = df.dropna(subset=['Input A', 'Daerah'])

###########################
# 1.3: Data-Type Correction
###########################
# Mapping the dataset to their correct datatypes.
df = df.astype({'Kod Sekolah': 'category', 'Input A': 'float64', 'Negeri': 'category', 'Kod PPD': 'category',
                'Daerah': 'category', 'Lokasi Sekolah': 'category', 'Gred Sekolah': 'category',
                'Jenis Sekolah': 'category',
                'Mata Pelajaran Dicerap ': 'category', 'Tahun/Tingkatan': 'category', 'Jantina': 'category',
                'Keturunan': 'category', 'Akademik': 'category', 'Ikhtisas': 'category', 'Opsyen': 'category',
                'PengalamanAjar': 'float64', 'Gred': 'category', 'Jawatan': 'category',
                'BilanganMurid': 'category', 'Skor1': 'float64', 'Skor2': 'float64', 'Skor3': 'float64',
                'Skor4': 'float64',
                'Skor5': 'float64', 'Skor6': 'category', 'Skor7': 'float64', 'Skor8': 'float64', 'Taraf ': 'category'})


# Lets do some integrity datatype checks & prints to the console
print(df['Kod Sekolah'].dtypes)
print(df['Negeri'].dtypes)
print(df.head(3))

###########################
# 1.5: Encoding Categorical Fields
###########################


Y = df.iloc[:, 27]
lex18 = LabelEncoder()

lex18.fit(Y)
encoded_Y = lex18.transform(Y)

y = np_utils.to_categorical(encoded_Y)
X = df.iloc[:, 0:27]


lex1 = LabelEncoder()
X['Kod Sekolah'] = lex1.fit_transform(X['Kod Sekolah'])
lex2 = LabelEncoder()
X['Negeri'] = lex2.fit_transform(X['Negeri'])
lex11 = LabelEncoder()
X['Kod PPD'] = lex11.fit_transform(X['Kod PPD'])
lex3 = LabelEncoder()
X['Daerah'] = lex3.fit_transform(X['Daerah'])
lex4 = LabelEncoder()
X['Lokasi Sekolah'] = lex4.fit_transform(X['Lokasi Sekolah'])
lex5 = LabelEncoder()
X['Gred Sekolah'] = lex5.fit_transform(X['Gred Sekolah'])
lex6 = LabelEncoder()
X['Jenis Sekolah'] = lex6.fit_transform(X['Jenis Sekolah'])
lex7 = LabelEncoder()
X['Mata Pelajaran Dicerap '] = lex7.fit_transform(X['Mata Pelajaran Dicerap '])
lex8 = LabelEncoder()
X['Tahun/Tingkatan'] = lex8.fit_transform(X['Tahun/Tingkatan'])
lex9 = LabelEncoder()
X['Jantina'] = lex9.fit_transform(X['Jantina'])

lex10 = LabelEncoder()
X['Keturunan'] = lex10.fit_transform(X['Keturunan'])

lex13 = LabelEncoder()
X['Akademik'] = lex13.fit_transform(X['Akademik'])
lex14 = LabelEncoder()
X['Ikhtisas'] = lex14.fit_transform(X['Ikhtisas'])
lex15 = LabelEncoder()
X['Opsyen'] = lex15.fit_transform(X['Opsyen'])
lex16 = LabelEncoder()
X['Gred'] = lex16.fit_transform(X['Gred'])
lex17 = LabelEncoder()
X['Jawatan'] = lex17.fit_transform(X['Jawatan'])


def baseline_model ():
    # create model
    model = Sequential()
    init = TruncatedNormal(stddev=0.01, seed=10)
    # config model: single-input model with 4 classes (categorical classification)
    model.add(Dense(units=50, input_dim=27, activation='relu',init = TruncatedNormal(stddev=0.01, seed=10)))
    model.add(Dense(units=4, activation='softmax', kernel_initializer=init))
    # Compile model
    # optimizer
    adam = Adam(lr=0.007)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

###########################
# 1.6: Split Training & Test Sets
###########################
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
X_train, X_test, Y_train, Y_test =  tts(X, y, test_size=0.33, random_state=0)
# Train the model, iterating on the data in batches of 32 samples
history = estimator.fit(X_train, Y_train, epochs=800, validation_data=(X_test, Y_test), shuffle=False, verbose=0)
y_pred = estimator.predict(X)

print("\nPrediction: \n" , lex18.inverse_transform(y_pred))
print("\nTest set actual labels:\n" , lex18.inverse_transform(np.argmax(y, axis=-1)))

acc = accuracy_score(np.argmax(y, axis=-1), y_pred) * 100.00
cm = confusion_matrix(np.argmax(y, axis=-1), y_pred)
print(f'Overall Accuracy : {acc}')
print(f'confusion_matrix : {cm}')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('Overall accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
outputfile = 'Export//prediction.txt'
f = open(outputfile, 'w+')
# f.write('Accuracy:%d\r\n' % acc)
# f.write(f'confusion_matrix: {cm}\r\n')
f.write(f'Prediction: {lex18.inverse_transform(y_pred)}\r\n')
f.write(f'Test set actual labels: {lex18.inverse_transform(np.argmax(y, axis=-1))}\r\n')
f.close()