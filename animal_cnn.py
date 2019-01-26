import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

# main
def main():
    X_train, X_test, y_train, y_test = np.load("./animal.npy")
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256 # 最大値で割る
    y_train = np_utils.to_categorical(y_train, num_classes) #one-hot-vector（正解地以外は0のvector）
    y_test = np_utils.to_categorical(y_test, num_classes) # monkey,boar,crowそれぞれで[1, 0, 0], [0, 1, 0]のようになる

    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)

def model_train(X, y):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding ='same', input_shape=X.shape[1:])) # 1以降のshapeを取る
    model.add(Activation('relu')) # 活性化関数
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu')) # 活性化関数
    model.add(MaxPooling2D(pool_size=(2,2))) # 一番大きい値を取り出す、特徴を強調する
    model.add(Dropout(0.25)) # 25%を削除し、データの偏りを減らす

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu')) # 活性化関数
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu')) # 活性化関数
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512)) #全て結合層
    model.add(Activation('relu')) # 負の値を捨てる
    model.add(Dropout(0.5)) # 半分捨てる
    model.add(Dense(3)) #最後の出力層のノードが3
    model.add(Activation('softmax')) #それぞれの画像である確率を合計1で表す

    # トレーニング時のアルゴリズム, 一回のトレーニングあたり、学習率を下げていく
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6) #最適化の手法の宣言
    # 評価手法
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) #どれ位正答したか

    model.fit(X, y, batch_size=32, nb_epochs=100) # トレーニングを何回？50回

    # modelの保存
    model.save=('./animal_cnn.h5')
    return model

def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])

# mainが呼ばれていたら、mainを呼び出す
if __name__ == "__main__":
    main()
