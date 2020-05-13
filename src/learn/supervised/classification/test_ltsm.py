# todo:
#  - https://www.curiousily.com/posts/time-series-classification-for-human-activity-recognition-with-lstms-in-keras/#classifying-human-activity
#  print(X_train.shape, y_train.shape)
#  (22454, 200, 3) (22454, 1)

from src.imports.science import random, np
from src.imports.learn import Sequential, Dense, Dropout, LSTM, Bidirectional
from src.imports.learn import MinMaxScaler, OneHotEncoder, to_categorical

from src.learn.supervised.classification import TrainClassificationModel


def main():

    x = []  # (n, m, p)
    y = []  # (n, 1)

    n = 5000
    m = 10
    p = 1

    for ni in range(n):

        val = random.choice([1, 0])

        row = [[float(val)]*p]*m

        label = val  # todo: convert categorical

        x.append(row)
        y.append(label)

    x = np.array(x)
    y = np.array(y).reshape((n, 1))

    print(x.shape, y.shape)

    assert x.shape == (n, m, p)
    assert y.shape == (n, 1)

    do_rescale = True
    if do_rescale:
        x_shape = x.shape
        x = x.reshape((x_shape[0], x_shape[1]*x_shape[2]))
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
        x = x.reshape(x_shape)

    tcm = TrainClassificationModel(x=x, y=y)

    tcm.train_with_random_forests()
    print(tcm.report)

    do_ohe = True
    if do_ohe:
        y = to_categorical(y)
        #onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        #y = onehot_encoder.fit_transform(y)

    x = x.astype('float64')
    #y = y.astype('int')

    print(x.shape, y.shape)

    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))

    model = Sequential()
    model.add(LSTM(units=32, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=100, activation='relu'))
    # last layr notes
    # softmax should be used with categorical crossentropy (multiclass)
    # relu should be used with binary crossentropy (binary)
    # n neurons must be = n of classes with softmax
    # https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax
    model.add(Dense(y.shape[1], activation='softmax'))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )

    history = model.fit(
        x, y,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        shuffle=True
    )

    # model.evaluate(x, y)

    return


if __name__ == "__main__":

    main()
