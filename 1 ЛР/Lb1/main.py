import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pnd
import numpy as np


def neighbour_classifier(k, Y, X):
    # Выбрали количество соседей для модели
    model = KNeighborsClassifier(n_neighbors=k)

    # Обучили модель
    model.fit(X_train, Y_train)

    # Предсказываем с помощью обученной модели
    predictions = model.predict(X)

    # Оцениваем правильность предсказаний
    result = accuracy_score(Y, predictions)
    # print('Правильность предсказаний текущей модели: ', result)
    return result


def cross_validation(attribute_x, answer_y):
    # Генератор разбиений для кросс-валидации
    kf = KFold(n_splits=5, shuffle=True)
    accuracyKFold = []
    kMax = -1
    kMaxIndex = 0
    # Поиск оптимального K
    for i in range(1, 50):
        model = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(model, attribute_x, answer_y, cv=kf, scoring='accuracy')
        accuracyKFold.append(score.mean())
        if kMax < score.mean():
            kMax = score.mean()
            kMaxIndex = i - 1
    return kMaxIndex, kMax, accuracyKFold


def logistic_regression(X_test, Y_test, attribute_x, answer_y):
    C = np.arange(0.01, 1, 0.01)
    c_max_score = -1
    c_max = 0
    c_array = []
    for c in C:
        logisticRegression = LogisticRegression(random_state=17, fit_intercept=True, n_jobs=-1, max_iter=10_000, C=c).fit(attribute_x,
                                                                                                      answer_y)
        score_c_current = logisticRegression.score(X_test, Y_test)
        c_array.append(score_c_current)

        if c_max_score < score_c_current:
            c_max_score = score_c_current
            c_max = c
    return c_max, c_max_score, c_array


data = pnd.read_csv('breast_cancer.csv')  # Метод возвращает объект класса DataFrame

# Заменили B (доброкачественная) на 0, М (злокачественная) на 1
answer_y = data['diagnosis'].map({'B': 0, 'M': 1})
attribute_x = data.drop(columns=['id', 'diagnosis', 'Unnamed: 32'], axis=1)

# Разделим выборки на обучающую и тестовую
X_train, X_test, Y_train, Y_test = train_test_split(attribute_x, answer_y, test_size=0.33, random_state=42)

accuracyArrayTestY = []
accuracyArrayTrainY = []
for i in range(1, 50):
    accuracyArrayTrainY.append(neighbour_classifier(i, Y_train, X_train))
    accuracyArrayTestY.append(neighbour_classifier(i, Y_test, X_test))

kMaxIndex, kMax, accuracyKFold = cross_validation(attribute_x, answer_y)
print('Оптимальное значение К для метода кросс валидации до масштабирования: ', kMaxIndex, ' точность: ', kMax)

c_max, c_max_score, c_array = logistic_regression(X_test, Y_test, attribute_x, answer_y)
print('Оптимальное значение C для метода логистической регрессии до масштабирования: ', c_max, ' точность: ', c_max_score)

# Масштабирование признаков
scaler = StandardScaler()
attribute_x = pnd.DataFrame(scaler.fit_transform(attribute_x, answer_y), columns=attribute_x.columns)
X_test = pnd.DataFrame(scaler.fit_transform(X_test, Y_test), columns=X_test.columns)

print('--------------------------------------------------------------------------------------')

kMaxIndex, kMax, accuracyKFold2 = cross_validation(attribute_x, answer_y)
print('Оптимальное значение К для метода кросс валидации после масштабирования: ', kMaxIndex, ' точность: ', kMax)

c_max, c_max_score, c_array2 = logistic_regression(X_test, Y_test, attribute_x, answer_y)
print('Оптимальное значение C для метода логистической регрессии после масштабирования: ', c_max, ' точность: ', c_max_score)

C = np.arange(0.01, 1, 0.01)

plt.figure(figsize=(12, 7))
# plt.subplot(3, 1, 1)
# plt.plot(accuracyArrayTestY, label='Результаты тестирования')
# plt.plot(accuracyArrayTrainY, label='Результаты обучения')
# plt.legend()
# plt.grid()
# plt.subplot(3, 1, 2)
# plt.plot(accuracyKFold, label='Результаты кросс-валидации разбиением')
# plt.plot(accuracyKFold2, label='Результаты кросс-валидации разбиением после масштабирования')
# plt.grid()
# plt.legend()
# plt.subplot(1, 1, 1)
plt.plot(C, c_array, label='Метод логистической регрессии')
plt.legend()
plt.grid()
plt.show()
plt.plot(C, c_array2, label='Метод логистической регрессии после масштабирования')
plt.legend()
plt.grid()
plt.show()

