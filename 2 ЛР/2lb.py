import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn import feature_extraction
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix as confusion_matrix_metric
from sklearn.svm import SVC

# TODO: 1. Загрузить анализируемые данные, выданные преподавателем
# ham - real msg; smap - fake msg
data = pd.read_csv('spam.csv', encoding="latin-1").drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# TODO: 2. Построить круговую диаграмму для принимаемых значений целевой переменной
explode = (0, 0.15)
target = pd.Series(data['v1']).value_counts()
target.plot(kind='pie', autopct='%1.1f%%', shadow=True, startangle=45, explode=explode)
plt.title('Pie diagram')
plt.ylabel('')

# TODO: 3. Построить столбиковую диаграмму для двадцати наиболее часто встречающихся слов в обоих классах
ham_words = Counter(''.join(data[data['v1'] == 'ham']['v2']).split()).most_common(20)
spam_words = Counter(''.join(data[data['v1'] == 'spam']['v2']).split()).most_common(20)
df_ham_words = pd.DataFrame(ham_words)
df_ham_words = df_ham_words.rename(columns={0: 'Слова в не спаме', 1: 'count'})

df_spam_words = pd.DataFrame(spam_words)
df_spam_words = df_spam_words.rename(columns={0: 'Слова в спаме', 1: 'count'})

y_pos = np.arange(len(df_ham_words['Слова в не спаме']))
df_ham_words.plot.bar(legend=False, title='Самые часто встречаемые слова в не спам сообщениях', xlabel='words', ylabel='number')
plt.xticks(y_pos, df_ham_words['Слова в не спаме'])
plt.tight_layout()

y_pos = np.arange(len(df_spam_words['Слова в спаме']))
df_spam_words.plot.bar(legend=False, title='Самые часто встречаемые слова в спам сообщениях', xlabel='words', ylabel='number')
plt.xticks(y_pos, df_spam_words['Слова в спаме'])
plt.tight_layout()

# TODO: 4. Выполнить токенизацию текстового признака, исключив неинформативные часто встречающиеся слова
tokenizer = feature_extraction.text.CountVectorizer(stop_words='english')
X = tokenizer.fit_transform(data['v2'])

# TODO: 5. Найти оптимальный параметр сглаживания alpha для наивного байесовского классификатора по метрикам
#  precision и accuracy.
train_score = []
test_recall = []
test_precision = []
test_score = []
data['v1'] = data['v1'].map({'spam': 1, 'ham': 0})
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, data['v1'], test_size=0.44)
alpha_range = np.arange(0.1, 20, 0.1)
# Попробуйте найти оптимальный параметр alpha в диапазоне от 0,1 до 20 с шагом 0,1:
for alpha in alpha_range:
    mnb = MultinomialNB(alpha=alpha).fit(X_train, Y_train)
    y_train = mnb.predict(X_train)
    y_test = mnb.predict(X_test)
    train_score.append(metrics.accuracy_score(Y_train, y_train))
    test_recall.append(metrics.recall_score(Y_test, y_test))
    test_precision.append(metrics.precision_score(Y_test, y_test))
    test_score.append(metrics.accuracy_score(Y_test, y_test))
    y_train = 0
    y_test = 0

matrix = np.matrix(np.c_[alpha_range, train_score, test_score, test_recall, test_precision])
models = pd.DataFrame(data=matrix,
                      columns=['alpha', 'train accuracy', 'test accuracy', 'test recall', 'test precision'])

# TODO: 6. Построить зависимость метрики accuracy на обучающих и тестовых данных от варьируемого параметра
f = plt.figure(figsize=(7, 5))
plt.plot(alpha_range, models['train accuracy'], "c", label='train accuracy')
plt.plot(alpha_range, models['test accuracy'], "r", label='test accuracy')
plt.plot(alpha_range, models['test recall'], "b", label='test recall')
plt.plot(alpha_range, models['test precision'], "g", label='test precision')
plt.ylabel('metrics')
plt.xlabel('alpha')
plt.title('Зависимость метрики accuracy для байесовского классификатора')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# TODO: Построить матрицы ошибок для модели с оптимальным выбранным параметром 2
best_alpha = models['alpha'][models['test precision'].idxmax()]
print("Best alpha: ", best_alpha, '\n')
mnb = MultinomialNB(alpha=best_alpha).fit(X_train, Y_train)

print("Confusion matrix for MultinomialNB:")
confusion_matrix = confusion_matrix_metric(Y_test, mnb.predict(X_test))
print(pd.DataFrame(data=confusion_matrix, columns=['predicted ham', 'predicted spam'],
                   index=['actual ham', 'actual spam']))

# TODO: 7. Построить ROC-кривую и рассчитать метрику AUC-ROC
y_pred_pr = mnb.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_pr)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC-кривая для байесовского классификатора')
plt.grid()
plt.show()

# TODO: 8. Найти оптимальный параметр регуляризатора С для модели опорных векторов по метрикам precision и accuracy
c_range = np.arange(0.1, 3, 0.1)
train_score = []
test_recall = []
test_precision = []
test_score = []
# Попробуйте найти оптимальный параметр c в диапазоне от 0,1 до 3 с шагом 0,1:
for C in c_range:
    svc = SVC(C=C).fit(X_train, Y_train)
    y_train = svc.predict(X_train)
    y_test = svc.predict(X_test)
    train_score.append(metrics.accuracy_score(Y_train, Y_train))
    test_recall.append(metrics.recall_score(Y_test, y_test))
    test_precision.append(metrics.precision_score(Y_test, y_test, zero_division=1))
    test_score.append(metrics.accuracy_score(Y_test, y_test))


# TODO: 9. Повторить пункты 6 и 7 для модели опорных векторов
matrix = np.matrix(np.c_[c_range, train_score, test_score, test_recall, test_precision])
models = pd.DataFrame(data=matrix, columns=['C', 'train accuracy', 'test accuracy', 'test recall', 'test precision'])
f = plt.figure(figsize=(7, 5))
plt.plot(c_range, models['train accuracy'], "c", label='train accuracy')
plt.plot(c_range, models['test accuracy'], "r", label='test accuracy')
plt.plot(c_range, models['test recall'], "b", label='test recall')
plt.plot(c_range, models['test precision'], "g", label='test precision')
plt.ylabel('metrics')
plt.xlabel('C')
plt.title('Зависимость метрики accuracy для модели опорных векторов')
plt.legend()
plt.grid()
plt.show()

best_c = models['C'][models['test precision'].idxmax()]
print("Best c: ", best_c, '\n')
svc = SVC(C=best_c, probability=True).fit(X_train, Y_train)

print("Confusion matrix for SVC:")
confusion_matrix = confusion_matrix_metric(Y_test, svc.predict(X_test))
print(pd.DataFrame(data=confusion_matrix, columns=['predicted ham', 'predicted spam'],
                   index=['actual ham', 'actual spam']))

y_pred_pr = svc.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, y_pred_pr)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC-кривая для модели опорных векторов')
plt.grid()
plt.show()
