import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
import data_helpers


#Fetching data
x_text, y_train = data_helpers.load_data_and_labels_forsvm('data/train.csv')
x_text1, y_train1 = data_helpers.load_data_and_labels_forsvm('data/dev.csv')
x_test, y_test = data_helpers.load_data_and_labels_forsvm('data/test.csv')
#combine data for doing CV
#x_text = np.concatenate([x_text, x_text1], 0)
#y_train = np.concatenate([y_train, y_train1], 0)

vectorizer = CountVectorizer(min_df=1, ngram_range=(1,2),
                             stop_words = 'english', lowercase = True)
x_train = vectorizer.fit_transform(x_text)

# C_range = np.logspace(-2, 10, 15)
# gamma_range = np.logspace(-9, 3, 15)
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=2)
# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
# grid.fit(x_train, y_train)
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

clf = SVC(C = 26.826957952797247, gamma=0.001)
clf.fit(x_train, y_train)

x_test = vectorizer.transform(x_test)
predictions = clf.predict(x_test)
print(predictions)
correct_predictions = float(sum(predictions == y_test))
print("Accuracy on test set of length %d is %f"
      % (len(y_test), correct_predictions/len(y_test)))