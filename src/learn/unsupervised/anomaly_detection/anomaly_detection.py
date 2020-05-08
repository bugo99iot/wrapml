
# https://scikit-learn.org/stable/modules/outlier_detection.html
def test():
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    X = [[1], [2], [3], [2], [2], [1], [2]]
    clf = IsolationForest(random_state=0).fit(X)
    print(clf.predict([[1], [2], [5]]))