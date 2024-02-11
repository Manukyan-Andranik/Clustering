from sklearn.cluster import KMeans, SpectralClustering, DBSCAN

class Model:
    def __init__(self, model = KMeans()):
        self.cluster = model

    def fit(self, X_train):
        self.cluster.fit(X=X_train)

    def fit_transform(self, X_train, y = None, sample_waigth = None):
        return self.cluster.fit_transform(X_train, y, sample_waigth)

    def get_params(self, deep = True):
        return self.cluster.get_params(deep)

    def fit_predict(self, X_train):
        self.cluster.fit(X_train)
        labels = self.cluster.labels_
        return labels

    def score(self, X, y = None, sample_waight = None):
        return self.cluster.score(X, y, sample_waight)
