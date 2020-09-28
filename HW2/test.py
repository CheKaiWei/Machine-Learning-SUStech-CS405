import numpy as np
class KNNClassfier(object):

    def __init__(self, k=5, distance='euc'):
        self.k = k
        self.distance = distance
        self.x = None
        self.y = None
        
    def fit(self,X, Y):
        '''
        X : array-like [n_samples,shape]
        Y : array-like [n_samples,1]
        '''        
        self.x = X
        self.y = Y
    def predict(self,X_test):
        '''
        X_test : array-like [n_samples,shape]
        Y_test : array-like [n_samples,1]
        output : array-like [n_samples,1]
        '''  
        output = np.zeros((X_test.shape[0],1))
        for i in range(X_test.shape[0]):
            dis = [] 
            for j in range(self.x.shape[0]):
                if self.distance == 'euc': # 欧式距离
                    dis.append(np.linalg.norm(X_test[i,:]-self.x[j,:]))
            labels = []
            index=sorted(range(len(dis)), key=dis.__getitem__)
            for j in range(self.k):
                labels.append(self.y[index[j]])
            counts = []
            for label in labels:
                counts.append(labels.count(label))
            output[i] = labels[np.argmax(counts)]
        print(len(dis))
        print(len(X_test))
        return output
    def score(self,x,y):
        pred = self.predict(x)
        err = 0.0
        for i in range(x.shape[0]):
            if pred[i]!=y[i]:
                err = err+1
        return 1-float(err/x.shape[0])


if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    # x = np.array([[0.5,0.4],[0.1,0.2],[0.7,0.8],[0.2,0.1],[0.4,0.6],[0.9,0.9],[1,1]]).reshape(-1,2)
    # y = np.array([0,1,0,1,0,1,1]).reshape(-1,1)
    clf = KNNClassfier(k=3)
    clf.fit(x,y)
    print('myknn score:',clf.score(x,y))

    from sklearn.neighbors import KNeighborsClassifier
    clf_sklearn = KNeighborsClassifier(n_neighbors=3)
    clf_sklearn.fit(x,y)
    print('sklearn score:',clf_sklearn.score(x,y))