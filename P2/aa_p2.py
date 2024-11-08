import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from itertools import cycle
from scipy.stats import multivariate_normal as mvn


#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
def plotModel(x, t, title, centers=None, new_figure=False):
    color_iter = cycle(['r', 'g', 'b', 'c', 'm'])
    labels = np.unique(t)
    
    if new_figure==True:
        plt.figure(figsize=(4, 4))
        
    for i, (label, color) in enumerate(zip(labels, color_iter)):
        plt.plot(x[t==label, 0], x[t==label, 1], '.', color=color)

    if centers is not None:
        color_iter = cycle(['r', 'g', 'b', 'c', 'm'])
        for i, (label, color) in enumerate(zip(labels, color_iter)):
            plt.plot(centers[i, 0], centers[i, 1], 'o', markersize=12, color='k')
            plt.plot(centers[i, 0], centers[i, 1], 'o', markersize=9, color=color)

    if new_figure==True:
        plt.title(title)
        plt.grid(True)
        plt.show()


#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
def plotModelGMM(x, clf, title):
    color_iter = cycle(['r', 'g', 'b', 'c', 'm'])

    y = np.argmax(clf.predict(x), axis=1)
    plt.figure(figsize=(6,6))
    splot = plt.subplot(1, 1, 1)

    for i, (mean, covar, color) in enumerate(zip(clf.centers, clf.covariances, color_iter)):
        v, w = np.linalg.eigh(covar)
        u = w[0] / np.linalg.norm(w[0])
        plt.plot(x[y == i, 0], x[y == i, 1], '.', color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / (u[0] + 1.e-20))
        angle = 180 * angle / np.pi  # convert to degrees
        ell = matplotlib.patches.Ellipse(mean, v[0]*3.0, v[1]*3.0, angle=180+angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()


#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
class KMeans:
    def __init__(self, num_clusters, num_iters, random_state=None):
        self.num_clusters = num_clusters
        self.num_iters = num_iters
        self.centers = []
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, x: np.ndarray):
        """
        Recibe un array de numpy de dimensiones (n, d), donde n es el número
        de puntos y d es su dimensión, y ejecuta el algoritmo K-Means para ajustar
        self.num_clusters clusters a los datos. El número de iteraciones del
        algoritmo vendrá dado por self.num_iters. Los centros de los clusters se
        almacenarán en la lista self.centers.
        """
        # Elección aleatoria de los centroides entre puntos del dataset
        # Garantiza que a cada centroide se asocie al menos un punto
        self.centers = x[np.random.choice(x.shape[0], size=self.num_clusters)]
        
        for _ in range(self.num_iters):
            # Matriz de distancias de todos los puntos
            distances = cdist(x, self.centers, metric='euclidean')
            
            # Asigna cada punto a un centroide (minimo de distancias en cada fila)
            tags = np.argmin(distances, axis=1)

            # Calcula los nuevos centroides
            for i in range(self.num_clusters):
                self.centers[i] = np.mean(x[np.where(tags == i)[0]], axis=0)


    def predict(self, x):
        """
        Recibe un array de numpy de dimensiones (n, d), donde n es el número
        de puntos y d es su dimensión, y devuelve un array de numpy de dimensiones
        (n,) con el índice del cluster al que pertenece cada punto. El método
        predict no se puede invocar sobre un objeto si no se ha hecho fit
        previamente.
        """
        if len(self.centers) != self.num_clusters:
            raise ValueError(f'Fit method was not called!')
        distances = cdist(x, self.centers, metric='euclidean')
        return np.argmin(distances, axis=1)

    def get_centers(self):
        return self.centers


#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
class GMM:
    def __init__(self, num_components, num_iters, random_state=None):
        self.num_components = num_components
        self.num_iters = num_iters
        self.centers = []
        self.covariances = []
        self.weights = []
        if random_state is not None:
            np.random.seed(random_state)
        
    def fit(self, x):
        """
        Recibe un array de numpy de dimensiones (n, d), donde n es el número
        de puntos y d es su dimensión, y ejecuta el algoritmo EM para ajustar
        una mezcla de self.num_components componentes gausianas a los datos.
        El número de iteraciones del algoritmo vendrá dado por self.num_iters.
        Los centros, covarianzas y prioris de cada componente se almacenarán
        en las listas self.centers, self.covariances y self.weights,
        respectivamente.
        """
        pass

    def predict(self, x):
        """
        Recibe un array de numpy de dimensiones (n, d), donde n es el número
        de puntos y d es su dimensión, y devuelve un array de numpy de dimensiones
        (n, self.num_components) con las probabilidades de pertenencia de cada
        punto a cada una de las componentes (la suma por filas debe ser 1). El
        método predict no se puede invocar sobre un objeto si no se ha hecho fit
        previamente.
        """
        pass

