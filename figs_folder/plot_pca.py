from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np




def plot_pca_illustration():
    rnd = np.random.RandomState(2)
    x_ = rnd.normal(size=(300,2))
    x_blob = np.dot(x_ , rnd.normal(size=(2,2)))  +  rnd.normal(size=2)

    pca = PCA(n_components=2)
    pca.fit(x_blob)
    x_pca = pca.transform(x_blob)

    s = x_pca.std(axis=0)

    fig , axes = plt.subplots(2 , 2 , figsize=(10,10))
    axes = axes.ravel()


    axes[0].set_title("Original data")
    axes[0].scatter(x_blob[:, 0] , x_blob[:, 1] , c=x_pca[:, 0] , linewidths=0 , s=60 , cmap='viridis')
    axes[0].set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    axes[0].arrow(pca.mean_[0] , pca.mean_[1] , s[0] * pca.components_[0, 0] , s[0] * pca.components_[0, 1] , width=.1 , head_width=.3 , color='k')
    axes[0].arrow(pca.mean_[0] , pca.mean_[1] , s[1] * pca.components_[1, 0] , s[1] * pca.components_[1, 1] , width=.1 , head_width=.3 , color='k')
    axes[0].text(-1.5 , -.5 , "Component 2" , size=14)
    axes[0].text(-4 , -4 , "Component 1" , size=14)
    axes[0].set_aspect('equal')


    axes[1].set_title("Transformed data")
    axes[1].scatter(x_pca[:, 0], x_pca[:, 1] , c=x_pca[:, 0] , linewidths=0 , s=60 , cmap='viridis')
    axes[1].set_xlabel("First principal component")
    axes[1].set_ylabel("Second principal component")
    axes[1].set_aspect('equal')
    axes[1].set_ylim(-8, 8)

    pca = PCA(n_components=1)
    pca.fit(x_blob)
    x_inverse = pca.inverse_transform(pca.transform(x_blob))

    axes[2].set_title("Transformed data w/ second component dropped")
    axes[2].scatter(x_pca[:, 0] , np.zeros(x_pca.shape[0]) , c=x_pca[:, 0] ,linewidths=0 , s=60 , cmap='viridis')
    axes[2].set_xlabel("First principal component")
    axes[2].set_aspect('equal')
    axes[2].set_ylim(-8, 8)

    axes[3].set_title("Back-rotation using only first component")
    axes[3].scatter(x_inverse[:, 0] , x_inverse[:, 1] , c=x_pca[:, 0] , linewidths=0, s=60 , cmap='viridis')
    axes[3].set_xlabel("feature 1")
    axes[3].set_ylabel("feature 2")
    axes[3].set_aspect('equal')
    axes[3].set_xlim(-8, 4)
    axes[3].set_ylim(-8, 4)





def plot_pca_whitening():
    rnd = np.random.RandomState(2)
    x_ = rnd.normal(size=(300,2))
    x_blob = np.dot(x_ , rnd.normal(size=(2,2))) + rnd.normal(size=2)

    pca = PCA(whiten= True)
    pca.fit(x_blob)
    x_pca = pca.transform(x_blob)

    fig , axes = plt.subplot(1 , 2 , figsize=(10,10))
    axes = axes.ravel()
    axes[0].set_title("Original data")
    axes[0].scatter(x_blob[:, 0] , x_blob[:, 1] , c= x_pca[:, 0] , linewidths=0 , s=60 , cmap='viridis')
    axes[0].set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    axes[0].set_aspect('equal')

    axes[1].set_title("Whitened data")
    axes[1].scatter(x_pca[:, 0] , x_pca[:, 1] , c=x_pca[:, 0] , linewidths=0 , s=60 , cmap='viridis')
    axes[1].set_xlabel("First principal component")
    axes[1].set_ylabel("Second principal component")
    axes[1].set_aspect('equal')
    axes[1].set_xlim(-3, 4)
