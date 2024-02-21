import numpy as np
import matplotlib.pyplot as plt

def plot_2D_separator(classifier , x , fill=False , ax=None , eps=None):
    if eps is None:
        eps = x.std() / 2
    
    x_min , x_max = x[:,0].min()-eps , x[:,0].max()+eps
    y_min , y_max = x[:,1].min()-eps , x[:,1].max()+eps
    x_linespace = np.linspace(x_min , x_max , 100)
    y_linespace = np.linspace(y_min , y_max , 100)

    x1 , x2 = np.meshgrid(x_linespace , y_linespace)
    x_grid = np.c_[x1.ravel() , x2.ravel()]
    try:
        decision_values = classifier.decision_function(x_grid)
        levels = [0]
        fill_levels = [decision_values.min() , 0 , decision_values.max()]
    except  AttributeError:
        decision_values = classifier.predict_proba(x_grid)[:,0]
        levels = [0.5]
        fill_levels = [0, 0.5 ,1]
    
    if ax is None:
        ax = plt.gca()
    if fill:
        ax.contourf(x1,x2,decision_values.reshape(x1.shape) , levels = fill_levels , colors = ['blue' , 'red'])
    else:
        ax.contour(x1,x2,decision_values.reshape(x1.shape) , levels = levels , colors = ['black'])
    
    ax.set_xlim(x_min , x_max)
    ax.set_ylim(y_min , y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.linear_model import LogisticRegression
    x , y = make_blobs(centers=2 , random_state=2)
    clf = LogisticRegression().fit(x,y)
    plot_2D_separator(clf , x , fill=True)
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.show()