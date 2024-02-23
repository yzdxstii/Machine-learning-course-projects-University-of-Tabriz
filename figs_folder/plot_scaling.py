import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler , MinMaxScaler ,   normalize , RobustScaler
from sklearn.model_selection import train_test_split
from .plot_helper import color_map2

def plot_scaling():
    x , y = make_blobs(n_samples=50 , centers=2 , random_state=2 , cluster_std=1)
    x += 3

    plt.figure(figsize=(15 , 8))
    main_ax = plt.subplot2grid((2,4) , (0,0) , rowspan=2 , colspan=2)
    plt.scatter(main_ax , x[:,0] , x[:,1] , c=y , cmap=color_map2 , s=60)
    maxx = np.abs(x[:,0]).max()
    maxy = np.abs(x[:,1]).max()
     
    main_ax.set_xlim(-maxx+1 , maxx+1)
    main_ax.set_ylim(-maxy+1 , maxy+1)
    main_ax.settitle('original data')
    other_axes = [plt.subplot2grid((2,4) , (i,j)) for j in range(2,4)  for i in range(2)]

