# designed to do some exploratory plots of AutoDQM
import json
import os
import uproot
import numpy as np
import awkward
import plotly
import matplotlib.pyplot as plt

def plotter(data,  name, name_full, folder):
    
    mean = np.mean( np.array(data) )
    std  = np.std( np.array(data) )

    # Plot the histogram
    plt.figure()
    plt.hist(data, bins = 50, range=(mean - 0.5*std, mean + 0.5*std), histtype='step', linewidth=2)
    plt.xlabel(str(name_full))
    plt.ylabel( 'Events' )
    #plt.title('Histogram Title')
    plt.grid(True)
    plt.savefig( './plots/'+ str(folder) + '/' + str(name) + '.png')
    plt.close()

def test(data, edges, name, name_full, folder):

    # Plot the histogram
    plt.figure()
    plt.bar( edges[:,0], data,  width=(edges[:,1] - edges[:,0]), align='edge')
    plt.xlabel(str(name_full))
    plt.ylabel( 'Events' )
    #plt.title('Histogram Title')
    plt.grid(True)
    plt.text(0.921, .99, 'Number of entries: ' + str(np.sum(data)), transform=plt.gca().transAxes, va='top', ha='right', fontsize=10)
    plt.savefig( './plots/'+ str(folder) + '/' + str(name) + '.png')
    plt.close()


def test_2d(x,y,z,folder,name,name_xaxis):

    fig ,ax = plt.subplots()
    # Plot the histogram using imshow
    cax = ax.matshow(z.T,  origin='upper', aspect='auto',  cmap = 'bwr' , vmin = 0.0, vmax = np.max(z))

    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))
    
    ax.set_xticklabels(x , rotation=90 )
    ax.set_yticklabels(y.T , rotation=0  )

    cbar = fig.colorbar(cax)
    #cbar.ax.tick_params(labelsize = 60)

    ax.tick_params(axis='x', which='major', width=2)

    print( str(name) )

    # You can choose any colormap here
    #fig.colorbar(surf)
    plt.savefig( './plots/'+ str(folder) + '/' + str(name) + '.png')
    plt.close()

def two_dimensional_plotter(data,  name, name_full, folder):
    
    #print( np.array(data[0].values()) , '\n')
    #print( 'esse:', np.array(data[0].values())[:,0] )
    #print( np.shape(np.array(data[0].values())))
    #exit()
    
    mean_x = np.mean( np.array(data[0].values())[:,0] )
    std_x  = np.std( np.array(data[0].values())[:,0] )

    x_min = mean_x - 0.5*std_x
    x_max = mean_x + 0.5*std_x

    mean_y = np.mean( np.array(data[1].values())[:,0] )
    std_y  = np.std( np.array(data[1].values())[:,0] )

    y_min = mean_y - 0.5*std_y
    y_max = mean_y + 0.5*std_y

    plt.figure()
    plt.hist2d(np.array(data[0].values())[:,0], np.array(data[1].values())[:,1], bins = 10, range=[[x_min, x_max], [y_min, y_max]])
    plt.colorbar(label='Frequency')
    plt.xlabel('X-axis label')
    plt.ylabel('Y-axis label')
    plt.title('2D Histogram Title')

    plt.text(.01, .99, 'Number of entries: ', np.sum(data), transform=plt.gca().transAxes, va='top', ha='right', fontsize=16)

    # Save the plot to a file
    plt.savefig( './plots/' + str(folder) + '/' + str(name) + '.png')
    plt.close()