# Imports and mpl definitions: 
import numpy as np
import os
from scipy import io as sio 
import scipy.stats as stats
import matplotlib.pyplot as plt

# These are the directions according to which stuff is sorted:
angles = np.array([0,  45,  90, 135, 180, 225, 270, 315])
    
# And these are the relative angles things get sorted by:
rel_angles = [0, 45, 45, 90, 90, 135, 135, 180]

def readQuest(fileName,condition):
    mat_file = sio.loadmat(fileName, squeeze_me=True, struct_as_record=False)
    history = mat_file['history'][condition]
    q = history.q 
    return q
    
def questQuantile(q,quantile):
    p=np.cumsum(q.pdf)
    idx = np.append(np.array([-1]),p)
    idx = np.where(np.diff(p)>0)[0]
    t=(q.tGuess+
       np.interp(quantile*p[-1],
                 p[idx],
                 q.x[idx]))
    return t

def questMean(q):
    t = q.tGuess + np.sum(q.pdf*q.x)/np.sum(q.pdf)
    return t

def getTh(subjectID, date, session, percentile = 0.05):
    """
    Extracts the thresholds/directions for subject/session
    
    returns: location, directions, thresholds and errors (based on
    questQuantile estimates)
    
    """
    #print(subjectID + date + str(session))
    thresh=[]
    err=[]
    data_file = ('./DATA/' +
                 'motion_th' + subjectID + date + '_' + str(session) + '.mat')
    x=sio.loadmat(data_file, squeeze_me=True, struct_as_record=False)
    a=x['results']
    directions= a[0].stimParams.dotDirections
    loc = a[0].stimParams.locat
    for thisa in a:
        thresh.append(10**questQuantile(thisa.scanHistory.q,0.5)*40)
        #If you want to use questMean instead of questQuantile:
        #thresh.append(10**questMean(thisa.scanHistory[0][0].q[0][0])*40)
        err.append(10**questQuantile(thisa.scanHistory.q,
        1-percentile/2.0)*40-10**questQuantile(thisa.scanHistory.q,
                                               percentile/2.0)*40)
        
    return (loc,
            np.array(directions[0:len(thresh)]),
            np.array(thresh), np.array(err)/2.0)

def absDirDiff(dir1,dir2):
    """ Calculates the minimal difference in angle between two angles given in
    degrees around a circle (e.g. absDirDiff(360,0) = 0, absDirDiff(270,45) =
    135, etc.)"""
    ans = np.zeros(dir1.shape)
    for a in range(len(dir1)):
        ans[a] = min([abs(dir1[a]-dir2[a]),
                      abs((360+dir1[a])-dir2[a]),
                      abs((360+dir2[a])-dir1[a])])
    
    return ans
    
def get_sub_data(sub_id, date, blocks):

    loc = []
    th = []
    # These are the blocks/files from that date:
    for block in blocks:
        l, d, t, e = getTh(sub_id, date, block) 
        # Reorder according to direction (absolute, not relative):
        idx = np.argsort(d)
        t = t[idx]
        e = e[idx]
        # nan out the large errors: 
        t[np.where(e>40)] = np.nan
        loc.append(l)
        th.append(t)

    loc = np.array(loc)
    th = np.array(th)
 
    return loc, th


def p_learning(pre,post):
    return 100*(1-(np.asarray(post)/np.asarray(pre)))

def th_improvement(pre,post):
    return np.asarray(pre) - np.asarray(post)


def plot_results(trained_p, trained_d, untrained_ave):
    trained_p = np.array(trained_p).squeeze()
    trained_d = np.array(trained_d).squeeze()
    untrained_ave = np.array(untrained_ave)
    p = trained_p-untrained_ave
    d = trained_d-untrained_ave
    
    print("Trained donepezil vs. trained placebo:")
    print stats.wilcoxon(trained_p - trained_d)
    print("Trained donepezil vs. untrained :")
    print stats.wilcoxon(trained_d - untrained_ave)
    print("Trained placebo vs. untrained :")
    print stats.wilcoxon(trained_p - untrained_ave)
    print("Differences from baseline:")
    print stats.wilcoxon(p, d)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.bar([1],
           [np.mean(trained_d)],
           yerr = [stats.sem(trained_d)],
           color='r', ecolor='r')

    ax.bar([2],
           [np.mean(trained_p)],
           yerr = [stats.sem(trained_p)],
           color='b', ecolor='b')

    ax.bar([3],
           [np.mean(untrained_ave)],
           yerr = [stats.sem(untrained_ave)],
           color='g', ecolor='g')

    plt.show()

    return fig


def polar_plots(th_arr, subplot=111):
    """

    """
    to_plot = np.mean(th_arr,0)
    to_plot = np.vstack([to_plot[0], to_plot[2], to_plot[1]])
    err = stats.sem(th_arr, 0)
    err = np.vstack([err[0], err[2], err[1]])
    fig = plt.figure()
    ax = fig.add_subplot(subplot, projection='polar')
    colors = [[0.09215686,  0.53921569,  0.12941176],
              [ 0.57843137,  0.76666667,  0.35686275],
              [ 0.86862745,  0.88823529,  0.5254902 ]]

    #colors = [[0.13725490196078433, 0.51764705882352946, 0.2627450980392157],
    #          [0.47058823529411764, 0.77647058823529413, 0.47450980392156861],
    #          [0.76078431372549016, 0.90196078431372551, 0.59999999999999998]]
    
    for ii, (p,e) in enumerate(zip(to_plot, err)):
        print p
        print e
        ax.plot(np.deg2rad(angles+45), p, 'o-', color=colors[ii])
        ax.errorbar(np.deg2rad(angles+45), p, yerr=e, color=colors[ii],
                    capsize=0)

    ax.set_xticklabels([str(x) for x in np.roll(angles,1)])
    ax.set_yticks([6,10,14,18])
    # This toggles the x grid off:
    ax.grid('off')
    ax.grid('on', axis='y', linestyle='-', color=[0.3,0.3,0.3])
    ax.set_frame_on(False)
    return fig
