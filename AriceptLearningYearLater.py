

import numpy as np
import os
from scipy import io as sio #needed in order to read in .mat files with
                            #thresholds
import scipy.stats as stats
import matplotlib.pyplot as plt
from rpy2.robjects import r as rstats

#Set svg fonts to be editable in AI: 
from matplotlib import rc
rc('svg', embed_char_paths='none')

# And set the font to be bolder and larger: 
font = {'family' : 'Helvetica',
        'weight' : 'regular',
        'size'   : 12}
    
rc('font', **font)  # pass in the font dict as kwargs

rc('lines', linewidth=2)

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

def getTh(subjectID,date,session, percentile = 0.05):
    """
    Extracts the thresholds/directions for subject/session
    
    returns: location, directions, thresholds and errors (based on questQuantile estimates) 
    
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
    """ Calculates the minimal difference in angle between two angles given in degrees 
    around a circle (e.g. absDirDiff(360,0) = 0, absDirDiff(270,45) = 135, etc.) """
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

if __name__ == '__main__':

    f = file('long_term_donepezil.csv','w')
    f.write('subject,trained_donepezil,trained_placebo,untrained_donepezil,untrained_placebo\n')
    f_R = file('ltd4R.csv', 'w')
    f_R.write('subject,training,p_learning,th_learning,first_th,second_th,year_th,group\n')
    
    placebo_first=['KAA', 'CCS', 'EJS', 'JHS', 'LJL'] #
    aricept_first=['APM', 'CNC', 'GMC'] #
   
    subjects = np.array(placebo_first + aricept_first)
        
    subjDict = {
        'KAA':[[[1,135],['111307','111907'],[[2,3,5],[1,2,3]]],
               [[2,315],['120307','120907'],[[1,2,3,4],[1,2,3,4]]],
               ['020509',[1,2,3,4]]],
        'CCS':[[[1,45],['022608','030408'],[[2,3,4,5],[1,2,4,5]]],
               [[2,225],['031808','032508'],[[1,2,3,4],[1,2,4,5]]],
               ['020609',[1,2,3,4]]],
        'EJS':[[[1,225],['022908','030708'],[[3,4,5],[1,2,3,4]]],
               [[2,45],['032108','032808'],[[1,2,3,4],[1,2,3,4]]],
               ['020509',[2,3,4,5]]],
        'JHS':[[[2,135],['121007','121607'],[[2,3,4,5],[1,2,3,4]]],
               [[1,315],['041408','042008'],[[2,3,4,5],[1,2,3,4]]],
               ['021009',[1,2,3,4]]],
        'LJL':[[[2,315],['080408','081208'],[[1,2,3,4],[1,2,3,4]]],
               [[1,135],['100608','101408'],[[1,2,3,4],[1,2,3,4]]],
               ['031609',[1,2,3,4]]],
        'CNC':[[[1,225],['030408','031008'],[[2,3,4,5],[1,2,3,4]]],
               [[2,45],['033108','040608'],[[4,5,6],[1,2,3,4]]],
               ['031109',[2,3,5]]], 
        'GMC':[[[1,315],['040708','041308'],[[3,4,5,6],[1,2,3,4]]],
               [[2,135],['042808','050408'],[[1,2,3,4],[1,2,3,4]]],
               ['022009',[1,2,3,4]]],   
        'APM':[[[1,315],['100608','101408'],[[2,3,4,5],[1,2,3,4]]],
               [[2,135],['103008','110708'],[[1,2,3,4],[1,2,3,4]]],
               ['031909',[1,2,3,4]]]}

    # These are the directions according to which stuff is sorted below:
    angles = np.array([0,  45,  90, 135, 180, 225, 270, 315])
    
    # And these are the relative angles things get sorted by:
    rel_angles = [0, 45, 45, 90, 90, 135, 135, 180]

    p_l = {}
    th = {}
    raw_th = {}
    
    trained_p = []
    trained_d = []
    untrained_p = []
    untrained_d = []
    untrained_ave = []

    trained_p_th = []
    trained_d_th = []
    untrained_p_th = []
    untrained_d_th = []
    untrained_ave_th = []

    trained_p_th_f = []
    trained_d_th_f = []
    untrained_p_th_f = []
    untrained_d_th_f = []
    untrained_ave_th_f = []
    
    trained_p_th_y = []
    trained_d_th_y = []
    untrained_p_th_y = []
    untrained_d_th_y = []
    untrained_ave_th_y = []

    trained_p_th_p = []
    trained_d_th_p = []
    untrained_p_th_p = []
    untrained_d_th_p = []
    untrained_ave_th_p = []

    for subject in subjects:
        if subject in placebo_first:
            placebo_cond = subjDict[subject][0][0]
            aricept_cond = subjDict[subject][1][0]
            group = 'p'
        else:
            aricept_cond = subjDict[subject][0][0]
            placebo_cond = subjDict[subject][1][0]
            group = 'd'

        raw_th[subject] = {}
        th[subject] = {}
        p_l[subject] = {}

        l_f, t_f = get_sub_data(subject,
                                   subjDict[subject][0][1][0],
                                   subjDict[subject][0][2][0])

	l_p, t_p = get_sub_data(subject,
				subjDict[subject][1][1][0],
				subjDict[subject][1][2][0])
	
        l_y, t_y = get_sub_data(subject,
                                   subjDict[subject][2][0],
                                   subjDict[subject][2][1])

        for this_loc in [1,2]:
            this_t_f = stats.nanmean(t_f[np.where(l_f==this_loc)],0)
            this_t_y = stats.nanmean(t_y[np.where(l_y==this_loc)],0)
	    this_t_p = stats.nanmean(t_p[np.where(l_p==this_loc)],0)
	    
            raw_th[subject][this_loc] = [this_t_f, this_t_y, this_t_p]
            th[subject][this_loc] = th_improvement(this_t_f, this_t_y)
            p_l[subject][this_loc] = p_learning(this_t_f, this_t_y)
            
        trained_p_dir = np.where(angles==placebo_cond[1])
        trained_d_dir = np.where(angles==aricept_cond[1])
        untrained_dir = np.intersect1d(np.where(angles!=aricept_cond[1])[0],
                                       np.where(angles!=placebo_cond[1])[0])
                                   
        trained_p_loc = placebo_cond[0]
        trained_d_loc = aricept_cond[0]
        
        trained_p.append(p_l[subject][trained_p_loc][trained_p_dir])
        trained_d.append(p_l[subject][trained_d_loc][trained_d_dir])
        untrained_d.append(np.mean(p_l[subject][trained_p_loc][untrained_dir]))
        untrained_p.append(np.mean(p_l[subject][trained_d_loc][untrained_dir]))
        untrained_ave.append(np.mean([untrained_d[-1],untrained_p[-1]]))

        trained_p_th.append(th[subject][trained_p_loc][trained_p_dir])
        trained_d_th.append(th[subject][trained_d_loc][trained_d_dir])
        untrained_d_th.append(np.mean(th[subject][trained_p_loc][untrained_dir]))
        untrained_p_th.append(np.mean(th[subject][trained_d_loc][untrained_dir]))
        untrained_ave_th.append(np.mean([untrained_d_th[-1],untrained_p_th[-1]]))

        trained_p_th_f.append(raw_th[subject][trained_p_loc][0][trained_p_dir])
        trained_d_th_f.append(raw_th[subject][trained_d_loc][0][trained_d_dir])
        untrained_d_th_f.append(np.mean(raw_th[subject][trained_p_loc][0][untrained_dir]))
        untrained_p_th_f.append(np.mean(raw_th[subject][trained_d_loc][0][untrained_dir]))
        untrained_ave_th_f.append(np.mean([untrained_d_th_f[-1],
                                           untrained_p_th_f[-1]]))

        trained_p_th_y.append(raw_th[subject][trained_p_loc][1][trained_p_dir])
        trained_d_th_y.append(raw_th[subject][trained_d_loc][1][trained_d_dir])
        untrained_d_th_y.append(np.mean(raw_th[subject][trained_p_loc][1][untrained_dir]))
        untrained_p_th_y.append(np.mean(raw_th[subject][trained_d_loc][1][untrained_dir]))
        untrained_ave_th_y.append(np.mean([untrained_d_th_y[-1],
                                           untrained_p_th_y[-1]]))

	trained_p_th_p.append(raw_th[subject][trained_p_loc][2][trained_p_dir])
        trained_d_th_p.append(raw_th[subject][trained_d_loc][2][trained_d_dir])
        untrained_d_th_p.append(np.mean(raw_th[subject][trained_p_loc][2][untrained_dir]))
        untrained_p_th_p.append(np.mean(raw_th[subject][trained_d_loc][2][untrained_dir]))
        untrained_ave_th_p.append(np.mean([untrained_d_th_p[-1],
                                           untrained_p_th_p[-1]]))
        
        
        f.write('%s,%s,%s,%s,%s\n'%(subject,
                                    trained_d[-1][0],
                                    trained_p[-1][0],
                                    untrained_d[-1],
                                    untrained_p[-1]))
        
        f_R.write('%s,%s,%s,%s,%s,%s,%s,%s\n'%(subject,'d',
                                      trained_d[-1][0],
                                      trained_d_th[-1][0],
                                      trained_d_th_f[-1][0],
                                      trained_d_th_p[-1][0],
                                      trained_d_th_y[-1][0],
                                      group))

        f_R.write('%s,%s,%s,%s,%s,%s,%s,%s\n'%(subject,'p',
                                      trained_p[-1][0],
                                      trained_p_th[-1][0],
                                      trained_p_th_f[-1][0],
                                      trained_p_th_p[-1][0],
                                      trained_p_th_y[-1][0],
                                      group))

        f_R.write('%s,%s,%s,%s,%s,%s,%s,%s\n'%(subject,'u',
                                      untrained_ave[-1],
                                      untrained_ave_th[-1],
                                      untrained_ave_th_f[-1],
                                      untrained_ave_th_p[-1],
                                      untrained_ave_th_y[-1],
                                      group))
        
    f.close()
    f_R.close()
    rstats('''
    library(ez)

    # Read the data you just made above:

    data = read.table("ltd4R.csv",sep=',',header = TRUE)

    aov_p = ezANOVA(data,
                 wid=.(subject),
                 dv=.(p_learning),
                 within=.(training),
                 between=.(group)
                 )

    aov_th = ezANOVA(data,
                 wid=.(subject),
                 dv=.(th_learning),
                 within=.(training),
                 between=.(group)
                 )
           ''')

    print(rstats.aov_p)
    print(rstats.aov_th)

    rstats('''

    #Read the data you just made above:
    data = read.table("ltd4R.csv",sep=',',header = TRUE)

    aov_th = aov(th_learning ~ (training*group) + Error(subject/training + group), 
                     data=data)

    aov_p = aov(p_learning ~ (training*group) + Error(subject/training + group), 
                       data=data)
    
    aov_f = aov(first_th ~ (training*group) + Error(subject/training + group), 
                       data=data)

    aov_s = aov(second_th ~ (training*group) + Error(subject/training + group), 
                       data=data)

    aov_y = aov(year_th ~ (training*group) + Error(subject/training + group), 
                       data=data)
    
    ''')

    print("***** ANOVA : THRESHOLDS:*******")
    print(rstats.summary(rstats.aov_th))

    print("***** ANOVA : % LEARNING:*******")
    print(rstats.summary(rstats.aov_p))

    print("***** ANOVA : First threshold:*******")
    print(rstats.summary(rstats.aov_f))

    print("***** ANOVA : Second threshold:*******")
    print(rstats.summary(rstats.aov_s))

    print("***** ANOVA : Third threshold:*******")
    print(rstats.summary(rstats.aov_y))

    fig = plot_results(trained_p, trained_d, untrained_ave)
    ax = fig.axes[0]
    ax.set_xlim([0.8, 4])
    ax.set_xticks([1.4, 2.4, 3.4])
    ax.set_xticklabels(['donepezil','placebo','untrained'])
    ax.set_ylabel('Percent reduction in threshold')
    fig.savefig('fig2_p.svg')
    
    fig = plot_results(trained_p_th, trained_d_th, untrained_ave_th)
    ax = fig.axes[0]
    ax.set_xlim([0.8, 4])
    ax.set_xticks([1.4, 2.4, 3.4])
    ax.set_xticklabels(['donepezil','placebo','untrained'])
    ax.set_ylabel('Percent reduction in threshold')
    fig.savefig('fig2_th.svg')

    fig = plot_results(trained_p_th_y, trained_d_th_y, untrained_ave_th_y)
    ax = fig.axes[0]
    ax.set_xlim([0.8, 4])
    ax.set_xticks([1.4, 2.4, 3.4])
    ax.set_xticklabels(['donepezil','placebo','untrained'])
    ax.set_ylabel('Percent reduction in threshold')
    fig.savefig('fig2_th.svg')


    fig = plot_results(trained_p_th_f, trained_d_th_f, untrained_ave_th_f)
    ax = fig.axes[0]
    ax.set_xlim([0.8, 4])
    ax.set_xticks([1.4, 2.4, 3.4])
    ax.set_xticklabels(['donepezil','placebo','untrained'])
    ax.set_ylabel('Percent reduction in threshold')
    fig.savefig('fig2_th.svg')


colors = ['b', 'r', 'g']
fig, ax = plt.subplots(1)
for idx, this in enumerate(
        zip([trained_p_th_f, trained_d_th_f, untrained_p_th_f],
            [trained_p_th_p, trained_d_th_p, untrained_p_th_p],
            [trained_p_th_y, trained_d_th_y, untrained_p_th_y])):

    x = [1,2,3]
    y = np.array([np.mean(this[0]),
		  np.mean(this[1]),
	          np.mean(this[2])]).squeeze()
    
    yerr = np.array([stats.sem(this[0]),
		     stats.sem(this[1]),
	             stats.sem(this[2])]).squeeze()
    ax.errorbar(x, y, yerr=yerr, color=colors[idx], linestyle='none',
                marker='o')
ax.set_xlim([0,4])
ax.set_ylabel('Percent learning')
ax.set_xlabel('Time point')
ax.set_xticks([1,2,3])
ax.set_xticklabels(['Pre','Post', 'long-term'])
fig.savefig('fig3.svg')


time_since_training =[15,11,11,14,5,5,11,9]
trained_d = [45.77, 61.1, 24.1, 43.8, 47.03, 66.89, 41.21, 46.97]
trained_p = [42.81, 49.89, 34.98, 31.67, 26.55, 59.59, -6.5, 34.37]
drug_effect = [2.96, 11.22, -10.87, 12.14, 20.48, 7.29, 47.7, 12.6]


fig, ax = plt.subplots(1)
ax.plot(time_since_training, trained_d, 'ro', markersize=10)
ax.plot(time_since_training, trained_p, 'bo', markersize=10)

ax.plot([4,16],[0,0],'k')
ax.set_xlabel('Time since training (months)')
ax.set_ylabel('Percent learning')
fig.savefig('fig4a.svg')


fig, ax = plt.subplots(1)
ax.plot([4,16],[0,0],'k')
ax.set_xlabel('Time since training (months)')
ax.set_ylabel('Percent learning (drug) - Percent learning (placebo)')
ax.plot(time_since_training, drug_effect, 'ko', markersize=10)
fig.savefig('fig4b.svg')


