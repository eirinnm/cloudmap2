# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:22:00 2017

@author: Eirinn
"""
from __future__ import division
import math
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import pandas as pd
import scipy.stats
import scipy.signal
import seaborn as sns
import statsmodels
starttime=time.time()

parser = argparse.ArgumentParser(description='Process a Freebayes VCF and generate mapping plots.')
parser.add_argument('input_vcf')
parser.add_argument('--minquality',type=int, default=800, help="Minimum mapping quality")
args = parser.parse_args()
datafilename = args.input_vcf

chrs = [str(i) for i in range(1,26)]
if datafilename.endswith('txt'):
    inputfile = np.genfromtxt(datafilename,skip_header=1)
    ## subset the columns
    data = inputfile[:,[1,5,7]] # pos, mutratio, sibratio
    # make a dictionary of each chromosome
    chrom = {i: data[data[:,0]==i] for i in range(1,26)}
elif datafilename.endswith('vcf'):
    import ParseVCF
    chrom = {c: np.array(data,dtype=float) for c, data in ParseVCF.parse(datafilename, chrs, minqual=args.minquality).iteritems()}
    
print "Finish reading in",time.time()-starttime

#%%
## columns are: chr, pos, altcount, refcount, mut_depth, mut_ratio, sib_depth, sib_ratio
#fig, axes = plt.subplots(5,5,sharex=False, sharey=True,figsize=(12,8))
##plt.figure(figsize=(12,8))
##gs = gridspec.GridSpec(1,5)
#for i,c in enumerate(chrs):
#    ax = axes.flat[i]
#    #ax = plt.subplot(gs[i],sharey=True)
#    filtered = chrom[c]#[chrom[c].min(axis=1)>0]
#    pos = filtered[:,0]
#    mutratio = filtered[:,1]
#    sibratio = filtered[:,2]
#    ## What's the differences in ratios? Take it to the 5th power
#    diffs = np.abs(mutratio-sibratio)**5
#    ax.plot(pos, diffs,lw=0.5)
#    smoothed = statsmodels.nonparametric.smoothers_lowess.lowess(diffs,pos,frac=0.3,delta=200000)
#    ax.plot(smoothed[:,0],smoothed[:,1]*2)
#    ax.text(0, 1,c,fontsize=20, alpha=0.8, color='gray',
#            horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)
#    ax.tick_params(
#        axis='x',          # changes apply to the x-axis
#        which='both',      # both major and minor ticks are affected
#        bottom='off',      # ticks along the bottom edge are off
#        top='off',         # ticks along the top edge are off
#        labelbottom='off') # labels along the bottom edge are off
#    ax.ticklabel_format(style='plain')
#    #if c == '24': break
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#plt.subplots_adjust(top=0.95)
#plt.suptitle('Difference betweeen sib and mut allele frequency')
#plt.savefig(datafilename[:-3]+'png')
##%% Plot a chr of interest at higher res
#c = '24'
#filtered = chrom[c]#[chrom[c].min(axis=1)>0]
#pos = filtered[:,0]
#mutratio = filtered[:,1]
#sibratio = filtered[:,2]
#diffs = np.abs(mutratio-sibratio)**5
#plt.plot(pos, diffs,lw=0.5)
#smoothed = statsmodels.nonparametric.smoothers_lowess.lowess(diffs,pos,frac=0.5,delta=100000)
#plt.plot(smoothed[:,0],smoothed[:,1]*2)
#plt.text(0, 1,c,fontsize=20, alpha=0.8, color='gray',
#        horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)

#%%
## Iterate through the chromosomes and remove wt snps
## Convert chromosome arrays into a big dataframe
## Load wt SNP data
wt_snps = np.load('wt_snps.npz')

dfs=[]
for c in chrs:
    df=pd.DataFrame(index=chrom[c][:,0],data=chrom[c][:,1:])
    if df.shape[1]==1:
        df.columns=['mut']
    elif df.shape[1]==2:
        df.columns=['mut','sib']
    else:
        raise IndexError('How many columns are there?')
    #create a new column indicating if the SNP is present in wildtypes
    df['wt']=False
    wt_found_here = df.index.intersection(wt_snps['chr'+c])
    df.loc[wt_found_here,'wt']=True
    #remaining = df.index.difference(wt_snps['chr'+c])
    #df=df.loc[remaining]
    df.index=df.index / 1000000 ##use megabase numbering
    df['chr']=c
    df.set_index('chr', append=True, inplace=True)
    df['homozygosity'] = abs(df.mut-0.5) * 2
    dfs.append(df)
dfs = pd.concat(dfs)
wt_snps.close()

#%% Now calculate significant homozygosity at different bins
## do we have sibling data? Then calculate allelic distance
USE_SIBS = True ##use sibling data if it's available
DRAW_STEMS = False
if 'sib' in dfs.columns:
    dfs['distance'] = np.abs(dfs.mut-dfs.sib)
else:
    USE_SIBS = False

binsize=2 #mb
min_pvalue = 20
min_distance = 0.6

def score_bin(group):
    if len(group)>20:
        ttest_paired = scipy.stats.ttest_rel(group.mut, group.sib)
        effectsize = np.mean(group.mut-group.sib)/group.sib.std()
        return pd.Series(dict(ttest_paired = -math.log(ttest_paired.pvalue,10), effectsize=effectsize))
    else:
        return pd.Series(dict(ttest_paired = 0, effectsize=0))

fig, axes = plt.subplots(5,5,sharex=False, sharey=True,figsize=(12,8))
for c,ax in zip(chrs,axes.flat):
    cdf = dfs.xs(c,level=1)
    scores = cdf.groupby(cdf.index//binsize).apply(score_bin)
    scores.index=scores.index*binsize
    #scores.mannwhitney.plot(kind='bar',ax=ax)
    #scores.ttest.plot(kind='bar',ax=ax)
    novels = cdf[~cdf.wt]
    ax.scatter(novels.index,novels.homozygosity,s=4,alpha=0.6,color=sns.color_palette("deep")[0],lw=0)
    smoothed = statsmodels.nonparametric.smoothers_lowess.lowess(novels.homozygosity,novels.index,frac=0.3,delta=1)
    ax.plot(smoothed[:,0],smoothed[:,1], color=sns.color_palette("deep")[2])
    if USE_SIBS:
        if DRAW_STEMS:
            insufficient_bins = 100-sum(scores.all(axis=1))/len(scores)*100.0
            if insufficient_bins:
                print "Chr%s: %0.1f %% of bins were too small to calculate effect size" % (c, insufficient_bins)
            scores_to_plot = scores[(scores.effectsize>0) & (scores.ttest_paired>min_pvalue)]
            if len(scores_to_plot):
                ax.stem(scores_to_plot.index,scores_to_plot.effectsize/2,'g',markerfmt='go')
        distances = cdf.distance**4
        ax.stackplot(cdf.index, scipy.signal.savgol_filter(distances,55,2)*2,color=sns.color_palette("deep")[1],alpha=0.6,lw=2)
        #distances.rolling(111).mean().plot(ax=ax)
        #ax.plot(cdf.index, distances.rolling(111).mean())
        #distant_snps = novels[novels.distance>min_distance]
        #ax.scatter(distant_snps.index,distant_snps.homozygosity,s=10, facecolors=sns.color_palette("deep")[2], lw=0,alpha=0.5)
        
    ax.set_ylim(0,1)
    ax.text(0, 1,c,fontsize=20, color='gray',
            horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        #bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        direction='out',
        pad=-10
        #labelbottom='off'
        ) # labels along the bottom edge are off
    ax.minorticks_on()
    ax.grid(b=True, which='minor',axis='x', color='lightgray', linestyle='-', alpha=1)

    #ax.ticklabel_format(style='plain')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
plt.subplots_adjust(top=0.95)
if USE_SIBS:
    plt.suptitle('Cloudmap2 homozygosity plot (with allelic distances)')
else:
    plt.suptitle('Cloudmap2 homozygosity plot')
plt.savefig(datafilename[:-3]+'png')
