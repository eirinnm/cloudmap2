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
import ParseVCF
starttime=time.time()

parser = argparse.ArgumentParser(description='Process a Freebayes VCF and generate mapping plots.')
parser.add_argument('input_vcf')
parser.add_argument('--minquality',type=int, default=800, help="Minimum mapping quality")
parser.add_argument('--closeup',type=str, nargs='*', help="Optional list of chromosomes for larger plots")
args = parser.parse_args()
datafilename = args.input_vcf

chrs = [str(i) for i in range(1,26)]
chrom = ParseVCF.parse(datafilename, chrs, minqual=args.minquality)
wt_snps = np.load('wt_snps.npz')
#%%
df=[]
for c in chrs:
    thischrom = pd.DataFrame(chrom[c])
    if thischrom.shape[1]==4:
        thischrom.columns=['pos','ref','alt','mut']
    elif thischrom.shape[1]==5:
        thischrom.columns=['pos','ref','alt','mut','sib']
    else:
        raise IndexError('How many columns are there?')
    thischrom['chr']=c
    thischrom.set_index('pos', inplace=True)
    thischrom['wt']=False
    #thischrom.set_index(['chr','pos'], append=False, inplace=True)
    wt_found_here = thischrom.index.intersection(wt_snps['chr'+c])
    thischrom.loc[wt_found_here,'wt']=True
    thischrom.index=thischrom.index / 1000000 ##use megabase numbering
    thischrom.set_index('chr', append=True, inplace=True)
    thischrom=thischrom.reorder_levels(['chr','pos'])
    df.append(thischrom)
    #chrom = {c: np.array(data,dtype=float) for c, data in .iteritems()}
df=pd.concat(df)
chrom=None ## free memory
wt_snps.close()
print "Finish reading in",time.time()-starttime
#%%
## Iterate through the chromosomes and remove wt snps
## Convert chromosome arrays into a big dataframe
## Load wt SNP data

#dfs=[]
#for c in chrs:
#    df=pd.DataFrame(index=chrom[c][:,0],data=chrom[c][:,1:])
#    if df.shape[1]==1:
#        df.columns=['mut']
#    elif df.shape[1]==2:
#        df.columns=['mut','sib']
#    else:
#        raise IndexError('How many columns are there?')
#    #create a new column indicating if the SNP is present in wildtypes
#    df['wt']=False
#    wt_found_here = df.index.intersection(wt_snps['chr'+c])
#    df.loc[wt_found_here,'wt']=True
#    #remaining = df.index.difference(wt_snps['chr'+c])
#    #df=df.loc[remaining]
#    df.index=df.index / 1000000 ##use megabase numbering
#    df['chr']=c
#    df.set_index('chr', append=True, inplace=True)
#    df['homozygosity'] = abs(df.mut-0.5) * 2
#    dfs.append(df)
#dfs = pd.concat(dfs)

#%% Now calculate significant homozygosity at different bins
## do we have sibling data? Then calculate allelic distance
df['homozygosity'] = abs(df.mut-0.5) * 2
#df['linked'] = cdf[(cdf.mut>=1) & (cdf.sib>=0.25) & (cdf.sib<=0.40)]
USE_SIBS = True ##use sibling data if it's available
DRAW_STEMS = False
if 'sib' in df.columns:
    df['distance'] = np.abs(df.mut-df.sib)
else:
    USE_SIBS = False
def score_bin(group):
    if len(group)>4:
        ttest_paired = scipy.stats.ttest_rel(group.mut, group.sib)
        effectsize = np.mean(group.mut-group.sib)/group.sib.std()
        linkage = sum((group.mut==1) & (group.sib>=0.25) & (group.sib<=0.40))/len(group)
        return pd.Series(dict(ttest_paired = -math.log(ttest_paired.pvalue,10), effectsize=effectsize, linkage=linkage))
    else:
        return pd.Series(dict(ttest_paired = 0, effectsize=0))
#%%

binsize=2 #mb
min_pvalue = 20
min_distance = 0.6
fig, axes = plt.subplots(5,5,sharex=False, sharey=True,figsize=(12,8))
for c,ax in zip(chrs,axes.flat):
    cdf = df.xs(c,level=0)

    #scores.mannwhitney.plot(kind='bar',ax=ax)
    #scores.ttest.plot(kind='bar',ax=ax)
    novels = cdf[~cdf.wt]
    #novels=cdf
    ax.scatter(novels.index,novels.homozygosity,s=4,alpha=0.6,color=sns.color_palette("deep")[0],lw=0)
    smoothed = statsmodels.nonparametric.smoothers_lowess.lowess(novels.homozygosity,novels.index,frac=0.3,delta=1)
    ax.plot(smoothed[:,0],smoothed[:,1], color=sns.color_palette("deep")[2])
    if USE_SIBS:
        if DRAW_STEMS:
            scores = cdf.groupby(cdf.index//binsize).apply(score_bin)
            scores.index=scores.index*binsize
            insufficient_bins = 100-sum(scores.all(axis=1))/len(scores)*100.0
            if insufficient_bins:
                print "Chr%s: %0.1f %% of bins were too small to calculate effect size" % (c, insufficient_bins)
            scores_to_plot = scores[(scores.effectsize>0) & (scores.ttest_paired>min_pvalue)]
            if len(scores_to_plot):
                ax.stem(scores_to_plot.index,scores_to_plot.effectsize/2,'g',markerfmt='go')
        distances = cdf.distance**4
        #ax.stackplot(cdf.index, scipy.signal.savgol_filter(distances,55,2),color=sns.color_palette("deep")[1],alpha=0.6,lw=2)
        linked = cdf[(cdf.mut>=1) & (cdf.sib>=0.3) & (cdf.sib<=0.36)]
        if len(linked):
            sns.kdeplot(linked.index,bw=0.5,ax=ax,lw=1,color=sns.color_palette("deep")[1],shade=True)
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
    plt.suptitle('Cloudmap2 homozygosity plot (with linkage in green)')
else:
    plt.suptitle('Cloudmap2 homozygosity plot')
plt.savefig(datafilename[:-3]+'png')
#%%
binsize=0.5 #mb
plot_distance = True
plot_kde = True
plot_homozygosity = True
if args.closeup:
    for c in args.closeup:
        ## draw a larger plot
        fig=plt.figure(figsize=(8,6))
        ax=plt.axes()
        cdf = df.xs(c,level=0)
#        bins = np.arange(0,cdf.index[-1]+0.5,0.5)
#        scores = cdf.groupby(pd.cut(cdf.index,bins,labels=bins[:-1]+binsize/2)).apply(score_bin)
#        linkage=scores.unstack().linkage
#        linkage.index=linkage.index.astype(float)
#        linkage.plot(ax=ax, color='red')
        #scores.index=scores.index*binsize
        if plot_homozygosity:
            ax.scatter(cdf.index, cdf.homozygosity,s=4,alpha=0.6,color=sns.color_palette("deep")[0],lw=0)
            smoothed = statsmodels.nonparametric.smoothers_lowess.lowess(cdf.homozygosity,cdf.index,frac=0.3,delta=1)
            ax.plot(smoothed[:,0],smoothed[:,1], color=sns.color_palette("deep")[2])
        if plot_distance:
            distances = cdf.distance**4
            smooth_distance = scipy.signal.savgol_filter(distances,55,1)
            ax.stackplot(cdf.index, smooth_distance,color='black',alpha=0.6,lw=1)
            peak = cdf.iloc[smooth_distance.argmax()]
            ax.annotate(s=int(peak.name*1000000), xy=(peak.name,smooth_distance.max()))
        linked = cdf[(cdf.mut>=0.95) & (cdf.sib>=0.25) & (cdf.sib<=0.45)]
        if len(linked) and plot_kde:
            sns.kdeplot(linked.index,bw=0.5,ax=ax,lw=1,color=sns.color_palette("deep")[1],shade=True)
        #ax.plot(linkage.index, linkage)
        ax.text(0.05, 0.95,c,fontsize=40, color='gray', horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)
        plt.savefig(datafilename[:-3]+c+'.png')

#%% Print a table of candidates
#snptable=pd.DataFrame(columns=['CHROM','POS','ID','REF','ALT'])
snptable=linked[~linked.wt].copy()
snptable['chr']=c
snptable=snptable.reset_index()
snptable['pos']=(snptable.pos*1000000).astype(int)
snptable['id']='.'
snptable=snptable[['chr','pos','id','ref','alt']]
snptable.to_clipboard(index=False)
#for snp in linked[~linked.wt].itertuples():
#    pos=int(snp.Index*1000000)
#    print "Chromosome\t{c}\t{pos}\t{snp.ref}\t{snp.alt}\t1".format(c=c,pos=pos,snp=snp)
