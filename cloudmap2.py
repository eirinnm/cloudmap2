# -*- coding: utf-8 -*-
## Cloudmap2: Read a VCF, extract allele balances at each SNP, and use LOWESS interpolation to find regions of homozygosity.
## The input VCF should have two samples, corresponding to sibling and mutant.
import math
import os.path
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import scipy.signal
import scipy.interpolate
import seaborn as sns
sns.set_context("talk")
sns.set_style("darkgrid")
import ParseVCF
starttime = time.time()

parser = argparse.ArgumentParser(description='Process a VCF and generate mapping plots.')
parser.add_argument('input_vcf')
parser.add_argument('--minquality', type=int, default=800, help="Minimum mapping quality")
parser.add_argument('--closeup', type=str, nargs='*', help="Optional list of chromosomes for larger plots")
parser.add_argument('--deseq', type=str, help="Optional DESeq table of differentially-expressed genes for marking")
parser.add_argument('--version', choices=('10','11'), default='10', help="Genome version for WT snp removal")
parser.add_argument('--flipsamples', action='store_true', help='Swap the sibling and mutant samples')
parser.add_argument('--ignore100', action='store_true', help='Ignore homozygous SNPs (for samples which have too many)')
parser.add_argument('--dominant', action='store_true', help='Search for dominant mutations')
args = parser.parse_args()
datafilename = args.input_vcf
samplename = os.path.basename(datafilename).replace('.vcf', '')
chrs = [str(i) for i in range(1, 26)]
chrom = ParseVCF.parse(datafilename, chrs, minqual=100)  # load in all SNPs with quality above 100
if args.version == '10':
    print('Loading WT SNPs from Zv10 database')
    wt_snps = np.load('wt_snps_zv10.npz')
elif args.version == '11':
    print('Loading WT SNPs from Zv11 database')
    wt_snps = np.load('wt_snps_zv11.npz')
# %%

df = []
for c in chrs:
    thischrom = pd.DataFrame(chrom[c])
    if thischrom.shape[1] == 6:
        thischrom.columns = ['pos', 'ref', 'alt', 'qual', 'mut', 'mut_depth']
    elif thischrom.shape[1] == 8:
        thischrom.columns = ['pos', 'ref', 'alt', 'qual', 'mut', 'sib', 'mut_depth', 'sib_depth']
        if args.flipsamples:
            thischrom.columns = ['pos', 'ref', 'alt', 'qual', 'sib', 'mut', 'sib_depth', 'mut_depth']
    else:
        raise IndexError('How many columns are there?')
    thischrom['chr'] = c
    thischrom.set_index('pos', inplace=True)
    thischrom['wt'] = False
    #thischrom.set_index(['chr','pos'], append=False, inplace=True)
    wt_found_here = thischrom.index.intersection(wt_snps['chr'+c])
    thischrom.loc[wt_found_here, 'wt'] = True
    thischrom.index = thischrom.index / 1000000  # use megabase numbering
    thischrom.set_index('chr', append=True, inplace=True)
    thischrom = thischrom.reorder_levels(['chr', 'pos'])
    df.append(thischrom)
    # chrom = {c: np.array(data,dtype=float) for c, data in .iteritems()}
df = pd.concat(df)
chrom = None  # free memory
wt_snps.close()
print("Finish reading in", time.time()-starttime)
median_quality = df.qual.median()
print("SNP quality: median=%0.1f, mean=%0.1f" % (median_quality, df.qual.mean()))
# %% Now calculate significant homozygosity at different bins
# do we have sibling data? Then calculate allelic distance
df['mut_homozygosity'] = abs(df.mut-0.5) * 2
#df['linked'] = cdf[(cdf.mut>=1) & (cdf.sib>=0.25) & (cdf.sib<=0.40)]
USE_SIBS = True  # use sibling data if it's available

if 'sib' in df.columns:
    df['sib_homozygosity'] = abs(df.sib-0.5) * 2
    df['distance'] = np.abs(df.mut-df.sib)
    df['distance_homozygosity'] = np.abs(df.mut_homozygosity-df.sib_homozygosity)
    # EXPERIMENTAL: remove SNPs that are homozygous in both
    df = df[(df.sib < 1) | (df.mut < 1)]
else:
    USE_SIBS = False


def score_bin(group):
    if len(group) > 4:
        ttest_paired = scipy.stats.ttest_rel(group.mut, group.sib)
        effectsize = np.mean(group.mut-group.sib)/group.sib.std()
        linkage = sum((group.mut == 1) & (group.sib >= 0.25) & (group.sib <= 0.40))/len(group)
        return pd.Series(dict(ttest_paired=-math.log(ttest_paired.pvalue, 10), effectsize=effectsize, linkage=linkage))
    else:
        return pd.Series(dict(ttest_paired=0, effectsize=0))


# %% Load the list of differentially-expressed genes
genedf = None
if args.deseq:
    genefilepath = os.path.join(os.path.dirname(datafilename), args.deseq)
    genedf = pd.read_csv(genefilepath, sep='\t', index_col=0)
    genedf.chr = genedf.chr.astype(str)
#    significant_genes = genedf[genedf.padj < 0.1]
    genedf['significant'] = genedf.padj < 0.1

# %% Calculate LOESS smoothing
from statsmodels.nonparametric.smoothers_lowess import lowess
# use SNPs that aren't in the wildtype database and that meet the minimum quality

print("%s total SNPs. Removing %s (%0.1f%%) that are known WT snps." % (len(df), df.wt.sum(), 100*df.wt.sum()/len(df)))
print("Selecting SNPs with QUAL greater than", args.minquality)
if args.dominant:
   print("Also excluding homozygous SNPs in mutant sample (looking for dominant mutations)")
   goodsnps = df[~df.wt & (df.qual > args.minquality) & ((df.mut_homozygosity < 0.20) | (df.sib_homozygosity > 0.70))].reset_index()
else:
    goodsnps = df[~df.wt & (df.qual > args.minquality)].reset_index()
if len(goodsnps) == 0:
    print("No SNPs meet the minimum quality threshold. Changing the threshold to the median quality (%s)" % median_quality)
    goodsnps = df[~df.wt & (df.qual > median_quality)].reset_index()
print("Using %s SNPs (%0.1f%% of input) for LOESS smoothing" % (len(goodsnps), 100*len(goodsnps)/len(df)))
# %%


def smoother(chrom, column):
    return lowess(chrom[column].values, chrom.pos.values, frac=0.3, delta=0.1)[:, 1]

    
print("Smoothing... ")
if not args.dominant:
    for c in chrs:
        print(c, end=' ')
        goodsnps.loc[goodsnps.chr == c, 'mut_smoothed'] = smoother(goodsnps[goodsnps.chr == c], 'mut')
        if USE_SIBS:
            goodsnps.loc[goodsnps.chr == c, 'sib_smoothed'] = smoother(goodsnps[goodsnps.chr == c], 'sib')
            goodsnps.loc[goodsnps.chr == c, 'distance_smoothed'] = smoother(goodsnps[goodsnps.chr == c], 'distance')
else:
    for c in chrs:
        print(c, end=' ')
        goodsnps.loc[goodsnps.chr == c, 'mut_smoothed'] = smoother(goodsnps[goodsnps.chr == c], 'mut_homozygosity')
        if USE_SIBS:
            goodsnps.loc[goodsnps.chr == c, 'sib_smoothed'] = smoother(goodsnps[goodsnps.chr == c], 'sib_homozygosity')
            goodsnps.loc[goodsnps.chr == c, 'distance_smoothed'] = smoother(goodsnps[goodsnps.chr == c], 'distance_homozygosity')

# %% Logistic regression: does being close to the linked region impact the likelihood of a gene being differentially expressed?
# first, find the linkage distance of each gene by interpolating the gene start position in amongst the SNP linkage data.
if genedf is not None:
   import scipy.interpolate
   import statsmodels.api as sm
   for c in chrs:
       x = goodsnps[goodsnps.chr == c].pos*1000000
       y = goodsnps[goodsnps.chr == c].distance_smoothed
       interfunc = scipy.interpolate.interp1d(x, y, fill_value=0, bounds_error=False)
       genedf.loc[genedf.chr == c, 'linkage'] = interfunc(genedf[genedf.chr == c].start)
#    genedf['significant'] = genedf.padj < 0.1
   ## drop genes without linkage info
   genedf = genedf[~pd.isna(genedf.linkage)]
   logit = sm.formula.glm('significant ~ linkage', data=genedf, family=sm.families.Binomial())
   # workaround for a bug in statsmodels that flips the coefficient of booleans (issue #2181)
   logit.endog = ~logit.endog.astype(bool)
   logfit = logit.fit()
   print(logfit.summary())
   test_points = np.array([genedf.linkage.median(),  genedf.linkage.max()])
   yvals = logfit.params.Intercept + logfit.params.linkage * test_points
   probs = np.exp(yvals)
   stattext = f'''logit regression: coeff = {logfit.params.linkage:3f}, p = {logfit.pvalues.linkage:.5f}
   prob(DE) at median linkage = {probs[0]:0.3f}, at max linkage = {probs[1]:0.3f}'''
   print(stattext)
   # predict some values for a plot
   predicted = pd.DataFrame({'linkage': np.arange(0, genedf.linkage.max(), 0.01), samplename: 0})
   predicted[samplename] = logfit.predict(predicted)
   predicted = predicted.set_index('linkage', drop=True)
   predicted.to_csv(datafilename[:-3]+'logregression.csv')
# %% Draw plots
fig, axes = plt.subplots(5, 5, sharex=False, sharey=True, figsize=(16, 10))
for c, ax in zip(chrs, axes.flat):
    cdf = goodsnps[goodsnps.chr == c]
    if args.dominant:
        ax.scatter(cdf.pos, cdf.mut_homozygosity, s=3, alpha=0.6, color=sns.color_palette("deep")[0], lw=0, label="SNPs")
    else:
        ax.scatter(cdf.pos, cdf.mut, s=3, alpha=0.6, color=sns.color_palette("deep")[0], lw=0, label="SNPs")
    ax.plot(cdf.pos, cdf.mut_smoothed, color=sns.color_palette("bright")[0], lw=3, label="Mut")
    if USE_SIBS:
        ax.plot(cdf.pos, cdf.sib_smoothed, color=sns.color_palette("bright")[1], lw=3, label="Sib")
        ax.plot(cdf.pos, cdf.distance_smoothed, color=sns.color_palette("bright")[2], lw=3, label="Difference")
    if c=='25':
        ax.legend(fontsize="xx-small",ncol=4, bbox_to_anchor=(1,-0.1), frameon=False)
    ax.set_ylim(0, 1)
    if genedf is not None:  # draw a line indicating the potential genes
#        for gene in significant_genes[significant_genes.chr.astype(str) == c].itertuples():
        for gene in genedf[(genedf.chr.astype(str) == c) & genedf.significant].itertuples():
            color = sns.color_palette("bright")[2] if gene.log2FoldChange < 0 else sns.color_palette("bright")[0]
            ax.axvline(gene.start/1000000.0, color=color, lw=1)
    ax.text(0, 1, c, fontsize=30, color='gray', horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)
    ax.tick_params(axis='x', which='both', top=False, direction='out', pad=-10, labelright=True,)
    ax.minorticks_on()
    ax.grid(b=True, which='minor', axis='x', color='lightgray', linestyle='-', alpha=1)
#plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
plt.subplots_adjust(top=0.95)
plt.suptitle(samplename+' Cloudmap2 SNP plot')
fig.add_subplot(111, frameon=False)
#plt.legend()
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Position (MB)")
if args.dominant:
    plt.ylabel("Homozygosity")
else:
    plt.ylabel("Allele ratio")
if genedf is not None:
#    plt.figtext(1, 0, stattext, horizontalalignment="right", verticalalignment="top", style='italic', size='x-small')
    plt.savefig(datafilename[:-3]+'genes.png', bbox_inches="tight")
else:
    plt.savefig(datafilename[:-3]+'png', bbox_inches="tight")

# %% Draw plots of a single selected chromosome (optional)
if args.closeup:
    for c in args.closeup:
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes()
        c = str(c)
        cdf = goodsnps[goodsnps.chr == c]
        ax.scatter(cdf.pos, cdf.mut, s=3, alpha=0.6, color=sns.color_palette("deep")[0], lw=0)
        ax.plot(cdf.pos, cdf.mut_smoothed, color=sns.color_palette("bright")[0], lw=3)
        if USE_SIBS:
            ax.plot(cdf.pos, cdf.sib_smoothed, color=sns.color_palette("bright")[1], lw=3)
            ax.plot(cdf.pos, cdf.distance_smoothed, color=sns.color_palette("bright")[2], lw=3)
            mapping_peak = cdf.loc[cdf.distance_smoothed.idxmax()]
            ax.annotate(s=int(mapping_peak.pos*1000000), xy=(mapping_peak.pos, mapping_peak.distance_smoothed))
        ax.legend().set_visible(False)
        ax.set_ylim(0, 1)
        if genedf is not None:  # draw a line indicating the potential genes
            for gene in genedf[(genedf.chr.astype(str) == c) & genedf.significant].itertuples():
#            for gene in significant_genes[significant_genes.chr.astype(str) == c].itertuples():
                color = sns.color_palette("bright")[2] if gene.log2FoldChange < 0 else sns.color_palette("bright")[0]
                ax.axvline(gene.start/1000000.0, color=color, lw=1)
        ax.text(0, 1, c, fontsize=30, color='gray', horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='x', which='both', top=False, direction='out', pad=-10, labelright='on',)
        ax.minorticks_on()
        ax.grid(b=True, which='minor', axis='x', color='lightgray', linestyle='-', alpha=1)
        plt.ylabel('Homozygosity')
        plt.xlabel("Position (MB)")
        plt.tight_layout()
        if genedf is not None:
            plt.savefig(datafilename[:-3]+c+'.genes.png')
        else:
            plt.savefig(datafilename[:-3]+c+'.png')

#%%
# Save a list of genes with linkage data, for logistic regression later
if genedf is not None:
    for c in chrs:
        x = goodsnps[goodsnps.chr == c].pos*1000000
        y = goodsnps[goodsnps.chr == c].distance_smoothed
        interfunc = scipy.interpolate.interp1d(x, y, fill_value=0, bounds_error=False)
        genedf.loc[genedf.chr == c, 'linkage'] = interfunc(genedf[genedf.chr == c].start)            
    #calculate proximity to the mapping peak
    genedf['proximity'] = 0
    proximity_thischr = -abs(genedf[genedf.chr == mapping_peak.chr].start - mapping_peak.pos*1000000)
    proximity_thischr = proximity_thischr / (proximity_thischr.max() - proximity_thischr.min()) + 1
    genedf.loc[genedf.chr == mapping_peak.chr, 'proximity'] = proximity_thischr
    genedf.to_csv(datafilename[:-3]+'genedf.csv')
# %%
# binsize=0.5 #mb
# mapping_percentile = 0.99 #percentage of peak height that we call as the linked region
#plot_distance = False
#plot_kde = True
#plot_homozygosity = True
# if args.closeup:
#    print("Closeup plots:",args.closeup)
#    for c in args.closeup:
#        ## draw a larger plot
#        fig=plt.figure(figsize=(8,6))
#        ax=plt.axes()
#        cdf = df.xs(c,level=0)
##        bins = np.arange(0,cdf.index[-1]+0.5,0.5)
##        scores = cdf.groupby(pd.cut(cdf.index,bins,labels=bins[:-1]+binsize/2)).apply(score_bin)
# linkage=scores.unstack().linkage
# linkage.index=linkage.index.astype(float)
##        linkage.plot(ax=ax, color='red')
#        #scores.index=scores.index*binsize
#        thesesnps = cdf[~cdf.wt & (cdf.qual>args.minquality)] #use unique SNPs only
# if USE_SIBS:
# thesesnps = thesesnps[thesesnps.sib<0.99] #filter out anything homozygous in sibs
#        ax.scatter(thesesnps.index, thesesnps.mut,s=6,alpha=0.6,color=sns.color_palette("deep")[0],lw=0)
#        smoothed = statsmodels.nonparametric.smoothers_lowess.lowess(thesesnps.mut,thesesnps.index,frac=0.3,delta=0.1)
#        lowess_argmax = smoothed[:,1].argmax()
#        lowess_peak = smoothed[lowess_argmax]
#        ## find the first value within the mapping_percentile of this height
#        region_left = smoothed[np.argmax(smoothed[:,1]>lowess_peak[1]*mapping_percentile)]
#        region_right = smoothed[lowess_argmax+np.argmax(smoothed[lowess_argmax:,1]<lowess_peak[1]*mapping_percentile)]
#        ax.plot(smoothed[:,0],smoothed[:,1], color=sns.color_palette("bright")[0])
#        ax.annotate(s=int(lowess_peak[0]*1000000), xy=lowess_peak)
#            #ax.axvspan(region_left[0],region_right[0],alpha=0.3)
#        if USE_SIBS:
#            ax.scatter(thesesnps.index, thesesnps.sib,s=6,alpha=0.6,color=sns.color_palette("deep")[1],lw=0)
#            smoothed_sib = statsmodels.nonparametric.smoothers_lowess.lowess(thesesnps.sib,thesesnps.index,frac=0.3,delta=0.1)
#            ax.plot(smoothed_sib[:,0],smoothed_sib[:,1], color=sns.color_palette("bright")[1])
#            smoothed_distance = statsmodels.nonparametric.smoothers_lowess.lowess(thesesnps.distance,thesesnps.index,frac=0.3,delta=0.1)
#            ax.plot(smoothed_distance[:,0],smoothed_distance[:,1], color='black',lw=1)
# if plot_distance:
##                distances = cdf.distance**4
##                smooth_distance = scipy.signal.savgol_filter(distances,55,1)
##                ax.stackplot(cdf.index, smooth_distance,color='black',alpha=0.6,lw=1)
##                peak = cdf.iloc[smooth_distance.argmax()]
##                ax.annotate(s=int(peak.name*1000000), xy=(peak.name,smooth_distance.max()))
#            linked = cdf[(cdf.mut>=0.9) & (cdf.sib<=0.5) & (cdf.qual>args.minquality)]
#            if len(linked)>5 and plot_kde:
#                sns.kdeplot(linked.index,bw=0.5,ax=ax,lw=1,color=sns.color_palette("deep")[2],shade=True)
#                kde_x, kde_y = sns.distributions._statsmodels_univariate_kde(linked.index,"gau",bw='scott', gridsize=100, cut=0, clip=(-np.inf, np.inf))
#                kde_peak = (kde_x[kde_y.argmax()], kde_y.max())
#                ax.annotate(s=int(kde_peak[0]*1000000), xy=kde_peak)
#        if genedf is not None: #draw a line indicating the potential genes
#            for gene in significant_genes[significant_genes.chr.astype(str)==c].itertuples():
#                color = sns.color_palette("bright")[2] if gene.log2FoldChange<0 else sns.color_palette("bright")[0]
#                ax.axvline(gene.start/1000000.0, color=color, lw=1)
#        ax.legend().set_visible(False)
#        ax.text(0.05, 0.95,c,fontsize=40, color='gray', horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)
#        plt.ylabel('Homozygosity')
#        plt.xlabel("Position (MB)")
#        if genedf is not None:
#            plt.savefig(datafilename[:-3]+c+'.genes.png')
#        else:
#            plt.savefig(datafilename[:-3]+c+'.png')
#

# %% Print a table of candidates
# snptable=pd.DataFrame(columns=['CHROM','POS','ID','REF','ALT'])
if args.dominant:
    maxmut_homozygosity = 0.10
    maxsib = 0.10
    if USE_SIBS:
        candidate_df = goodsnps.query("mut_homozygosity <= @maxmut_homozygosity and sib < @maxsib")
    else:
        candidate_df = goodsnps.query("mut_homozygosity <= @maxmut_homozygosity")
else:
    minmut = 0.90
    maxsib = 0.55
    if USE_SIBS:
        candidate_df = goodsnps.query("mut >= @minmut and sib < @maxsib")
    else:
        candidate_df = goodsnps.query("mut >= @minmut")
print("Closeup tables:", args.closeup)
#if closeup has been specified, only use those chromosomes. Otherwise dump them all.
if args.closeup:
    snptable = candidate_df[candidate_df.chr.isin(args.closeup)].copy()
    # also print all SNPs on this chr to a table in case I want to process them later
    df[~df.wt & df.chr.isin(args.closeup)].to_csv(datafilename[:-3]+'.all.txt', sep='\t', index=False)

snptable = candidate_df.copy()
snptable = snptable.reset_index()
snptable['pos'] = (snptable.pos*1000000).round().astype(int)
snptable['id'] = '.'
snptable['filter'] = '.'
if USE_SIBS:
    snptable['info'] = snptable.apply(lambda row: "mut=%0.2f;sib=%0.2f;mutdp=%s;sibdp=%s" % (
        row.mut, row.sib, row.mut_depth, row.sib_depth), axis=1)
else:
    snptable['info'] = snptable.apply(lambda row: "mut=%0.2f;mutdp=%s" % (row.mut, row.mut_depth), axis=1)
snptable = snptable[['chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info']]
print(len(snptable), "possible candidate SNPs")
with open(datafilename[:-3]+'candidates.vcf', 'w') as vcf_outfile:
    if args.dominant:
        vcf_outfile.write(f"#SNP table generated with following filter settings: Mut_homozygosity<={maxmut_homozygosity} Sib<{maxsib}\n#")
    else:
        vcf_outfile.write(f"#SNP table generated with following filter settings: Mut>={minmut} Sib<{maxsib}\n#")
    snptable.to_csv(vcf_outfile, sep='\t', index=False)


# %% Generate a list of markers in a given region
#region_left = 26
#region_right = 33
#desired_marker_num = 12
#linked_region = cdf[region_left:region_right].copy()
#linked_region.index = (linked_region.index*1000000).astype(int)
# print len(linked_region), "SNPs between", region_left, "and",region_right,"MB on chr",c
# what's the distance between subsequent SNPs?
#linked_region['next'] = np.ediff1d(linked_region.index,to_end=0)
#linked_region['prev'] = np.ediff1d(linked_region.index,to_begin=0)
# now select snps that make good markers
#markers = linked_region.query('qual >  200 and mut > 0.95 and sib < 0.45 and next > 50 and prev > 50')
#markers = markers[markers.ref.str.len() == 1].copy()
# print len(markers), "SNPs that meet quality and homozygosity thresholds"
# we probably have more than we need, so divide the region into 12
#markers['regionbin'] = pd.cut(markers.index,desired_marker_num)
# get the highest-quality SNP from each regionbin
#chosen = markers.groupby('regionbin').apply(lambda group: group.nlargest(1,'qual'))
# print len(chosen), "good quality markers found in", desired_marker_num, "bins"
# for snp in chosen.itertuples():
#    print c, snp.Index[1], snp.ref+snp.alt

# %%
# What's the distribution of sibling allele balances on the selected chromosome?
# plt.subplots(1,2,figsize=(12,6))
# plt.subplot(121)
#sns.distplot(cdf.query("mut>=0.97 and wt==False and pos>20 and pos<30").sib,kde_kws={'bw':0.1,'cut':0},bins=10)
#plt.suptitle("Distribution of sibling alleles in 10MB linked region when mut>0.97 (and no wts)")
#plt.title("All sib alleles")
# plt.subplot(122)
#sns.distplot(cdf.query("mut>=0.97 and sib<1 and wt==False and pos>20 and pos<30").sib,kde_kws={'bw':0.1,'cut':0},bins=10)
#plt.title("Where sib<1")

# %%

#fig, axes = plt.subplots(5,5,sharex=False, sharey=True,figsize=(12,8))
# for c,ax in zip(chrs,axes.flat):
#    thischrom = df.xs(c,level=0).query('wt==False')
#    pearson = -thischrom.mut.rolling(100,center=True).corr(thischrom.sib)
#    if pearson.max()>0:
#        print pearson.max(), "at", pearson.argmax(), "on chr",c
#    pearson.plot(ax=ax)
#    ax.set_ylim(0,1)
#plt.suptitle("Allele frequences for novel SNPs")
# plt.savefig(datafilename[:-3]+'kde'+'.png')
# %%
#thischrom = df.xs('7',level=0).query('wt==False')
#pearson = -thischrom.mut.rolling(100,center=True).corr(thischrom.sib)
# pearson.plot()
# df.query('wt==False').sib.hist(bins=np.arange(0,1.01,0.01))
# thischrom[['mut','sib']].rolling(10,center=True).mean().plot()
# thischrom[thischrom.sib<0.5].mut.rolling(20).mean().plot()
#plt.scatter(thischrom.sib, thischrom.mut,alpha=0.2)
#sns.jointplot(thischrom.sib, thischrom.mut, kind='reg')
# sns.kdeplot(thischrom.mut)
# sns.kdeplot(thischrom.sib)
# sns.distplot(thischrom.mut)
# sns.distplot(thischrom.sib)
