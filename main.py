from scipy import stats
import statistics
import math


#Confidence Interval with Known population Variance
def confidenceIntervalKnown(mean,CL,n,std):
     return [mean-(stats.norm.ppf((1-CL)/2 + CL,0,1) * (std/math.sqrt(n))),
             mean + (stats.norm.ppf((1-CL)/2 + CL,0,1) * (std/math.sqrt(n)))]

#Confidence Interval with Unknown population Variance
def confidenceIntervalUnknown(mean,CL,n,std):
    return [mean - (stats.t.ppf((1 - CL) / 2 + CL, n-1) * (std / math.sqrt(n))),
            mean + (stats.t.ppf((1 - CL) / 2 + CL, n-1) * (std / math.sqrt(n)))]


#Confidence Interval for Difference of Two Means(KnownVariance)
#Mean,n and std must be array with 1,2
def differenceConfidenceIntervalKnown(mean, CL, n, std):
    return [(mean[0]-mean[1]) - (stats.norm.ppf((1 - CL) / 2 + CL, 0, 1) * (math.sqrt(std[0] ** 2 / n[0] + std[1] ** 2 / n[1]))),
            (mean[0]-mean[1]) + (stats.norm.ppf((1 - CL) / 2 + CL, 0, 1) * (math.sqrt(std[0] ** 2 / n[0] + std[1] ** 2 / n[1])))]

#Confidence Interval for Difference of Two Means(UnknownVariance)
# + Population Variances Equal and
# for small Sample

def pooledSampleVariance(n,std):
    return math.sqrt(((n[0]-1) * (std[0] ** 2) + (n[1]-1)* (std[1] ** 2)) /(n[0]+n[1]-2))

def differenceConfidenceIntervalUnknownSmall(mean, CL, n, std):
    return [(mean[0]-mean[1]) - (stats.t.ppf((1 - CL) / 2 + CL, n[0]+n[1] -2) * (pooledSampleVariance(n,std) * math.sqrt(1/ n[0] + 1 / n[1]))),
            (mean[0]-mean[1]) + (stats.t.ppf((1 - CL) / 2 + CL, n[0]+n[1] -2) * (pooledSampleVariance(n,std)* math.sqrt(1 / n[0] + 1 / n[1])))]


#Confidence Interval for Difference of Two Means(UnknownVariance)
# + Population Variances Equal and
# for large Sample

def differenceConfidenceIntervalUnknownLarge(mean, CL, n, std):
    return [(mean[0]-mean[1]) - (stats.norm.ppf((1 - CL) / 2 + CL, 0, 1) * (pooledSampleVariance(n,std) * math.sqrt(1/ n[0] + 1 / n[1]))),
            (mean[0]-mean[1]) + (stats.norm.ppf((1 - CL) / 2 + CL, 0, 1) * (pooledSampleVariance(n,std)* math.sqrt(1 / n[0] + 1 / n[1])))]

#Confidence Interval for Variances
#Case 1:Population Mean is known
def partA(entry,mean):
    result = 0
    for i in entry:
        result += (entry[i] - mean) ** 2
    return result

def varianceConfidenceIntervalKnown(entry,mean,n,CL):
    return [partA(entry,mean)/stats.chi2.ppf((1 - CL) / 2,n), partA(entry,mean)/stats.chi2.ppf((1 - CL) / 2 +CL,n)]

#Case 2: Population Mean is unknown

def varianceConfidenceIntervalUnknown(SV,n,CL):
    return[(n-1) * SV/stats.chi2.ppf((1 - CL) / 2  + CL,n-1),(n-1)*SV/stats.chi2.ppf((1 - CL) / 2 ,n-1)]

# C.I for ratio of two variance with Unknown Means
def ratioVariancewithUnknownMean(SV,n,CL):
    return[(SV[0]/SV[1]) * (1/stats.f.ppf((1 - CL) / 2  + CL ,n[0]-1,n[1]-1)) , (SV[0]/SV[1]) * (stats.f.ppf((1 - CL) / 2 + CL,n[1]-1,n[0]-1))]

#p value for not equal

def pValueforNotEqual(x):
    return 2 * min(stats.norm.cdf(x,0,1),1-stats.norm.cdf(x,0,1))


