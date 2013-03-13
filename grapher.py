import pylab as pl
import matplotlib.ticker

x = [50, 100, 200, 350, 500, 750, 1000, 1500, 2000, 3000]
nb = [0.325482729659, 0.429439275187, 0.531682285183, 0.596954498991, 0.632602954403, 0.671056209368, 0.695411665197, 0.726261189874, 0.733298868566, 0.766558023762]
em = [0.464244069456, 0.582009952695, 0.659264285884, 0.697520642079, 0.721493016203, 0.741669928971, 0.747793463953, 0.744452134587, 0.757114779634, 0.770803256508]
sfe = [0.402058091271478, 0.514201599079121, 0.604026520030728, 0.648003198629936, 0.686753382406235, 0.712486292248545, 0.732678845017837, 0.741577172506784, 0.750471280188248, 0.771471755719851]

pl.clf()

fig = pl.figure()
ax = fig.add_subplot(1,1,1)

#ax.scatter(x, nb, c='r', lw=0)
#ax.scatter(x, sfe, c='g', lw=0)
#ax.scatter(x, em, c='b', lw=0)

ax.plot(x, em, 'b', marker='o', label='EM')
ax.plot(x, sfe, 'g', marker='s', label='SFE')
ax.plot(x, nb, 'r', marker='^', label='MNB')


#ax.set_xscale('log')
ax.set_xscale('log')
ax.set_ylim([0.0, 1.0])
ax.set_xlim([0, 3100])
#ax.get_xaxis().get_major_formatter().labelOnlyBase = False

#ax.get_xaxis().set_major_formatter(pl.ticker.ScalarFormatter())
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(round(x)))))

ax.set_xticks(x2)

pl.xlabel('Number of Labeled Documents')
pl.ylabel('Accuracy')
pl.legend(loc="lower right")
pl.show()                
