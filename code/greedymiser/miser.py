import shlex, subprocess,sys,time
from math import sqrt
import numpy as np
from operator import itemgetter as iget
import os, sys, inspect
from . import evtools #import evaluate, rel2ranks
class miser:
    def __init__(self):
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        dirname = os.path.dirname(os.path.abspath(filename))
        self.INPUTDATA=dirname+"/data/small.txt"
        self.TESTDATA=dirname+"/data/small.txt"
        self.NUM_FEATURES=700
        self.DEPTH=4
        self.ITER=16
        self.STEPSIZE=0.1
        self.PROCESSORS=8
        self.COSTS=[1.0]*self.NUM_FEATURES
    def fit(self, path=None,costs=None,l=None):
        if path is not None:
            filename = inspect.getframeinfo(inspect.currentframe()).filename
            dirname = os.path.dirname(os.path.abspath(filename))
            dirname=os.path.abspath(os.path.join(dirname, os.path.pardir))
            dirname=os.path.abspath(os.path.join(dirname, os.path.pardir))
            self.INPUTDATA=dirname+"/data/" + path+"/svm/train.txt"
            self.TESTDATA=dirname+"/data/" + path +"/svm/test.txt"
            f= open(self.INPUTDATA,"r")
            self.NUM_FEATURES=int(f.readline().split()[-1].split(":")[0])
            f.close()
        # read in costs
        if costs == None:
            costs=[1.0]*self.NUM_FEATURES
        else:
            W=[a for a in open(dirname+'/data/'+path+'/svm/costs.txt', 'r').read().split(',') if len(a)>0];
            costs=[float(a) for a in W]
        if l is not None:
            costs=np.array(costs)*l
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        dirname = os.path.dirname(os.path.abspath(filename))
        cmdline = '%s/cart/rtrank %s %s /dev/null -f %i -r %i -d %i -z -p %i -C' %  (dirname,self.INPUTDATA,self.TESTDATA,self.NUM_FEATURES,self.ITER,self.DEPTH,self.PROCESSORS)
        args = shlex.split(cmdline);



        # start loading the data
        p = subprocess.Popen(args, stdin=subprocess.PIPE,stdout=subprocess.PIPE)

        # read in targets
        def readtargets(filename):
        	p1=subprocess.Popen(shlex.split('cut -f 1,2 -d\  %s' % filename),stdout=subprocess.PIPE);
        	output=[a.decode('UTF-8').split(' ',1) for a in p1.stdout.readlines()  if len(a)>0];
        	ys=[float(a[0]) for a in output];
        	qs=[int(a[1].split(':')[1]) for a in output];
        	return(ys,qs)

        [traintargets,trainqueries]=readtargets(self.INPUTDATA)
        [testtargets,testqueries]=readtargets(self.TESTDATA)
        targets=traintargets+testtargets



        ntra=len(traintargets)
        ntst=len(testtargets)
        nall=ntra+ntst

        preds = [0]*nall
        # Run boosting
        for itr in range(self.ITER):

            [TRrmse,TRerr,TRndcg]=evtools.evaluate(preds[0:ntra],trainqueries,traintargets)
            [TErmse,TEerr,TEndcg]=evtools.evaluate(preds[ntra:nall],testqueries,testtargets)
            print("%i,  %2.5f,    %2.5f" % (itr,TRrmse,TErmse), file=sys.stderr)
            sys.stdout.flush()

            # write target
            k=0
            for cost in costs:
                p.stdin.write('%2.4f\n'.encode('UTF-8')  %(cost) )
                # print(cost)
                # p.stdin.write('%2.4f\n'.encode('UTF-8')  %(k) )
                # k+=1
            for i in range(ntra):
                p.stdin.write('%2.4f\n'.encode('UTF-8') % (traintargets[i]-preds[i]))
                # print(traintargets[i]-preds[i])
                # p.stdin.write('%2.4f\n'.encode('UTF-8') % (-k))
                # k+=1
            p.stdin.flush()

            # print([p.decode('UTF-8') for p in p.stdout.readlines()])
            # break
            # read costs
            for c in range(len(costs)):
                l=p.stdout.readline()
                costs[c]=float(l.decode('UTF-8').split(' ',1)[0])
                # print(str(c)+"  feature = " +l.decode('UTF-8'))
            self.COSTS=costs
            print(costs)
            print("feature unused:" +str(np.count_nonzero(costs)))
            # read all predictions
            for i in range(nall):
                l=p.stdout.readline()
                # print(str(i)+"  line = " +l.decode('UTF-8'))
                preds[i] += self.STEPSIZE*float(l.decode('UTF-8').split(' ',1)[0])
        #     if itr==1:
        #         break
        # print('\n'.join(map(str,preds)))