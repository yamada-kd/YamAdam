#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,re,argparse
import numpy as np
import theano
import theano.tensor as T
np.set_printoptions(threshold=np.inf,linewidth=np.inf,suppress=True,precision=12)

def main():
	parser=argparse.ArgumentParser()
	parser.add_argument('-d',type=str,dest="dataset",required=True,help="Dataset name.")
	parser.add_argument('-m',type=str,dest="method",required=True,help="Method name.")
	parser.add_argument('-e',type=int,dest="epoch",required=False,default=30,help="The number of epochs.")
	parser.add_argument('-r',type=int,dest="randomseed",required=False,default=0,help="Random number seed.")
	parser.add_argument('-o',type=str,dest="output",required=False,default=None,help='Output file name.')
	parser.add_argument('-a',type=str,dest="architecture",required=False,default=None,help='Network architecture.')
	args=parser.parse_args()
	
	np.random.seed(args.randomseed)
	
	EPOCH=args.epoch
	DATASET=args.dataset
	dataset=Dataset(DATASET)
	TN=dataset.TN
	CLASSNUM=dataset.CLASSNUM
	TBATCHSIZE=100
	TBATCHNUM=TN//TBATCHSIZE
	INPUT="db/sample/%(DATASET)s.txt"%locals()
	optimizer=eval(args.method)
	outfile=args.output
	
	litrainx,litraint,livalidx,livalidt=[],[],[],[]
	fin=open(INPUT,"r")
	for i,line in enumerate(fin):
		litmp=re.split("\s+",line.rstrip())
		lix=[float(i) for i in litmp[:-1]]
		lit=[int(litmp[-1])]
		if i<TN:
			litrainx.append(lix)
			litraint.append(lit)
		elif i<TN+dataset.VN:
			livalidx.append(lix)
			livalidt.append(lit)
	fin.close()
	litrainx,litraint,livalidx,livalidt=np.asarray(litrainx,dtype=np.float32),np.asarray(litraint,dtype=np.int32),np.asarray(livalidx,dtype=np.float32),np.asarray(livalidt,dtype=np.int32)
	
	x,t=T.fmatrix(name="x"),T.imatrix(name="t")
	activation=relu
	lilayer=[]
	if args.architecture==None:
		lilayer=[
		Layer(litrainx.shape[1],250,activation,0.5),
		Layer(250,250,activation,0.5),
		Layer(250,CLASSNUM,T.nnet.softmax,0)
		]
	elif args.architecture=="mlp01":
		lilayer=[
		Layer(litrainx.shape[1],500,activation,0.5),
		Layer(500,500,activation,0.5),
		Layer(500,CLASSNUM,T.nnet.softmax,0)
		]
	elif args.architecture=="lr01":
		lilayer=[
		Layer(litrainx.shape[1],CLASSNUM,T.nnet.softmax,0)
		]
	
	liparameter,lil2sqr,y=[],[],x
	for i,layer in enumerate(lilayer):
		liparameter+=layer.liparameter
#		lil2sqr.append(layer.L2_sqr)
		y=layer.forward(y)
	
	cost=-T.mean(T.log(y)[T.arange(t.shape[0]),t[:,0]])
	
	ligradient=T.grad(cost=cost,wrt=liparameter)
	lioptimize=optimizer(liparameter,ligradient)
	PARANUM=len(lioptimize)
	if PARANUM==2:
		if args.method=="vanillasgd":
			liupdate=optimizer(liparameter,ligradient)
			train=theano.function(inputs=[x,t],outputs=[cost,T.argmax(y,axis=1)],updates=liupdate)
		else:
			liupdate,b=optimizer(liparameter,ligradient)
			train=theano.function(inputs=[x,t],outputs=[cost,T.argmax(y,axis=1),b],updates=liupdate)
	else:
		liupdate=optimizer(liparameter,ligradient)
		train=theano.function(inputs=[x,t],outputs=[cost,T.argmax(y,axis=1)],updates=liupdate)
	
	valid=theano.function(inputs=[x,t],outputs=[cost,T.argmax(y,axis=1)])
	
	fout=open(outfile,"w") if outfile is not None else sys.stdout
	for epoch in range(EPOCH):
		index=np.random.permutation(TN)
		traincost,trainacc=0,0
		for i in range(TBATCHNUM):
			start=i*TBATCHSIZE
			end=start+TBATCHSIZE
			if PARANUM==2:
				if args.method=="vanillasgd":
					costnow,prednow=train(litrainx[index[start:end]],litraint[index[start:end]])
				else:
					costnow,prednow,b=train(litrainx[index[start:end]],litraint[index[start:end]])
					print(b,end=" ")
			else:
				costnow,prednow=train(litrainx[index[start:end]],litraint[index[start:end]])
			traincost+=costnow
			trainacc+=accuracy(litraint[index[start:end]],prednow)
		if PARANUM==2:
			if args.method=="vanillasgd":
				pass
			else:
				print("\n",end="")
		mean_traincost,mean_trainacc=traincost/TBATCHNUM,trainacc/TBATCHNUM
		mean_validcost,validpred=valid(livalidx,livalidt)
		mean_validacc=accuracy(livalidt,validpred)
		print('Epoch {0:6d}: Training cost={1:8.4f}, acc={2:8.4f}, Validation cost={3:8.4f}, acc={4:8.4f}'.format(epoch+1,float(mean_traincost),float(mean_trainacc),float(mean_validcost),float(mean_validacc)),file=fout)
	fout.close()

class Layer(object):
	def __init__(self,in_dim,out_dim,function,dropoutrate):
		self.W=theano.shared(value=np.asarray(np.random.randn(in_dim,out_dim)*np.sqrt(2/in_dim),dtype=theano.config.floatX),name="W") # He
		self.b=theano.shared(value=np.asarray(np.zeros(out_dim),dtype=theano.config.floatX),name="b") # He
		self.func=function
		self.liparameter=[self.W,self.b]
		self.L2_sqr=(self.W**2).sum()
		self.rate=dropoutrate
	def forward(self,x):
		rng=np.random.RandomState(0)
		u=T.dot(x,self.W)+self.b
		self.z=self.func(u)
		if self.rate>0: self.z=dropout(rng,self.z,self.rate)
		return self.z

def dropout(rng,x,p):
	srng=theano.tensor.shared_randomstreams.RandomStreams(rng.randint(1000))
	mask=srng.binomial(n=1,p=1-p,size=x.shape)
	output=x*T.cast(mask,theano.config.floatX)
	return output

class Dataset(object):
	def __init__(self,DATASET):
		if DATASET=="drive":
			self.TN,self.VN,self.CLASSNUM=40000,18509,11
		elif DATASET=="mnist":
			self.TN,self.VN,self.CLASSNUM=60000,10000,10
		elif DATASET=="letter":
			self.TN,self.VN,self.CLASSNUM=15000,5000,26

def accuracy(lix,liy):
	tp=0
	for x,y in zip(lix,liy):
		if x==y: tp+=1
	return tp/len(lix)

def yamadam(liparameter,ligradient,e=1e-6):
	liupdate=[]
	for pc,gc in zip(liparameter,ligradient):
		mc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='mc')
		vc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='vc')
		sc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='sc')
		hc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='hc')
		bc=theano.shared(value=np.array([0],dtype=theano.config.floatX)[0],name='bc')
		mn=bc*mc+(1-bc)*gc
		vn=bc*vc+(1-bc)*(gc-mc)**2
		sn=bc*sc+(1-bc)*hc**2
		hn=mn*T.sqrt((sn+e)/(vn+e))
		bn=T.nnet.sigmoid(((abs(hc)).sum()+e)/((abs(hn)).sum()+e))-e
		pn=pc-hn
		liupdate.append((mc,mn))
		liupdate.append((vc,vn))
		liupdate.append((sc,sn))
		liupdate.append((hc,hn))
		liupdate.append((pc,pn))
		liupdate.append((bc,bn))
	return liupdate

def adam(liparameter,ligradient,a=0.001,b1=0.9,b2=0.999,e=1e-6):
	liupdate=[]
	t=theano.shared(value=np.float32(1),name="t")
	liupdate.append((t,t+1))
	for pc,gc in zip(liparameter,ligradient):
		mc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='mc')
		vc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='vc')
		mn=b1*mc+(1-b1)*gc
		vn=b2*vc+(1-b2)*gc**2
		mh=mn/(1-b1**t)
		vh=vn/(1-b2**t)
		pn=pc-(a*mh)/(T.sqrt(vh+e))
		liupdate.append((mc,mn))
		liupdate.append((vc,vn))
		liupdate.append((pc,pn))
	return liupdate

def adadelta(liparameter,ligradient,b=0.95,e=1e-6):
	liupdate=[]
	for pc,gc in zip(liparameter,ligradient):
		vc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='vc')
		sc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='sc')
		hc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='hc')
		vn=b*vc+(1-b)*gc**2
		sn=b*sc+(1-b)*hc**2
		hn=gc*T.sqrt((sn+e)/(vn+e))
		pn=pc-hn
		liupdate.append((pc,pn))
		liupdate.append((vc,vn))
		liupdate.append((sc,sn))
		liupdate.append((hc,hn))
	return liupdate

def rmsprop(liparameter,ligradient,a=0.001,b=0.9,e=1e-6):
	liupdate=[]
	for pc,gc in zip(liparameter,ligradient):
		rc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='rc')
		rn=b*rc+(1-b)*gc**2
		pn=pc-(a*gc)/T.sqrt(rn+e)
		liupdate.append((pc,pn))
		liupdate.append((rc,rn))
	return liupdate

def adagrad(liparameter,ligradient,a=0.01,e=1e-6):
	liupdate=[]
	for pc,gc in zip(liparameter,ligradient):
		rc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='rc')
		rn=rc+gc**2
		pn=pc-(a*gc)/T.sqrt(rn+e)
		liupdate.append((pc,pn))
		liupdate.append((rc,rn))
	return liupdate

def momentumsgd(liparameter,ligradient,a=0.01,b=0.9):
	liupdate=[]
	for pc,gc in zip(liparameter,ligradient):
		vc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='vc')
		vn=vc*b-a*gc
		pn=pc+vn
		liupdate.append((pc,pn))
		liupdate.append((vc,vn))
	return liupdate

def vanillasgd(liparameter,ligradient,a=0.01):
	liupdate=[]
	for pc,gc in zip(liparameter,ligradient):
		pn=pc-a*gc
		liupdate.append((pc,pn))
	return liupdate

def relu(x):
	return x*(x>0)

if __name__ == '__main__':
	main()
