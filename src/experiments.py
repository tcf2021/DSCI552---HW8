"""Reproducible Monte Carlo experiments for DSCI 552 HW8."""
from pathlib import Path
import json
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
from scipy.special import softmax
from scipy.optimize import linear_sum_assignment, minimize
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits
SEED=552
BREAST_C=(.01,.1,1.,10.,100.)
BANK_C=(.01,.1,1.,10.)

class ConvergedLinearSVC(LinearSVC):
    def fit(self,X,y,sample_weight=None):
        for limit in (50000,500000,2000000):
            self.max_iter=limit
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always",ConvergenceWarning)
                super().fit(X,y,sample_weight=sample_weight)
            failed=any(issubclass(w.category,ConvergenceWarning) for w in caught)
            for w in caught:
                if not issubclass(w.category,ConvergenceWarning):
                    warnings.warn(str(w.message),w.category)
            if not failed:return self
        # Rare ill-conditioned fits: solve the same convex primal objective.
        # LIBLINEAR penalizes the synthetic intercept feature as well as weights.
        Z=np.c_[X,np.full(len(X),self.intercept_scaling)]
        signs=np.where(y==self.classes_[1],1.,-1.)
        weights=np.ones(len(y)) if sample_weight is None else sample_weight
        d=Z.shape[1]
        def objective(v):
            beta=v[:d]-v[d:]
            residual=np.maximum(0.,1.-signs*(Z@beta))
            loss=v.sum()+self.C*np.dot(weights,residual**2)
            grad=-2*self.C*(Z.T@(weights*signs*residual))
            return loss,np.r_[1+grad,1-grad]
        beta=np.r_[self.coef_[0],self.intercept_[0]/self.intercept_scaling]
        start=np.r_[np.maximum(beta,0),np.maximum(-beta,0)]
        result=minimize(objective,start,jac=True,method="L-BFGS-B",bounds=[(0,None)]*(2*d),
            options={"maxiter":100000,"maxls":100,"ftol":1e-15,"gtol":1e-7})
        _,gradient=objective(result.x)
        projected=np.where(result.x>1e-9,gradient,np.minimum(gradient,0))
        if np.max(np.abs(projected))>1e-3:
            raise RuntimeError("L1 primal solver failed its optimality check")
        beta=result.x[:d]-result.x[d:]
        self.coef_=beta[:-1][None,:]
        self.intercept_=np.array([beta[-1]*self.intercept_scaling])
        self.n_iter_=result.nit
        return self

def svm(C,seed):
    return ConvergedLinearSVC(C=C,penalty="l1",loss="squared_hinge",dual=False,
                     max_iter=50000,tol=1e-4,random_state=seed)

def choose_c(X,y,grid,seed,scale_class):
    """Five-fold accuracy; scaling fitted inside each fold."""
    if np.bincount(y,minlength=2).min()>=5:
        folds=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y))
    else:
        # A ten-point pool may contain fewer than five members of a class.
        # Keep five folds, allowing one-class validation folds but not fitting folds.
        attempt=0
        while True:
            folds=list(KFold(5,shuffle=True,random_state=seed+attempt).split(X))
            if all(np.unique(y[a]).size==2 for a,b in folds):break
            attempt+=1
    prepared=[]
    for a,b in folds:
        sc=scale_class().fit(X[a])
        prepared.append((sc.transform(X[a]),y[a],sc.transform(X[b]),y[b]))
    rows=[]
    for C in grid:
        acc=[accuracy_score(yb,svm(C,seed).fit(xa,ya).predict(xb)) for xa,ya,xb,yb in prepared]
        rows.append({"C":C,"cv_accuracy":np.mean(acc)})
    idx=int(np.argmax([r["cv_accuracy"] for r in rows]))
    return grid[idx],rows

def measures(run,method,split,y,pred,score):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(run=run,method=method,split=split,accuracy=accuracy_score(y,pred),
        precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),
        F1=f1_score(y,pred,zero_division=0),AUC=roc_auc_score(y,score),
        TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))

def breast_trial(X,y,r,out):
    with threadpool_limits(limits=1):
        rng=np.random.default_rng(SEED+r)
        te=np.concatenate([rng.choice(np.flatnonzero(y==c),int(np.ceil(.2*sum(y==c))),replace=False) for c in (0,1)])
        tr=np.setdiff1d(np.arange(len(y)),te)
        A,B=X[tr],X[te];ya,yb=y[tr],y[te]
        norm=MinMaxScaler().fit(A);a,b=norm.transform(A),norm.transform(B)
        records=[];predictions=[];params=[];self_trace=[]
        def record(name,pt,pb,st,sb):
            records.extend([measures(r+1,name,"Train",ya,pt,st),measures(r+1,name,"Test",yb,pb,sb)])
            if r==0:
                for split,idx,yy,pp,ss in [("Train",tr,ya,pt,st),("Test",te,yb,pb,sb)]:
                    predictions.extend(dict(method=name,split=split,row_index=int(i),truth=int(v),prediction=int(p),score=float(s))
                        for i,v,p,s in zip(idx,yy,pp,ss))
        C,cv=choose_c(A,ya,BREAST_C,SEED+r,MinMaxScaler)
        model=svm(C,SEED+r).fit(a,ya)
        record("Supervised",model.predict(a),model.predict(b),model.decision_function(a),model.decision_function(b))
        params.extend(dict(run=r+1,method="Supervised",selected=C,**v) for v in cv)
        labeled=np.concatenate([rng.choice(np.flatnonzero(ya==c),sum(ya==c)//2,replace=False) for c in (0,1)])
        pool=np.setdiff1d(np.arange(len(ya)),labeled)
        # The true labels of pool are not accessed during acquisition.
        labels=ya[labeled].copy()
        C,cv=choose_c(A[labeled],labels,BREAST_C,SEED+r,MinMaxScaler)
        params.extend(dict(run=r+1,method="Self-training",selected=C,**v) for v in cv)
        model=svm(C,SEED+r).fit(a[labeled],labels)
        initial=labeled.copy()
        for step in range(len(pool)):
            margin=model.decision_function(a[pool]); pick=int(np.argmax(np.abs(margin)))
            idx=pool[pick];pseudo=int(margin[pick]>0)
            self_trace.append(dict(run=r+1,step=step+1,row_index=int(tr[idx]),pseudo_label=pseudo))
            labeled=np.append(labeled,idx);labels=np.append(labels,pseudo);pool=np.delete(pool,pick)
            model=svm(C,SEED+r).fit(a[labeled],labels)
        record("Self-training",model.predict(a),model.predict(b),model.decision_function(a),model.decision_function(b))
        km=KMeans(n_clusters=2,init="random",n_init=30,random_state=SEED+r).fit(a)
        da,db=km.transform(a),km.transform(b)
        # Closest 30 training rows to each centroid, regardless of assigned cluster.
        nearest=[np.argsort(da[:,j],kind="stable")[:30] for j in (0,1)]
        fractions=np.array([ya[idx].mean() for idx in nearest])
        classes=(fractions>.5).astype(int)
        pa,pb=classes[km.labels_],classes[km.predict(b)]
        sa=softmax(-da,axis=1)@classes;sb=softmax(-db,axis=1)@classes
        record("K-means",pa,pb,sa,sb)
        def spectral(x,seed):
            return SpectralClustering(n_clusters=2,affinity="rbf",gamma=1.,assign_labels="kmeans",
                n_init=30,random_state=seed,eigen_solver="arpack").fit_predict(x)
        ca=spectral(a,SEED+r);cb=spectral(b,SEED+1000+r)
        rates=np.array([ya[ca==j].mean() for j in (0,1)])
        W=rbf_kernel(b,a,gamma=1.)
        overlap=np.array([[W[np.ix_(cb==j,ca==k)].mean() for k in (0,1)] for j in (0,1)])
        rr,cc=linear_sum_assignment(-overlap)
        align=np.empty(2,dtype=int);align[rr]=cc
        sa,sb=rates[ca],rates[align[cb]]
        record("Spectral",(sa>.5).astype(int),(sb>.5).astype(int),sa,sb)
        info=dict(run=r+1,train_negative=int(sum(ya==0)),train_positive=int(sum(ya==1)),
            test_negative=int(sum(yb==0)),test_positive=int(sum(yb==1)),
            labeled_negative=int(sum(ya[initial]==0)),labeled_positive=int(sum(ya[initial]==1)),
            pseudo_count=len(self_trace),kmeans_inertia=km.inertia_,
            kmeans_cluster_0=int(sum(km.labels_==0)),kmeans_cluster_1=int(sum(km.labels_==1)),
            kmeans_positive_fraction_0=fractions[0],kmeans_positive_fraction_1=fractions[1],
            spectral_cluster_0=int(sum(ca==0)),spectral_cluster_1=int(sum(ca==1)),
            spectral_rate_0=rates[0],spectral_rate_1=rates[1])
        pd.DataFrame(records).to_csv(out/f"breast_run_{r+1:02}.csv",index=False)
        return records,params,predictions,info,self_trace,[(r+1,int(i),"Test" if i in te else "Train") for i in range(len(y))]

def run_breast(data,out):
    raw=pd.read_csv(data/"wdbc.data",header=None)
    X=raw.iloc[:,2:].to_numpy();y=(raw[1]=="M").to_numpy(dtype=int)
    with parallel_config(backend="loky",inner_max_num_threads=1):
        results=Parallel(n_jobs=4)(delayed(breast_trial)(X,y,r,out) for r in range(30))
    names=["breast_metrics","breast_cv","breast_run1_predictions","breast_run_info","self_training_trace"]
    for j,name in enumerate(names):
        rows=[v for row in results for v in (row[j] if j!=3 else [row[j]])]
        pd.DataFrame(rows).to_csv(out/(name+".csv"),index=False)
    pd.DataFrame([v for row in results for v in row[5]],columns=["run","row_index","split"]).to_csv(out/"breast_splits.csv",index=False)
    return pd.read_csv(out/"breast_metrics.csv")

def bank_trial(A,y,B,yt,r,out):
    with threadpool_limits(limits=1):
        rng=np.random.default_rng(SEED+2000+r)
        # Same initial random ten observations for both strategies.
        while True:
            initial=rng.choice(len(y),10,replace=False)
            if np.bincount(y[initial],minlength=2).min()>=2:break
        remaining=np.setdiff1d(np.arange(len(y)),initial)
        passive_order=np.r_[initial,rng.permutation(remaining)]
        rows=[];cvrows=[];orders=[]
        for method in ["Passive","Active"]:
            chosen=initial.copy();pool=remaining.copy()
            for size in range(10,901,10):
                if method=="Passive":chosen=passive_order[:size]
                C,cv=choose_c(A[chosen],y[chosen],BANK_C,SEED+2000+r,StandardScaler)
                scaler=StandardScaler().fit(A[chosen])
                model=svm(C,SEED+2000+r).fit(scaler.transform(A[chosen]),y[chosen])
                error=np.mean(model.predict(scaler.transform(B))!=yt)
                rows.append(dict(run=r+1,method=method,size=size,C=C,test_error=error))
                cvrows.extend(dict(run=r+1,method=method,size=size,**v) for v in cv)
                if method=="Active" and size<900:
                    d=np.abs(model.decision_function(scaler.transform(A[pool])))
                    near=np.argsort(d,kind="stable")[:10]
                    chosen=np.r_[chosen,pool[near]];pool=np.delete(pool,near)
            orders.extend(dict(run=r+1,method=method,order=j+1,train_position=int(idx)) for j,idx in enumerate(chosen))
        pd.DataFrame(rows).to_csv(out/f"bank_run_{r+1:02}.csv",index=False)
        return rows,cvrows,orders

def run_bank(data,out):
    raw=pd.read_csv(data/"data_banknote_authentication.txt",header=None)
    X=raw.iloc[:,:4].to_numpy();y=raw[4].to_numpy(dtype=int)
    tr,te=train_test_split(np.arange(len(y)),test_size=472,random_state=SEED)
    pd.DataFrame({"row_index":np.r_[tr,te],"split":["Train"]*900+["Test"]*472}).to_csv(out/"bank_split.csv",index=False)
    with parallel_config(backend="loky",inner_max_num_threads=1):
        results=Parallel(n_jobs=4)(delayed(bank_trial)(X[tr],y[tr],X[te],y[te],r,out) for r in range(50))
    for j,name in enumerate(["bank_learning_errors","bank_cv","bank_acquisition_order"]):
        pd.DataFrame([v for row in results for v in row[j]]).to_csv(out/(name+".csv"),index=False)
    return pd.read_csv(out/"bank_learning_errors.csv")
