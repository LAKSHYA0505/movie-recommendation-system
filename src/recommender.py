"""Starter pipeline: clean -> EDA -> user-CF / item-CF / SVD -> RMSE, Precision@K, diversity.
Run: python recommender.py   (needs pandas, numpy, matplotlib)"""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
df=pd.read_csv('dataset.csv',parse_dates=['timestamp'])
# ---------- 1. Cleaning ----------
n0=len(df)
for c in ['genre','language','release_year']:          # one canonical value per title
    df[c]=df.groupby('movie_title')[c].transform(lambda s:s.mode()[0])
df=df.sort_values('timestamp')
gap=df.groupby('user_id').timestamp.diff().dt.total_seconds()
burst=df.assign(g=gap<60).groupby('user_id').g.mean()
bots=burst[(burst>0.5)&(df.user_id.value_counts()[burst.index]>=10)].index       # many ratings seconds apart
df=df[~df.user_id.isin(bots)]
df=df.drop_duplicates(['user_id','movie_title'],keep='first')                   # keep first rating per pair
print(f'Cleaning: {n0} -> {len(df)} rows, removed bot users {list(bots)}')
# ---------- 2. EDA ----------
fig,ax=plt.subplots(1,3,figsize=(16,4))
df.rating.plot.hist(bins=9,ax=ax[0],title='Rating distribution')
df.language.value_counts().plot.bar(ax=ax[1],title='Ratings per language')
df.groupby('genre').rating.mean().sort_values().plot.barh(ax=ax[2],title='Average rating by genre')
plt.tight_layout(); plt.savefig('eda.png',dpi=120)
# ---------- 3. Split: hold out 20% of each user's ratings (try a time-based split as an extension) ----------
te=df.groupby('user_id').sample(frac=.2,random_state=1); tr=df.drop(te.index)
te=te[te.user_id.isin(tr.user_id)&te.movie_title.isin(tr.movie_title)]
users=sorted(tr.user_id.unique()); items=sorted(tr.movie_title.unique())
ui={u:i for i,u in enumerate(users)}; ii={m:i for i,m in enumerate(items)}
R=np.zeros((len(users),len(items))); R[tr.user_id.map(ui),tr.movie_title.map(ii)]=tr.rating
M=R>0; mu=tr.rating.mean()
umean=np.where(M.sum(1)>0,R.sum(1)/np.maximum(M.sum(1),1),mu)
def cos(X):
    n=np.linalg.norm(X,axis=1,keepdims=True)+1e-9; return (X/n)@(X/n).T
def baseline(ep=20,reg=5):                           # mean + user bias + item bias
    bu=np.zeros(len(users)); bi=np.zeros(len(items))
    for _ in range(ep):
        bi=((R-mu-bu[:,None])*M).sum(0)/(M.sum(0)+reg); bu=((R-mu-bi[None])*M).sum(1)/(M.sum(1)+reg)
    return mu+bu[:,None]+bi[None]
B=baseline(); C=np.where(M,R-B,0)            # CF works on residuals after removing biases
def user_cf(k=40,lam=2):
    S=cos(C); np.fill_diagonal(S,0); P=np.zeros_like(R)
    for u in range(len(users)):
        nb=np.argsort(-S[u])[:k]; w=S[u,nb][:,None]*M[nb]
        P[u]=B[u]+(w*C[nb]).sum(0)/(np.abs(w).sum(0)+lam)
    return P
def item_cf(k=40,lam=2):
    S=cos(C.T); np.fill_diagonal(S,0)
    top=np.argsort(-S,axis=1)[:,:k]; mask=np.zeros_like(S,bool); np.put_along_axis(mask,top,True,1); S=S*mask
    return B+(C@S.T)/((M.astype(float))@np.abs(S).T+lam)
def svd_mf(f=10,ep=80,lr=.007,reg=.1,seed=0):       # Funk-SVD with biases (same idea as surprise.SVD)
    r=np.random.default_rng(seed); P=r.normal(0,.1,(len(users),f)); Q=r.normal(0,.1,(len(items),f)); bu=np.zeros(len(users)); bi=np.zeros(len(items))
    obs=np.argwhere(M)
    for _ in range(ep):
        for u,i in r.permutation(obs):
            e=R[u,i]-(mu+bu[u]+bi[i]+P[u]@Q[i]); bu[u]+=lr*(e-reg*bu[u]); bi[i]+=lr*(e-reg*bi[i])
            P[u],Q[i]=P[u]+lr*(e*Q[i]-reg*P[u]),Q[i]+lr*(e*P[u]-reg*Q[i])
    return mu+bu[:,None]+bi[None]+P@Q.T
meta=df.drop_duplicates('movie_title').set_index('movie_title').loc[items]
pop=M.sum(0); novelty_w=-np.log2((pop+1)/len(users))
def evaluate(name,P,K=10):
    P=np.clip(P,1,5); u=te.user_id.map(ui).values; i=te.movie_title.map(ii).values
    rmse=np.sqrt(np.mean((P[u,i]-te.rating.values)**2))
    prec,ild,nov,recs=[],[],[],set()
    for uid,g in te.groupby('user_id'):
        x=ui[uid]; s=np.where(M[x],-np.inf,P[x]); top=np.argsort(-s)[:K]; recs.update(top)
        rel=set(g[g.rating>=4].movie_title.map(ii)); 
        if rel: prec.append(len(rel&set(top))/K)
        gg=meta.genre.values[top]; ll=meta.language.values[top]
        ild.append(np.mean([(gg[a]!=gg[b])|(ll[a]!=ll[b]) for a in range(K) for b in range(a+1,K)]))
        nov.append(novelty_w[top].mean())
    return dict(model=name,RMSE=round(rmse,3),**{f'Precision@{K}':round(np.mean(prec),3)},
                coverage=round(len(recs)/len(items),3),intra_list_diversity=round(np.mean(ild),3),novelty_bits=round(np.mean(nov),2))
res=[evaluate('Global mean',np.full_like(R,mu)),evaluate('Baseline (biases)',B),evaluate('User-based CF',user_cf()),evaluate('Item-based CF',item_cf()),evaluate('SVD (MF)',svd_mf())]
out=pd.DataFrame(res); print(out.to_string(index=False)); out.to_csv('results.csv',index=False)
