import pandas as pd, numpy as np, re
rng=np.random.default_rng(28)
orig=pd.read_csv('/data/.agent-service/files/835300b6-7d73-463f-8a38-e4854f0da30e/dataset.csv')
START,END=pd.Timestamp('2024-01-01'),pd.Timestamp('2025-12-01 23:59:59')
# ---- Indian regional catalogue (real films, primary genre) ----
cat=[("3 Idiots","Comedy","Hindi",2009),("Dangal","Drama","Hindi",2016),("Pink","Drama","Hindi",2016),("Andhadhun","Thriller","Hindi",2018),
("Lagaan","Drama","Hindi",2001),("Zindagi Na Milegi Dobara","Comedy","Hindi",2011),("Queen","Comedy","Hindi",2014),("Gully Boy","Drama","Hindi",2019),
("Kahaani","Thriller","Hindi",2012),("Barfi!","Romance","Hindi",2012),("PK","Sci-Fi","Hindi",2014),("Koi... Mil Gaya","Sci-Fi","Hindi",2003),
("Jab We Met","Romance","Hindi",2007),("Tumbbad","Horror","Hindi",2018),("Stree","Horror","Hindi",2018),("War","Action","Hindi",2019),
("Pathaan","Action","Hindi",2023),("12th Fail","Drama","Hindi",2023),("Dilwale Dulhania Le Jayenge","Romance","Hindi",1995),("Article 15","Thriller","Hindi",2019),
("Baahubali","Action","Telugu",2015),("RRR","Action","Telugu",2022),("Pushpa: The Rise","Action","Telugu",2021),("Arjun Reddy","Romance","Telugu",2017),
("Eega","Sci-Fi","Telugu",2012),("Jersey","Drama","Telugu",2019),("Mahanati","Drama","Telugu",2018),("Ala Vaikunthapurramuloo","Comedy","Telugu",2020),
("Kalki 2898 AD","Sci-Fi","Telugu",2024),("Geetha Govindam","Romance","Telugu",2018),("Evaru","Thriller","Telugu",2019),("Jathi Ratnalu","Comedy","Telugu",2021),
("Vikram","Action","Tamil",2022),("Enthiran","Sci-Fi","Tamil",2010),("Jai Bhim","Drama","Tamil",2021),("Super Deluxe","Drama","Tamil",2019),
("Ratsasan","Thriller","Tamil",2018),("96","Romance","Tamil",2018),("Master","Action","Tamil",2021),("Soorarai Pottru","Drama","Tamil",2020),
("Kaithi","Action","Tamil",2019),("Doctor","Comedy","Tamil",2021),("Vada Chennai","Thriller","Tamil",2018),("Indru Netru Naalai","Sci-Fi","Tamil",2015),
("KGF","Action","Kannada",2018),("KGF: Chapter 2","Action","Kannada",2022),("Kantara","Thriller","Kannada",2022),("777 Charlie","Drama","Kannada",2022),
("Kirik Party","Comedy","Kannada",2016),("U Turn","Thriller","Kannada",2016),("Lucia","Sci-Fi","Kannada",2013),("Mungaru Male","Romance","Kannada",2006),
("Ondu Motteya Kathe","Comedy","Kannada",2017),("Garuda Gamana Vrishabha Vahana","Action","Kannada",2021),
("Drishyam","Thriller","Malayalam",2013),("Premam","Romance","Malayalam",2015),("Kumbalangi Nights","Drama","Malayalam",2019),("Minnal Murali","Sci-Fi","Malayalam",2021),
("Manjummel Boys","Thriller","Malayalam",2024),("Bangalore Days","Comedy","Malayalam",2014),("Android Kunjappan Version 5.25","Sci-Fi","Malayalam",2019),("Aavesham","Action","Malayalam",2024),
("Pather Panchali","Drama","Bengali",1955),("Bhooter Bhabishyat","Comedy","Bengali",2012),("Byomkesh Bakshi","Thriller","Bengali",2010),("Praktan","Romance","Bengali",2016),
("Sairat","Romance","Marathi",2016),("Natsamrat","Drama","Marathi",2016),("Court","Drama","Marathi",2014),("Duniyadari","Romance","Marathi",2013),
("Qissa","Drama","Punjabi",2013),("Chal Mera Putt","Comedy","Punjabi",2019)]
ind=pd.DataFrame(cat,columns=['movie_title','genre','language','release_year'])
# ---- MovieLens 100K ----
item=pd.read_csv('/data/ml-100k/u.item',sep='|',header=None,encoding='latin-1')
gnames=['unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gmap={"Children's":'Animation','Fantasy':'Adventure','Film-Noir':'Crime','Mystery':'Thriller','War':'Drama','Western':'Action','Musical':'Romance','unknown':'Drama'}
prio=['Sci-Fi','Animation','Horror','Documentary','Thriller','Crime','Action','Adventure','Romance','Comedy','Drama']
def prim(r):
    g=[gmap.get(n,n) for n,v in zip(gnames,r[5:24]) if v==1] or ['Drama']
    return next(p for p in prio if p in g)
def clean(t):
    m=re.match(r'^(.*)\s\((\d{4})\)\s*$',t.strip())
    if not m: return None,None
    n=m.group(1); n=re.sub(r'\s*\(.*?\)$','',n)
    a=re.match(r'^(.*), (The|A|An|Il|La|Le|Les|L\')$',n)
    if a: n=f"{a.group(2)} {a.group(1)}"
    return n,int(m.group(2))
rows=[]
for _,r in item.iterrows():
    n,y=clean(r[1]); rows.append((r[0],n,prim(r),'English',y))
ml_items=pd.DataFrame(rows,columns=['movie_id','movie_title','genre','language','release_year']).dropna()
ml_items=ml_items.drop_duplicates('movie_title',keep=False)
data=pd.read_csv('/data/ml-100k/u.data',sep='\t',names=['ml_user','movie_id','rating','ts'])
data=data[data.movie_id.isin(ml_items.movie_id)]
pop=data.movie_id.value_counts(); keep=pop.index[:150]   # ~300 well-rated titles so CF has overlap
data=data[data.movie_id.isin(keep)]
ml_users=rng.choice(data.ml_user.unique(),500,replace=False)
umap=dict(zip(ml_users,rng.permutation(np.arange(1,501))))
data=data[data.ml_user.isin(ml_users)]
N_ML=1800
s=data.sample(N_ML,random_state=28).merge(ml_items,on='movie_id')
s['user_id']=s.ml_user.map(umap)
# map 1997-98 timestamps linearly into the project window (keeps relative order)
t=(s.ts-data.ts.min())/(data.ts.max()-data.ts.min())
s['timestamp']=START+(END-START)*t
s['rating']=s.rating.astype(float)
ml=s[['user_id','movie_title','genre','language','release_year','rating','timestamp']]
# ---- synthetic Indian ratings via latent factors (taste-driven, no demographics) ----
langs=ind.language.unique(); genres=sorted(set(ind.genre))
U=np.arange(1,501)
lang_aff={u:dict(zip(langs,rng.normal(0,.6,len(langs)))) for u in U}
for u in U:  # each user has 1-2 home languages, but nobody is locked in
    for l in rng.choice(langs[:5],rng.integers(1,3),replace=False): lang_aff[u][l]+=0.8
gen_aff={u:dict(zip(genres,rng.normal(0,.5,len(genres)))) for u in U}
ubias=dict(zip(U,rng.normal(0,.35,500)))
k=8; P=rng.normal(0,.5,(501,k)); Q=rng.normal(0,.5,(len(ind),k))
ibias=rng.normal(0.25,.35,len(ind))
w=np.where(ind.release_year>=2015,1.3,1.0)*np.exp(ibias)   # mild popularity skew
N_SYN=4500-N_ML; seen=set(); syn=[]
while len(syn)<N_SYN:
    u=int(rng.integers(1,501))
    lp=np.array([np.exp(lang_aff[u][l]) for l in ind.language])
    p=w*lp; p/=p.sum(); i=int(rng.choice(len(ind),p=p))
    if (u,i) in seen: continue
    seen.add((u,i)); m=ind.iloc[i]
    r=3.0+ubias[u]+ibias[i]+0.5*lang_aff[u][m.language]+0.6*gen_aff[u][m.genre]+P[u]@Q[i]+rng.normal(0,.45)
    r=float(np.clip(np.round(r*2)/2,1,5))
    syn.append((u,m.movie_title,m.genre,m.language,int(m.release_year),r,START+(END-START)*rng.random()))
syn=pd.DataFrame(syn,columns=ml.columns)
# ---- original 500 rows: keep as provided, add a time-of-day ----
o=orig.copy(); o['timestamp']=pd.to_datetime(o.timestamp)+pd.to_timedelta(rng.integers(8*3600,23*3600,len(o)),unit='s')
o['source']='original'; ml=ml.assign(source='movielens_100k'); syn=syn.assign(source='synthetic_indian')
df=pd.concat([o,ml,syn],ignore_index=True)
# ---- anomalies (~3%) ----
an=[]
allcat=pd.concat([ind,ml_items.drop(columns='movie_id')])
# 1) shilling bots: 4 new accounts, burst-rating many titles 5.0 within minutes
for b in range(4):
    uid=501+b; t0=START+(END-START)*rng.random()
    for j,ii in enumerate(rng.choice(len(ind),20,replace=False)):
        m=ind.iloc[ii]; an.append((uid,m.movie_title,m.genre,m.language,int(m.release_year),5.0,t0+pd.Timedelta(seconds=int(15*j+rng.integers(0,10))),'injected','shilling_bot_burst'))
# 2) review-bombing: 1.0 ratings on top-rated titles in a 2-hour window
t0=START+(END-START)*rng.random()
for j in range(40):
    m=ind[ind.movie_title.isin(['RRR','KGF: Chapter 2','Kantara','Jai Bhim'])].sample(1,random_state=j).iloc[0]
    an.append((int(rng.integers(1,501)),m.movie_title,m.genre,m.language,int(m.release_year),1.0,t0+pd.Timedelta(seconds=int(rng.integers(0,7200))),'injected','review_bombing'))
# 3) duplicate submissions: same user+movie re-posted seconds later with a contradicting rating
for _,r in syn.sample(30,random_state=1).iterrows():
    an.append((r.user_id,r.movie_title,r.genre,r.language,r.release_year,float(6-r.rating) if r.rating!=3 else 1.0,r.timestamp+pd.Timedelta(seconds=int(rng.integers(5,60))),'injected','duplicate_conflicting_rating'))
an=pd.DataFrame(an,columns=['user_id','movie_title','genre','language','release_year','rating','timestamp','source','anomaly_type'])
df['anomaly_type']='none'
df=pd.concat([df,an],ignore_index=True)
# trim normal synthetic rows so total = 5000
extra=len(df)-5000
drop=df[(df.source=='synthetic_indian')&(df.anomaly_type=='none')].sample(extra,random_state=2).index
df=df.drop(drop)
# sequential, unique timestamps
df=df.sort_values('timestamp',kind='stable').reset_index(drop=True)
ts=df.timestamp.dt.floor('s').values.astype('datetime64[s]').astype(np.int64)
for i in range(1,len(ts)):
    if ts[i]<=ts[i-1]: ts[i]=ts[i-1]+1
df['timestamp']=pd.to_datetime(ts,unit='s').strftime('%Y-%m-%d %H:%M:%S')
df['user_id']=df.user_id.astype(int); df['release_year']=df.release_year.astype(int); df['rating']=df.rating.astype(float).round(1)
cols=['user_id','movie_title','genre','language','release_year','rating','timestamp']
df[cols].to_csv('/data/movie_project/dataset.csv',index=False)
df.insert(0,'row_id',range(1,len(df)+1))
df[['row_id']+cols+['source','anomaly_type']].assign(is_anomaly=lambda d:(d.anomaly_type!='none').astype(int)).to_csv('/data/movie_project/dataset_labeled.csv',index=False)
print(len(df),df.source.value_counts().to_dict(),df.anomaly_type.value_counts().to_dict())
print(df.language.value_counts().to_dict()); print(df.genre.value_counts().to_dict()); print(df.rating.describe().round(2).to_dict())
print(df.user_id.nunique(),df.movie_title.nunique(),df.timestamp.is_monotonic_increasing,df.timestamp.is_unique,df.timestamp.iloc[[0,-1]].tolist())
