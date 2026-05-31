# Core feature engineering (the 90.3 feature set, calibration removed) + heavy-model ensemble.
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")
SEED=42

_B32="0123456789bcdefghjkmnpqrstuvwxyz"; _DEC={c:i for i,c in enumerate(_B32)}
def gh_decode(gh):
    if not isinstance(gh,str) or not gh: return (np.nan,np.nan)
    lat,lon,is_lon=[-90.,90.],[-180.,180.],True
    for ch in gh.lower():
        cd=_DEC.get(ch)
        if cd is None: continue
        for mask in (16,8,4,2,1):
            if is_lon: mid=(lon[0]+lon[1])/2; lon[0 if cd&mask else 1]=mid
            else:      mid=(lat[0]+lat[1])/2; lat[0 if cd&mask else 1]=mid
            is_lon=not is_lon
    return ((lat[0]+lat[1])/2,(lon[0]+lon[1])/2)
def mod_of(t): h,m=str(t).split(":"); return int(h)*60+int(m)

def kfold_te(ktr,kte,y,folds,sm=10.0):
    gm=y.mean(); oof=pd.Series(np.nan,index=ktr.index)
    for tr,va in folds:
        s=pd.DataFrame({"k":ktr.iloc[tr],"y":y.iloc[tr]}).groupby("k")["y"].agg(["mean","count"])
        smv=(s["mean"]*s["count"]+gm*sm)/(s["count"]+sm); oof.iloc[va]=ktr.iloc[va].map(smv).values
    oof=oof.fillna(gm)
    f=pd.DataFrame({"k":ktr,"y":y}).groupby("k")["y"].agg(["mean","count"])
    smf=(f["mean"]*f["count"]+gm*sm)/(f["count"]+sm)
    return oof.values, kte.map(smf).fillna(gm).values
def mkkey(df,cols):
    s=df[cols[0]].astype(str)
    for c in cols[1:]: s=s+"|"+df[c].astype(str)
    return s

def build(train,test,y,folds):
    def base(df):
        o=pd.DataFrame(index=df.index); m=df["timestamp"].map(mod_of)
        o["mod"]=m; o["hour"]=m//60; o["minute"]=m%60
        a=2*np.pi*m/1440.; o["mod_sin"],o["mod_cos"]=np.sin(a),np.cos(a)
        o["day"]=pd.to_numeric(df["day"],errors="coerce")
        gh=df["geohash"].astype(str); c={g:gh_decode(g) for g in gh.dropna().unique()}
        o["gh_lat"]=gh.map(lambda g:c[g][0]); o["gh_lon"]=gh.map(lambda g:c[g][1])
        for p in (4,5): o[f"gh_pre{p}"]=gh.str.slice(0,p).astype("category").cat.codes
        o["gh_code"]=gh.astype("category").cat.codes
        o["NumberofLanes"]=pd.to_numeric(df["NumberofLanes"],errors="coerce")
        o["Temperature"]=pd.to_numeric(df["Temperature"],errors="coerce")
        o["LargeVehicles"]=df["LargeVehicles"].astype(str).str.strip().str.lower().map({"allowed":1,"not allowed":0}).fillna(-1)
        o["Landmarks"]=df["Landmarks"].astype(str).str.strip().str.lower().map({"yes":1,"no":0}).fillna(-1)
        o["RoadType"]=df["RoadType"].astype("category").cat.codes
        o["Weather"]=df["Weather"].astype("category").cat.codes
        return o,m
    Xtr,mtr=base(train); Xte,mte=base(test)
    Xtr=Xtr.reset_index(drop=True); Xte=Xte.reset_index(drop=True); yr=y.reset_index(drop=True)
    rt,re=train.copy(),test.copy(); rt["mod"],re["mod"]=mtr.values,mte.values; rt["hour"],re["hour"]=mtr.values//60,mte.values//60
    for k in (4,5): rt[f"gh{k}"]=train.geohash.astype(str).str.slice(0,k); re[f"gh{k}"]=test.geohash.astype(str).str.slice(0,k)
    specs=[["geohash"],["geohash","hour"],["geohash","mod"],["gh4"],["gh4","hour"],["RoadType"],["Weather"],
           ["geohash","RoadType"],["gh5","hour"],["RoadType","hour"],["Weather","hour"]]
    for cols in specs:
        otr,ote=kfold_te(mkkey(rt,cols).reset_index(drop=True),mkkey(re,cols).reset_index(drop=True),yr,folds)
        Xtr["te_"+"_".join(cols)]=otr; Xte["te_"+"_".join(cols)]=ote
    d49=train[train.day==49]; gma=train.groupby("geohash")["demand"].mean(); gm=y.mean()
    rec=d49.groupby("geohash")["demand"].mean()
    rf=lambda df: df.geohash.map(rec).fillna(df.geohash.map(gma)).fillna(gm).values
    Xtr["recent_gh"]=rf(train); Xte["recent_gh"]=rf(test)
    d48=train[train.day==48].copy(); d48["mod"]=d48.timestamp.map(mod_of)
    prof=d48.groupby(["geohash","mod"])["demand"].mean(); piv=prof.unstack("mod").sort_index(axis=1); pT=piv.T.sort_index()
    rs=pT.rolling(5,center=True,min_periods=1).sum(); rc=pT.notna().rolling(5,center=True,min_periods=1).sum()
    sx=((rs-pT.fillna(0))/(rc-pT.notna()).replace(0,np.nan)).T.stack().to_dict()
    si=pT.rolling(5,center=True,min_periods=1).mean().T.stack().to_dict(); pd_=prof.to_dict()
    lk=lambda d,gs,ms: np.array([d.get((g,m),np.nan) for g,m in zip(gs,ms)])
    gt,ge=train.geohash.values,test.geohash.values
    Xtr["d48_smooth"]=lk(sx,gt,mtr.values); Xte["d48_smooth"]=lk(si,ge,mte.values)
    for off,nm in [(-15,"prev"),(15,"next"),(-30,"prev2"),(30,"next2")]:
        Xtr[f"d48_{nm}"]=lk(pd_,gt,mtr.values+off); Xte[f"d48_{nm}"]=lk(pd_,ge,mte.values+off)
    Xtr["d48_trend"]=Xtr["d48_next"]-Xtr["d48_prev"]; Xte["d48_trend"]=Xte["d48_next"]-Xte["d48_prev"]
    # raw categoricals for native-categorical models (CatBoost/LightGBM/XGBoost)
    cat_cols=[]
    for c in ["geohash","gh4","gh5","RoadType","Weather","LargeVehicles","Landmarks"]:
        src_tr = (train.geohash.astype(str).str.slice(0,4) if c=="gh4" else
                  train.geohash.astype(str).str.slice(0,5) if c=="gh5" else train[c].astype(str))
        src_te = (test.geohash.astype(str).str.slice(0,4) if c=="gh4" else
                  test.geohash.astype(str).str.slice(0,5) if c=="gh5" else test[c].astype(str))
        Xtr["cat_"+c]=src_tr.values; Xte["cat_"+c]=src_te.values; cat_cols.append("cat_"+c)
    cols=sorted(set(Xtr.columns)&set(Xte.columns))
    return Xtr[cols], Xte[cols], cat_cols

if __name__=="__main__":
    tr=pd.read_csv("train.csv"); te=pd.read_csv("test.csv")
    y=pd.to_numeric(tr["demand"],errors="coerce")
    folds=list(KFold(5,shuffle=True,random_state=SEED).split(tr))
    Xtr,Xte,cat=build(tr,te,y,folds)
    print("features:",Xtr.shape[1],"| categorical:",cat)
    # verify HGBR path here (heavy libs added in the Colab notebook)
    from sklearn.ensemble import HistGradientBoostingRegressor
    num=[c for c in Xtr.columns if not c.startswith("cat_")]
    p=np.zeros(len(te))
    for tri,vai in folds:
        m=HistGradientBoostingRegressor(max_iter=1500,learning_rate=0.03,max_leaf_nodes=128,
            min_samples_leaf=30,l2_regularization=1.0,early_stopping=True,random_state=SEED)
        m.fit(Xtr[num].iloc[tri],y.iloc[tri]); p+=m.predict(Xte[num])/5
    p=np.clip(p,y.min(),y.max())
    print("HGBR path OK, pred mean",round(p.mean(),4))
