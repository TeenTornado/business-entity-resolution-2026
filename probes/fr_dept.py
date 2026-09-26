"""France: does a region vs departement mismatch depress acceptance of obvious copies?"""
import sys; import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','code','business_entity_resolution','src'))
import pandas as pd, numpy as np, glob, collections
from anyascii import anyascii
from io_utils import read_p1, read_p23
W=sys.argv[1]; SC=sys.argv[2]; FEAT=sys.argv[3]
P1=read_p1(W,'test',['country','n_core','a_nums','raw_addr'])
fr=np.flatnonzero(P1.country.values=='France')
b=pd.concat([pd.read_parquet(f,columns=['i1','i2']) for f in sorted(glob.glob(f'{FEAT}/block*.parquet'))],ignore_index=True)
b['p']=np.load(SC); b=b[np.isin(b.i1.values,fr)].reset_index(drop=True)
R=read_p23(W,'test',['n_core','a_nums','raw_addr'])
norm=lambda s: anyascii(s).lower().replace('-',' ').strip()
comps=lambda s: [norm(c) for c in s.split(',') if c.strip()]
regions=collections.Counter(comps(a)[-1] for a in P1.raw_addr.values[fr] if comps(a))
regions={r for r,n in regions.items() if n>2000}
print('S1 regions:',sorted(regions))
def first(s):
    for t in s.split():
        if t.isdigit(): return t
    return ''
typ=[];strong=[]
for x,y in zip(b.i1.values,b.i2.values):
    cs=comps(R.raw_addr.values[y]); s1=comps(P1.raw_addr.values[x])
    t='none' if not cs else 'region' if any(c in regions for c in cs) else 'other'
    typ.append(t)
    strong.append(P1.n_core.values[x]==R.n_core.values[y] and first(P1.a_nums.values[x])!='' and first(P1.a_nums.values[x])==first(R.a_nums.values[y]))
b['typ']=typ; b['strong']=strong; b['acc']=b.p>=0.7375
print(b[b.strong].groupby('typ').agg(n=('acc','size'),acc=('acc','mean'),mean_p=('p','mean')).round(3).to_string())
oth=collections.Counter(comps(R.raw_addr.values[y])[-1] for y,t in zip(b.i2.values,typ) if t=='other')
print('top last components of non-region copies:',oth.most_common(15))
