import numpy as np, re
def load(fn):
    txt=open(fn).read().replace('\\\n',' ')
    ins=[];outs=[];gates=[]
    lines=txt.split('\n'); i=0
    while i<len(lines):
        l=lines[i].strip()
        if l.startswith('.inputs'): ins=l.split()[1:]
        elif l.startswith('.outputs'): outs=l.split()[1:]
        elif l.startswith('.names'):
            sig=l.split()[1:]; cov=[]; i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].startswith('.'):
                cov.append(lines[i].split()); i+=1
            gates.append((sig[:-1],sig[-1],cov)); continue
        i+=1
    return ins,outs,gates
def sim(net, inputs):
    ins,outs,gates=net; v=dict(inputs); T=len(next(iter(inputs.values())))
    v['$false']=np.zeros(T,bool); v['$true']=np.ones(T,bool); v['$undef']=np.zeros(T,bool)
    tog=0
    pending=list(gates)
    while pending:
        rest=[]
        for (a,o,cov) in pending:
            if o in ('$false','$true','$undef'): continue
            if not all(x in v for x in a): rest.append((a,o,cov)); continue
            if len(a)==0:
                r=np.full(T, bool(cov and cov[0][0]=='1'))
            else:
                r=np.zeros(T,bool)
                for row in cov:
                    pat,val=(row[0],row[1]) if len(row)==2 else (row[0],'1')
                    t=np.ones(T,bool)
                    for ch,x in zip(pat,a):
                        if ch=='1': t&=v[x]
                        elif ch=='0': t&=~v[x]
                    r|=t
            v[o]=r; tog+=int(np.count_nonzero(r[1:]!=r[:-1]))
        if len(rest)==len(pending): raise RuntimeError("loop")
        pending=rest
    return v,tog
