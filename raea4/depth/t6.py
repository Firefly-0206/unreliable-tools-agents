import json
R=json.load(open("results_D6.json"))
pols=[p for p in R["RETRY"]["pooled"]["all_L1-4"]]
for f in ["RETRY","SHARED"]:
    best_viol=[];mx=0;rat=[]
    for k,c in R[f]["cells"].items():
        L=int(k.split("|")[1])
        dep=c["Fitted dependence (keep all)"]["regret"]
        others={p:c[p]["regret"] for p in c if p not in("Fitted dependence (keep all)",)}
        b=min(others,key=others.get)
        if L>=2:
            mx=max(mx,dep)
            if others[b]<dep: best_viol.append((k,b,round(others[b],3),round(dep,3)))
            rat.append(others[b]/max(dep,1e-9))
    print(f,"max dep cell L>=2",round(mx,3),"viol",best_viol,"ratio min/max",round(min(rat),1),round(max(rat),1))
    for lp,d in R[f]["by_logger"].items():
        dep=d["Fitted dependence (keep all)"]["regret"]; alt=min(v["regret"] for p,v in d.items() if p not in("Fitted dependence (keep all)","Fitted dependence + forbid") )
        print("  ",lp,round(dep,2),"best alt (excl dep)",round(alt,2),[p for p,v in d.items() if abs(v['regret']-alt)<1e-9])
    print("  maxdep all cells",max(c["Fitted dependence (keep all)"]["regret"] for c in R[f]["cells"].values()))
