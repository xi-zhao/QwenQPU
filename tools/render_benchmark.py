"""Render public aggregate evidence, without raw model logs or model calls."""
from pathlib import Path
import os
import collections
import csv
import json
import math

ROOT=Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"]=str(ROOT/"review/cache/matplotlib")
os.environ["XDG_CACHE_HOME"]=str(ROOT/"review/cache")
os.environ["TMPDIR"]=str(ROOT/"review/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True,exist_ok=True)
Path(os.environ["TMPDIR"]).mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BG,INK,MUTED,GRID="#0b1219","#edf3f5","#92aab7","#233744"
COLORS={"plain":"#65b8ed","enhanced":"#63e5d3"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"text.color":INK,
    "axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":INK,"axes.facecolor":BG,
    "figure.facecolor":BG,"axes.edgecolor":GRID,"svg.fonttype":"path","axes.spines.top":False,
    "axes.spines.right":False,"savefig.facecolor":BG})
rows=list(csv.DictReader((ROOT/"evidence/public/computation-pairs.csv").open()))
assert len(rows)==40 and {r["arm"] for r in rows}=={"plain","enhanced"}
qs=sorted({r["question_id"] for r in rows})
by={(r["question_id"],r["arm"]):r for r in rows}
wins=collections.Counter()
for q in qs:
    p,e=(int(by[q,a]["posthoc_consistent_json_correct"]) for a in ["plain","enhanced"])
    wins["both_correct" if p and e else "both_wrong" if not p and not e else "enhanced_only" if e else "plain_only"]+=1
assert len(qs)==20 and wins=={"both_correct":7,"plain_only":1,"enhanced_only":12}


def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h),min(1,c+h)


fig=plt.figure(figsize=(12,6.2))
gs=fig.add_gridspec(1,2,width_ratios=[1.25,1],left=.21,right=.97,bottom=.28,top=.76,wspace=.35)
ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1])
metrics=[("joint_correct","Frozen parser\nknown false negatives"),
         ("posthoc_consistent_json_correct","Identical-JSON diagnostic\npost hoc"),
         ("posthoc_naturally_finished_and_correct","Diagnostic + natural end\npost hoc")]
for i,(key,name) in enumerate(metrics):
    for j,a in enumerate(["plain","enhanced"]):
        k=sum(int(r[key]) for r in rows if r["arm"]==a);p=k/20;lo,hi=wilson(k,20);y=2-i+(j-.5)*.15
        ax.errorbar(p,y,xerr=[[p-lo],[hi-p]],fmt="o",color=COLORS[a],capsize=3,ms=6,lw=1.2)
        ax.annotate(f"{k}/20",(p,y),xytext=(4,9 if j else -15),textcoords="offset points",fontsize=9,color=COLORS[a])
ax.set(yticks=[2,1,0],yticklabels=[x[1] for x in metrics],xlim=(-.015,1.055),ylim=(-.4,2.4),
       xticks=[0,.25,.5,.75,1],xticklabels=["0%","25%","50%","75%","100%"],xlabel="Labelled score · Wilson 95% intervals")
ax.grid(axis="x",color=GRID,lw=.6);ax.set_axisbelow(True)
bx.text(.0,.95,"Paired diagnostic outcomes",transform=bx.transAxes,fontsize=11,va="top")
for i,q in enumerate(qs):
    for j,a in enumerate(["plain","enhanced"]):
        correct=int(by[q,a]["posthoc_consistent_json_correct"])
        bx.scatter(i,1-j,marker="s",s=91,color=COLORS[a] if correct else GRID)
for x in [3.5,7.5,11.5,15.5]:bx.axvline(x,color=GRID,lw=.6)
bx.set(xlim=(-1,20),ylim=(-.8,1.7),yticks=[1,0],yticklabels=["Qwen","+ network"],
       xticks=[1.5,5.5,9.5,13.5,17.5],xticklabels=["1","3","8","12","16"],xlabel="Qubits · 4 paired tasks per width")
bx.text(.0,.07,f"{wins['enhanced_only']} enhanced-only wins / {wins['plain_only']} plain-only win\n{wins['both_correct']} both correct · {wins['both_wrong']} both wrong",transform=bx.transAxes,fontsize=9,color=MUTED,linespacing=1.7,va="bottom")
for side in ["left","bottom"]:bx.spines[side].set_visible(False)
bx.tick_params(length=0)
fig.text(.045,.925,"SAME WEIGHTS. EXPLICIT COMPUTATION.",fontsize=22,weight="bold")
fig.text(.045,.865,"20 supplied-circuit tasks · 1–16 qubits · 16,384 generated-token cap · 4,096 simulated draws",fontsize=11,color=MUTED)
fig.legend([Line2D([],[],marker="o",ls="",color=COLORS[a]) for a in ["plain","enhanced"]],
           ["Qwen","Qwen + fixed quantum network"],loc="upper left",bbox_to_anchor=(.205,.835),ncol=2,frameon=False)
fig.text(.045,.135,"The frozen parser rejects identical repeated JSON. Diagnostics are post hoc, retain all 20 pairs, and are not a full semantic audit.",fontsize=9,color=MUTED)
fig.text(.045,.09,"Greedy decoding · same-question re-evaluation · 5–161 gate operations · ideal classical simulation, not physical-QPU execution.",fontsize=9,color=MUTED)
fig.text(.045,.045,"Planning is reported separately: original joint score 1/20 in each arm. No quantum-advantage or decoder-fusion claim.",fontsize=9,color=MUTED)
for ext in ["png","svg"]:fig.savefig(ROOT/"assets"/("computation-evidence."+ext),dpi=180)
plt.close(fig)
print(json.dumps({"given_circuit_conditions":len(rows),"posthoc_diagnostic":[8,19],"frozen_scores_preserved":True,"new_model_calls":0}))
