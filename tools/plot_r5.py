"""Four publication figures from complete frozen r5 scores; no model inference."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS={'plain':'#758092','qwenqpu':'#8975E7'}
LABELS={'plain':'Qwen','qwenqpu':'QwenQPU'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--score',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.score.read_text());assert data['complete'] and len(data['rows'])==96
    rows=data['rows'];tasks=list(dict.fromkeys(r['id'] for r in rows));lookup={(r['id'],r['condition']):r for r in rows}
    a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':180})
    metadata=dict(score_sha256=hashlib.sha256(a.score.read_bytes()).hexdigest(),protocol=data['protocol'],
                  design_reference='https://github.com/divagr18/qwenputer/blob/main/docs/benchmarks-arith.png',figures={})
    def save(fig,name,caption):
        for ext in ['png','svg','pdf']:fig.savefig(a.out/f'{name}.{ext}',bbox_inches='tight',facecolor=fig.get_facecolor())
        metadata['figures'][name]=caption;plt.close(fig)

    groups=[('nonclifford_gates_12','12 gates'),('nonclifford_gates_18','18 gates'),('nonclifford_gates_24','24 gates'),
            ('nonclifford_primary','Non-Clifford\nprimary'),('clifford_control','Clifford\ncontrol')]
    fig,ax=plt.subplots(figsize=(13,6));bg='#101114';fg='#F2F2F4';muted='#B5B9C2';fig.patch.set_facecolor(bg);ax.set_facecolor(bg)
    for j,c in enumerate(['plain','qwenqpu']):
        vals=[100*data['subsets'][k]['summary'][c]['correct']/data['subsets'][k]['summary'][c]['scheduled'] for k,_ in groups]
        bars=ax.bar(np.arange(len(groups))+(j-.5)*.36,vals,.28,color=COLORS[c],label=LABELS[c],zorder=3)
        for bar,v,(k,_) in zip(bars,vals,groups):
            s=data['subsets'][k]['summary'][c]
            ax.text(bar.get_x()+bar.get_width()/2,v+2.5,f'{v:.1f}%',ha='center',color=fg,weight='bold')
            if v>15:ax.text(bar.get_x()+bar.get_width()/2,v/2,f'{s["correct"]}/{s["scheduled"]}',ha='center',color='white')
    ax.set_xticks(range(len(groups)),[label+f'\nn = {data["subsets"][k]["summary"]["plain"]["scheduled"]}' for k,label in groups])
    ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.tick_params(colors=muted);ax.set_ylabel('Tasks correct (%)',color=muted)
    ax.grid(axis='y',color='#292C32');ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_color('#292C32')
    ax.set_title('QwenQPU: circuit-probability calculation',loc='left',fontsize=18,color=fg,weight='bold',pad=22)
    legend=ax.legend(ncol=2,frameon=False,loc='upper right')
    for t in legend.get_texts():t.set_color(fg)
    fig.text(.125,.01,'48 paired tasks · 3/4/6 logical qubits · 16,384 total predictions · both event probabilities must pass',color=muted,fontsize=9)
    fig.subplots_adjust(bottom=.22);save(fig,'benchmark-main','Frozen r5. Main endpoint: 36 non-Clifford tasks; gate strata n=12 each. Controls n=12. Exact probabilities, independent Qiskit reference, absolute tolerance 5e-5. Native QPU is the fixed added-layer simulator. The primary group overlaps its three gate strata.')

    fig,axs=plt.subplots(1,2,figsize=(13,7),gridspec_kw={'width_ratios':[2,3]})
    from matplotlib.colors import ListedColormap
    outcome=np.array([[int(lookup[(t,c)]['passed']) for c in ['plain','qwenqpu']] for t in tasks])
    axs[0].imshow(outcome,cmap=ListedColormap(['#E6E9EE','#8975E7']),vmin=0,vmax=1,aspect='auto')
    axs[0].set_xticks([0,1],['Qwen','QwenQPU']);axs[0].set_yticks(np.arange(0,48,4),[str(i+1) for i in range(0,48,4)])
    axs[0].set_ylabel('Task index in frozen order');axs[0].set_title('All 48 paired outcomes')
    groups2=[('nonclifford_primary','Non-Clifford'),('clifford_control','Clifford')]
    for j,c in enumerate(['plain','qwenqpu']):
        ss=[data['subsets'][k]['summary'][c] for k,_ in groups2];xs=np.arange(2)+(j-.5)*.34
        vals=np.array([100*s['correct']/s['scheduled'] for s in ss]);ci=np.array([s['wilson_95'] for s in ss])*100
        axs[1].bar(xs,vals,.28,color=COLORS[c],label=LABELS[c]);axs[1].errorbar(xs,vals,yerr=[vals-ci[:,0],ci[:,1]-vals],fmt='none',color='#333945',capsize=5)
    axs[1].set_xticks([0,1],['Non-Clifford (n=36)','Clifford (n=12)']);axs[1].set_ylim(0,115);axs[1].set_ylabel('Tasks correct (%)');axs[1].legend(frameon=False)
    paired=data['subsets']['nonclifford_primary']['paired']
    axs[1].set_title(f'Primary pairs: QPU-only {paired["qwenqpu_only"]}, Qwen-only {paired["plain_only"]}',fontsize=11)
    fig.text(.02,.01,'Grey = fail; purple = pass. All 96 conditions retained. Error bars: Wilson 95% intervals. Paired interval and exact test in score JSON.',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);save(fig,'benchmark-paired','Paired end-to-end outcomes, including format, time and budget failures. No core-only substitution. Wilson intervals describe the predefined task sample.')

    fig,ax=plt.subplots(figsize=(13,5.5))
    for j,c in enumerate(['plain','qwenqpu']):
        for i,t in enumerate(tasks):
            errors=list(lookup[(t,c)].get('absolute_errors',{}).values())
            if len(errors)==2 and all(e is not None for e in errors):ax.scatter(i+(j-.5)*.25,max(max(errors),1e-9),s=25,color=COLORS[c],marker='o' if c=='plain' else 's',zorder=3)
            else:ax.text(i+(j-.5)*.25,1.5,'×',ha='center',color=COLORS[c],fontsize=9)
    ax.axhline(5e-5,color='#B16E2E',ls='--');ax.set_yscale('log');ax.set_ylim(5e-10,4);ax.set_xticks(range(0,48,4),range(1,49,4));ax.set_xlabel('Task index in frozen order');ax.set_ylabel('Maximum absolute probability error')
    ax.set_title('Final numerical answers against Qiskit',loc='left',weight='bold',fontsize=16)
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color=COLORS['plain'],label='Qwen'),Line2D([],[],marker='s',ls='',color=COLORS['qwenqpu'],label='QwenQPU'),Line2D([],[],ls='--',color='#B16E2E',label='5e-5 tolerance')],ncol=3,frameon=False)
    fig.text(.02,.01,'× = no valid numerical answer. Errors below 1e-9 are displayed at 1e-9. Full values and failure reasons are preserved.',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);save(fig,'benchmark-errors','Two event probabilities per task; maximum absolute error shown. Faithfulness is separately required even when numerical outputs happen to match.')

    fig,axs=plt.subplots(1,2,figsize=(13,5.5))
    for c in ['plain','qwenqpu']:
        rr=[r for r in rows if r['condition']==c]
        for passed,marker in [(True,'o'),(False,'x')]:
            group=[r for r in rr if r['passed']==passed]
            axs[0].scatter([r['total_predictions'] for r in group],[r['seconds'] for r in group],color=COLORS[c],marker=marker,s=30,label=LABELS[c]+(' pass' if passed else ' fail'),alpha=.75)
        xs=[12,18,24];median=[np.median([r['seconds'] for r in rr if r['family']=='nonclifford' and r['gate_count']==g]) for g in xs]
        axs[1].plot(xs,median,marker='o',color=COLORS[c],label=LABELS[c])
    axs[0].axvline(16384,color='#B16E2E',ls='--',lw=1);axs[0].set_xlabel('Total forward predictions');axs[0].set_ylabel('End-to-end seconds');axs[0].legend(frameon=False,fontsize=9)
    axs[1].set_xticks([12,18,24]);axs[1].set_xlabel('Sequential gate count');axs[1].set_ylabel('Median seconds, non-Clifford tasks (n=12 per point)');axs[1].legend(frameon=False)
    for ax in axs:ax.grid(alpha=.15)
    fig.suptitle('Measured execution cost',x=.02,ha='left',weight='bold',fontsize=16)
    fig.text(.02,.01,'Existing A100 · model loading excluded · language tokens and native QPU predictions recorded separately · CUDA synchronized',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.95]);save(fig,'benchmark-cost','Wall time includes decoding, native execution and progress I/O. Points include every condition; success and failure markers are distinct. Prediction count is not a hardware speed comparison.')
    (a.out/'figure-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(figures=list(metadata['figures']),score_sha256=metadata['score_sha256'])))


if __name__=='__main__':main()
