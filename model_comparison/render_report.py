"""Render a compact measured-results report from verified team predictions."""
from pathlib import Path
import sys,json,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
wd=Path(__file__).resolve().parent
sys.path.insert(0,str(wd))
from evaluate import load,matches
root=wd/'predictions'; out=wd/'analysis';out.mkdir(exist_ok=True)
fig=out/'figures';fig.mkdir(exist_ok=True)
g=load(wd/'gold.jsonl');ids=sorted(g)
info=json.load((root/'metrics.json').open())
labels={'dino':'Grounding DINO tiny','owl':'OWLv2 base','qwen':'Qwen3-VL-2B-Instruct'}
ns=list(labels)
rng=np.random.default_rng(20261120)
ds=rng.integers(0,len(ids),size=(10000,len(ids)))
rep={}; counts={}
def micro(x):
    tp,fp,fn=x.T
    return 2*tp/np.maximum(1,2*tp+fp+fn)
for n in ns:
    pr=load(root/f'{n}.jsonl')
    t=np.array([matches(g[i],pr[i],0.5) for i in ids])
    counts[n]=np.column_stack([t,[len(pr[i])-m for i,m in zip(ids,t)],[len(g[i])-m for i,m in zip(ids,t)]])
    rep[n]=micro(counts[n][ds].sum(axis=1))
ranking=sorted(ns,key=lambda n:(-info[n]['groups']['all']['0.5']['f1'],info[n]['seconds']))
names=[labels[n] for n in ranking]
summary=['# Сравнение на командном golden set','',
         '400 фотографий, 193 эталонные рамки, 209 пустых кадров. Каждая конфигурация обработала весь набор. Оценка относится к указанным настройкам моделей.','',
         '| Модель | F1@0,5 | P@0,5 | R@0,5 | TP | FP | FN | F1@0,75 | FP на пустых | Пустых с FP | Время, с |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for n in ranking:
    r=info[n];a=r['groups']['all']['0.5'];b=r['groups']['all']['0.75']
    values=[labels[n],*(f'{a[k]:.4f}' for k in ('f1','precision','recall')),*(str(a[k]) for k in ('tp','fp','fn')),f"{b['f1']:.4f}",str(a['fp_on_empty']),str(a['empty_with_fp']),f"{r['seconds']:.1f}"]
    summary.append('| '+' | '.join(values)+' |')
summary+=['',f'Лучший object F1@IoU 0,5 у {names[0]}. Время измерено на одном A100 и суммирует успешную обработку 400 снимков.','',
          '## Результаты по группам источников','',
          '| Группа | Фото | Эталонных рамок | '+' | '.join(names)+' |',
          '|---|---:|---:|---:|---:|---:|']
for key, grp in info[ns[0]]['groups'].items():
    if key=='all': continue
    a=grp['0.5']; total=a['fn']+a['tp']
    cells=[]
    for n in ranking:
        r=info[n]['groups'][key]['0.5']
        cells.append(f"{r['f1']:.4f}" if total else f"{r['fp']} FP")
    summary.append('| '+' | '.join([key,str(a['n']),str(total),*cells])+' |')
summary+=['','В группах с эталонными объектами приведён F1@0,5. Для групп без эталонных объектов приведено число ложных рамок.']
(out/'analysis-report.md').write_text('\n'.join(summary)+'\n')
st=['# Неопределённость сравнения','',
    'Для каждого чекпойнта выполнен один прогон. Интервалы отражают повторную выборку фотографий из командного набора. В 10 000 повторов бутстрэпа по 400 кадров с возвращением каждая модель получала одни и те же индексы. В каждом повторе заново суммировались TP, FP и FN. Seed 20261120.','',
    '| Модель | F1@0,5 | 95% процентильный интервал |','|---|---:|---:|']
for n in ranking:
    lo,hi=np.quantile(rep[n],[.025,.975]);st.append(f"| {labels[n]} | {info[n]['groups']['all']['0.5']['f1']:.4f} | {lo:.4f}–{hi:.4f} |")
st+=['','| Разность F1@0,5 | Наблюдаемая разность | 95% парный интервал |','|---|---:|---:|']
for a,b in itertools.combinations(ranking,2):
    lo,hi=np.quantile(rep[a]-rep[b],[.025,.975]); gap=info[a]['groups']['all']['0.5']['f1']-info[b]['groups']['all']['0.5']['f1']
    st.append(f'| {labels[a]} минус {labels[b]} | {gap:.4f} | {lo:.4f}…{hi:.4f} |')
st+=['','Парные интервалы описательные, без поправки на множественные сравнения. Выбор конфигурации основан на наблюдаемом F1@0,5 согласно условиям Контеста. В командном наборе всего два кадра с несколькими объектами.']
(out/'stats-appendix.md').write_text('\n'.join(st)+'\n')
for kind in ['f1','errors']:
    f,ax=plt.subplots(figsize=(7.1,3.3),layout='constrained');y=np.arange(3)
    if kind=='f1':
        for k,color,offset in [('0.5','#276D79',-.15),('0.75','#CC7952',.15)]:
            vals=[info[n]['groups']['all'][k]['f1'] for n in ranking]
            ax.barh(y+offset,vals,height=.27,color=color,label='IoU '+k);ax.set_xlim(0,1.06)
            for j,v in enumerate(vals): ax.text(v+.012,j+offset,f'{v:.3f}',va='center',fontsize=8)
        ax.set_xlabel('Object F1');ax.set_title('Localization performance on 400 team photographs')
    else:
        for k,color,offset in [('fp','#BC643D',-.15),('fn','#526F9C',.15)]:
            vals=[info[n]['groups']['all']['0.5'][k] for n in ranking]
            ax.barh(y+offset,vals,height=.27,color=color,label=k.upper())
            for j,v in enumerate(vals): ax.text(v+.5,j+offset,str(v),va='center',fontsize=8)
        ax.margins(x=.16);ax.set_xlabel('Number of boxes');ax.set_title('Extra predictions and missed objects at IoU 0.5')
    ax.set_yticks(y,names);ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,frameon=False);ax.spines[['top','right']].set_visible(False);f.savefig(fig/f'{kind}.pdf');plt.close(f)
(out/'figure-catalog.md').write_text('# Графики по metrics.json\n\n- [Качество локализации](figures/f1.pdf). Сопоставляет F1 при двух порогах IoU по всем 400 фото. Разница между порогами показывает чувствительность к точности границ.\n- [Ошибки](figures/errors.pdf). Показывает число лишних рамок и пропусков при IoU 0,5 для каждой конфигурации. Пустые кадры входят в подсчёт FP.\n')
