"""Post-hoc analysis of saved official-validation predictions only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT=Path("reports/final_evaluation"); OUT=Path("reports/error_analysis"); KEYS={"logreg":"Logistic Regression","mlp_original":"Original MLP","mlp_regularized":"Regularized MLP","lstm":"LSTM"}; N_BOOT=2000; THRESHOLD=.5
def metric(y,p):
 q=(p>=THRESHOLD).astype(int);tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
 return dict(accuracy=accuracy_score(y,q),precision=precision_score(y,q,zero_division=0),recall=recall_score(y,q,zero_division=0),f1=f1_score(y,q,zero_division=0),roc_auc=roc_auc_score(y,p),tp=tp,tn=tn,fp=fp,fn=fn,fpr=fp/(fp+tn),fnr=fn/(fn+tp),error_count=fp+fn,error_rate=(fp+fn)/len(y))
def load_frames(root=ROOT):
 fs={k:pd.read_csv(root/f"{k}_predictions.csv").sort_values("id").reset_index(drop=True) for k in KEYS}; ref=None
 for k,d in fs.items():
  if list(d.columns)!=["id","true_label","predicted_label","probability_ai"] or d.id.nunique()!=len(d) or not np.isfinite(d.probability_ai).all() or not d.probability_ai.between(0,1).all(): raise ValueError(f"Invalid prediction file: {k}")
  if not np.array_equal(d.predicted_label,(d.probability_ai>=.5).astype(int)): raise ValueError(f"Threshold mismatch: {k}")
  if ref is None: ref=d
  elif not np.array_equal(ref.id,d.id) or not np.array_equal(ref.true_label,d.true_label): raise ValueError("ID/label mismatch")
 return fs
def all_models_wrong(fs):
 """Return the ID-aligned cases misclassified by every final model."""
 y=fs["logreg"].true_label.to_numpy()
 result=fs["logreg"][["id","true_label","predicted_label"]].rename(columns={"predicted_label":"logreg_pred"})
 for k in ("mlp_original","mlp_regularized","lstm"):
  current=fs[k][["id","true_label","predicted_label"]].rename(columns={"true_label":f"{k}_label","predicted_label":f"{k}_pred"})
  result=result.merge(current,on="id",validate="one_to_one")
  if not np.array_equal(result.true_label.to_numpy(),result[f"{k}_label"].to_numpy()): raise ValueError("True-label mismatch while aligning all-model errors")
 wrong=np.ones(len(result),dtype=bool)
 for k in ("logreg","mlp_original","mlp_regularized","lstm"): wrong &= result[f"{k}_pred"].to_numpy()!=y
 return result.loc[wrong,["id","true_label","logreg_pred","mlp_original_pred","mlp_regularized_pred","lstm_pred"]].copy()
def bootstrap(fs,n=N_BOOT,seed=42):
 y=fs["logreg"].true_label.to_numpy(); ix=[np.flatnonzero(y==c) for c in (0,1)];rng=np.random.default_rng(seed); ks=list(fs); out=np.empty((n,4,3))
 for i in range(n):
  s=np.r_[rng.choice(ix[0],len(ix[0]),True),rng.choice(ix[1],len(ix[1]),True)]
  for j,k in enumerate(ks):
   m=metric(y[s],fs[k].probability_ai.to_numpy()[s]);out[i,j]=[m["accuracy"],m["f1"],m["roc_auc"]]
 rows=[];diff=[]
 for j,k in enumerate(ks):
  full=metric(y,fs[k].probability_ai.to_numpy())
  for z,nm in enumerate(["accuracy","f1","roc_auc"]): rows.append(dict(model=KEYS[k],key=k,metric=nm,estimate=full[nm],ci95_low=np.quantile(out[:,j,z],.025),ci95_high=np.quantile(out[:,j,z],.975)))
 for b in ("mlp_regularized","logreg"):
  for z,nm in ((1,"f1"),(2,"roc_auc")):
   v=out[:,ks.index("lstm"),z]-out[:,ks.index(b),z];diff.append(dict(comparison=f"LSTM - {KEYS[b]}",metric=nm,estimate=metric(y,fs["lstm"].probability_ai)[nm]-metric(y,fs[b].probability_ai)[nm],ci95_low=np.quantile(v,.025),ci95_high=np.quantile(v,.975),includes_zero=np.quantile(v,.025)<=0<=np.quantile(v,.975)))
 return pd.DataFrame(rows),pd.DataFrame(diff)
def save(fig,name,out): fig.tight_layout();fig.savefig(out/f"{name}.png",dpi=300);fig.savefig(out/f"{name}.pdf");plt.close(fig)
def mdframe(df):
 return "| " + " | ".join(df.columns) + " |\n|"+"|".join("---" for _ in df.columns)+"|\n"+"\n".join("| "+" | ".join(str(x) for x in row)+" |" for row in df.itertuples(index=False,name=None))
def repair(out=OUT):
 """Targeted stage-6 repair: no bootstrap rerun and no existing result rewrite."""
 fs=load_frames();fixed=all_models_wrong(fs)
 fixed.to_csv(out/"all_models_wrong_ids.csv",index=False)
 conf=pd.read_csv(out/"confidence_analysis.csv"); x=np.arange(len(conf));fig,ax=plt.subplots(figsize=(7,4));ax.bar(x-.18,conf.mean_confidence_correct,.36,label="Correct");ax.bar(x+.18,conf.mean_confidence_error,.36,label="Error");ax.set(xticks=x,xticklabels=conf.model,ylabel="Mean confidence",ylim=(0,1),title="Confidence of correct and erroneous predictions");ax.legend();save(fig,"confidence_correct_vs_wrong",out)
 fig,axes=plt.subplots(2,2,figsize=(7,5.5));bins=np.linspace(0,1,31)
 for ax,(k,d) in zip(axes.flat,fs.items()):
  ax.hist(d.loc[d.true_label==0,"probability_ai"],bins=bins,alpha=.55,label="True 0",density=True);ax.hist(d.loc[d.true_label==1,"probability_ai"],bins=bins,alpha=.55,label="True 1",density=True);ax.set(title=KEYS[k],xlim=(0,1),xlabel="P(AI)",ylabel="Density");ax.legend(fontsize=7)
 save(fig,"probability_distributions",out)
 fig,ax=plt.subplots(figsize=(7,4));ax.bar(conf.model,conf.high_confidence_error_share);ax.set(ylabel="Share of erroneous predictions",ylim=(0,max(.01,conf.high_confidence_error_share.max()*1.2)),title="High-confidence errors (confidence ≥ 0.9)");ax.tick_params(axis="x",rotation=15);save(fig,"high_confidence_errors",out)
 err=pd.read_csv(out/"error_summary.csv");length=pd.read_csv(out/"length_analysis.csv");dis=pd.read_csv(out/"model_disagreements.csv");ci=pd.read_csv(out/"bootstrap_model_ci.csv");dif=pd.read_csv(out/"bootstrap_pairwise_differences.csv");q=json.loads((out/"analysis_config.json").read_text())["length_quartile_bounds"]
 lines=["# Анализ ошибок и статистической неопределённости","","## 1. Цель анализа","Оценка структуры ошибок, связи качества с длиной и неопределённости фиксированных результатов без изменения моделей.","","## 2. Ошибки классификации","",mdframe(err[["model","tp","tn","fp","fn","fpr","fnr"]]),"","FP означают ошибочное отнесение человеческого текста к AI, FN — пропуск AI-текста.","","## 3. Длина текста","",f"Границы квартилей по словам: {q}; F1 по группам приведён в `length_analysis.csv`. Связь описательная и не устанавливает причинность.","","## 4. Уверенность","",mdframe(conf),"","Выходные вероятности не интерпретируются как идеально калиброванные вероятности истинного происхождения.","","## 5. Сравнение LSTM и regularized MLP","",mdframe(dis),"",f"Все четыре модели одновременно ошиблись на {len(fixed)} текстах.","","## 6. Bootstrap","",mdframe(ci[ci.metric.isin(["f1","roc_auc"])]),"",mdframe(dif),"","Интервалы ΔF1 включают ноль; интервалы ΔROC-AUC ноль не включают. Текстовый bootstrap может недооценивать неопределённость при кластерной зависимости по источнику.","","## 7. Итог","","LSTM имеет наибольший наблюдаемый F1, но bootstrap не даёт достаточных оснований утверждать о стабильном преимуществе по F1. ROC-AUC LSTM выше у рассмотренных сравнений в рамках использованного bootstrap."]
 (out/"error_analysis_summary.md").write_text("\n".join(lines),encoding="utf-8")
def run(out=OUT):
 out.mkdir(parents=True,exist_ok=True);fs=load_frames(); y=fs["logreg"].true_label.to_numpy(); saved={x["key"]:x for x in json.loads((ROOT/"metrics.json").read_text())}
 errors=[];conf=[]
 for k,d in fs.items():
  m=metric(y,d.probability_ai.to_numpy());
  if not np.isclose(m["f1"],saved[k]["f1_label_1"]): raise ValueError("Final metric mismatch")
  errors.append(dict(model=KEYS[k],key=k,**m)); ok=d.predicted_label.to_numpy()==y;c=np.maximum(d.probability_ai,1-d.probability_ai)
  conf.append(dict(model=KEYS[k],mean_confidence_correct=c[ok].mean(),mean_confidence_error=c[~ok].mean(),high_confidence_error_share=(c[~ok]>=.9).mean(),brier_score=brier_score_loss(y,d.probability_ai)))
 err=pd.DataFrame(errors);err.to_csv(out/"error_summary.csv",index=False);(out/"error_summary.json").write_text(err.to_json(orient="records",indent=2));pd.DataFrame(conf).to_csv(out/"confidence_analysis.csv",index=False)
 val=load_dataset("RussianNLP/coat","binary",split="validation").to_pandas(); base=val[["id","text"]].copy();base["char_length"]=base.text.str.len();base["word_length"]=base.text.str.findall(r"\S+").str.len();q=np.quantile(base.word_length,[.25,.5,.75]);base["length_group"]=np.select([base.word_length<=q[0],base.word_length<=q[1],base.word_length<=q[2]],["Q1_short","Q2","Q3"],default="Q4_long")
 lr=[]
 for k,d in fs.items():
  z=d.merge(base[["id","char_length","word_length","length_group"]],on="id")
  for g,x in z.groupby("length_group",sort=False): lr.append(dict(model=KEYS[k],key=k,length_group=g,examples=len(x),class_0=(x.true_label==0).sum(),class_1=(x.true_label==1).sum(),**metric(x.true_label.to_numpy(),x.probability_ai.to_numpy())))
 length=pd.DataFrame(lr);length.to_csv(out/"length_analysis.csv",index=False)
 merged=fs["logreg"].rename(columns={"predicted_label":"logreg_pred","probability_ai":"logreg_prob"})
 for k in ("mlp_regularized","lstm"): merged=merged.merge(fs[k][["id","predicted_label","probability_ai"]].rename(columns={"predicted_label":k+"_pred","probability_ai":k+"_prob"}),on="id")
 dis=[];ids=[]
 for a,b in (("lstm","mlp_regularized"),("lstm","logreg")):
  ap=merged[a+"_pred"] if a!="logreg" else merged.logreg_pred;bp=merged[b+"_pred"] if b!="logreg" else merged.logreg_pred;ac=ap==y;bc=bp==y;dis.append(dict(comparison=f"{KEYS[a]} vs {KEYS[b]}",both_correct=(ac&bc).sum(),both_wrong=(~ac&~bc).sum(),first_correct_second_wrong=(ac&~bc).sum(),second_correct_first_wrong=(~ac&bc).sum()));ids.append(merged.loc[ac^bc,["id","true_label"]].assign(comparison=f"{KEYS[a]} vs {KEYS[b]}"))
 pd.DataFrame(dis).to_csv(out/"model_disagreements.csv",index=False);pd.concat(ids).to_csv(out/"model_disagreement_ids.csv",index=False);all_models_wrong(fs).to_csv(out/"all_models_wrong_ids.csv",index=False)
 ci,diff=bootstrap(fs);ci.to_csv(out/"bootstrap_model_ci.csv",index=False);diff.to_csv(out/"bootstrap_pairwise_differences.csv",index=False)
 examples=[]
 for cat,mask in {"lstm_fp":(y==0)&(merged.lstm_pred==1),"lstm_fn":(y==1)&(merged.lstm_pred==0),"lstm_wrong_reg_correct":(merged.lstm_pred!=y)&(merged.mlp_regularized_pred==y),"reg_wrong_lstm_correct":(merged.mlp_regularized_pred!=y)&(merged.lstm_pred==y)}.items(): examples.append(merged.loc[mask].sort_values("id").head(5).assign(category=cat))
 ex=pd.concat(examples).merge(base[["id","text"]],on="id");ex["text_snippet"]=ex.text.str.slice(0,200);ex.drop(columns="text").to_csv(out/"error_examples.csv",index=False)
 fig,ax=plt.subplots();x=np.arange(4);ax.bar(x-.2,err.fp,.4,label="FP");ax.bar(x+.2,err.fn,.4,label="FN");ax.set(xticks=x,xticklabels=err.model,ylabel="Texts");ax.legend();save(fig,"fp_fn_comparison",out)
 fig,ax=plt.subplots();[ax.plot(g.length_group,g.f1,marker="o",label=m) for m,g in length.groupby("model")];ax.legend(fontsize=7);ax.set(ylabel="F1",ylim=(0,1));save(fig,"f1_by_length",out)
 f=ci[ci.metric=="f1"];fig,ax=plt.subplots();ax.errorbar(range(4),f.estimate,yerr=[f.estimate-f.ci95_low,f.ci95_high-f.estimate],fmt="o");ax.set(xticks=range(4),xticklabels=f.model,ylabel="F1",ylim=(0,1));save(fig,"f1_confidence_intervals",out)
 d=diff[diff.metric=="f1"];fig,ax=plt.subplots();ax.errorbar(range(len(d)),d.estimate,yerr=[d.estimate-d.ci95_low,d.ci95_high-d.estimate],fmt="o");ax.axhline(0,color="gray",ls="--");ax.set(xticks=range(len(d)),xticklabels=d.comparison,ylabel="ΔF1");save(fig,"f1_difference_confidence_intervals",out)
 (out/"analysis_config.json").write_text(json.dumps({"bootstrap_repetitions":N_BOOT,"seed":42,"threshold":.5,"length_quartile_bounds":[float(v) for v in q]},indent=2));repair(out)
def main():
 p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=OUT);p.add_argument("--repair",action="store_true");a=p.parse_args();repair(a.output_dir) if a.repair else run(a.output_dir)
if __name__=="__main__":main()
