"""Comparación P1 identidad vs P1 + A54 (mismas OT, corte cronológico fijo).
Incluye evaluación estricta: sólo train con objetivo terminado antes del corte.
P2 mantiene resultados anteriores; sin hito validado no hay entrenamiento operativo nuevo.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error
from entrenar_P1_P2_temporal import estimator, BASE
A54=['otras_ots_30d','movimientos_maquina_30d','maquinas_distintas_30d','operadores_distintos_30d','horas_registradas_maquina_30d']

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',default='A70_dataset_consolidado_bombas.csv');p.add_argument('--a54',default='q 54.csv');p.add_argument('--corte',default='2025-01-01');p.add_argument('--salida',default='resultados_P1_v02');args=p.parse_args()
    out=Path(args.salida);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.csv,encoding='utf-8-sig');a=pd.read_csv(args.a54,encoding='utf-8-sig');d.columns=d.columns.str.strip();a.columns=a.columns.str.strip()
    assert not d.idordentrabajo.duplicated().any() and not a.idordentrabajo.duplicated().any()
    missing=set(A54)-set(a.columns)
    if missing:raise ValueError(f'Faltan columnas A54: {sorted(missing)}')
    d=d.merge(a[['idordentrabajo','fecharecepcion']+A54],on='idordentrabajo',how='left',validate='one_to_one',suffixes=('','_a54'),indicator=True)
    if not d._merge.eq('both').all():raise ValueError('Faltan OT A54; se detiene para no cambiar población')
    d['fecharecepcion']=pd.to_datetime(d.fecharecepcion,format='mixed',errors='coerce');d['fecha_a54']=pd.to_datetime(d.fecharecepcion_a54,format='mixed',errors='coerce');d['fin_observado']=pd.to_datetime(d.fin_observado,format='mixed',errors='coerce')
    delta=(d.fecharecepcion-d.fecha_a54).dt.total_seconds().abs()
    if not (delta<1).all():raise ValueError('Fechas de recepción distintas entre A54 y A70')
    d['objetivo_dias']=(d.fin_observado-d.fecharecepcion).dt.total_seconds()/86400
    d['mes_recepcion']=d.fecharecepcion.dt.month
    d=d[d.fecharecepcion.notna() & d.fin_observado.gt(d.fecharecepcion)].copy()
    cut=pd.Timestamp(args.corte)
    rows=[]; pred=[]; checks=[]
    for modo in ['replica_original','estricto_sin_futuros']:
        train=d[d.fecharecepcion<cut].copy();test=d[d.fecharecepcion>=cut].copy()
        after=int((train.fin_observado>=cut).sum());checks.append(dict(modo=modo,train_candidatas=len(train),train_fin_no_conocido_en_corte=after,test=len(test)))
        if modo=='estricto_sin_futuros':train=train[train.fin_observado<cut].copy()
        if len(train)<30 or len(test)<10:continue
        known=set(train.idmodelo.dropna());test['grupo']=np.where(test.idmodelo.isin(known),'conocido','no_visto')
        models={'MEDIANA':None,'P1_IDENTIDAD':BASE+['mes_recepcion'],'P1_IDENTIDAD_A54':BASE+['mes_recepcion']+A54}
        for name,cols in models.items():
            if cols is None: pp=np.full(len(test),float(train.objetivo_dias.median()))
            else:
                m=estimator(cols);m.fit(train[cols],train.objetivo_dias);pp=np.maximum(m.predict(test[cols]),0)
            y=test.objetivo_dias.to_numpy()
            for grp in ['todos','conocido','no_visto']:
                mask=np.ones(len(test),bool) if grp=='todos' else test.grupo.eq(grp).to_numpy()
                if mask.sum()==0:continue
                rows.append(dict(modo=modo,subgrupo=grp,modelo=name,ot_entrenamiento=len(train),ot_prueba=int(mask.sum()),mae_dias=mean_absolute_error(y[mask],pp[mask]),medae_dias=median_absolute_error(y[mask],pp[mask])))
            for ot,actual,estimated,grupo in zip(test.idordentrabajo,y,pp,test.grupo):pred.append(dict(modo=modo,modelo=name,idordentrabajo=int(ot),grupo=grupo,real_dias=actual,predicho_dias=estimated,error_absoluto_dias=abs(actual-estimated)))
    pd.DataFrame(rows).to_csv(out/'metricas_P1_v02.csv',index=False)
    pd.DataFrame(pred).to_csv(out/'predicciones_P1_v02.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'auditoria_P1_v02.csv',index=False)
    print(pd.DataFrame(checks).to_string(index=False));print(pd.DataFrame(rows).to_string(index=False,float_format=lambda f:f'{f:.2f}'))
    print('Archivos:',out.resolve())
if __name__=='__main__':main()
