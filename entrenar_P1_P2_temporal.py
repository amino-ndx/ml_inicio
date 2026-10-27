"""P1 recepción y P2 exploratorio posdiagnóstico. A70, SIN acceso PostgreSQL.
Python 3.10+, pandas numpy scikit-learn.
P2 usa último checklist como PROXY y NO acredita diagnóstico validado ni versión as-of.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE=['idproducto','idmodelo','idsubfamilia','idfamilia']
A47=['registros_totales','componentes_distintos','registros_fabricar','unidades_fabricar','registros_reponer','unidades_reponer','registros_limpieza','unidades_limpieza','registros_compuestos','unidades_compuestas']
A57=['criterios_distintos','registros_criterio_no_resuelto','registros_fabricacion_criterio','registros_compra_criterio','registros_limpieza_general','registros_recuperacion_menor','registros_recuperacion_estandar','registros_recuperacion_mayor','registros_recuperacion_critica','unidades_recuperacion_mayor','unidades_recuperacion_critica']
A69=['componentes_ot','componentes_con_checklist','componentes_sin_checklist','porcentaje_componentes_diagnosticados','lineas_checklist_relacionadas']
CAT=set(BASE)

def estimator(cols):
    categorical=[c for c in cols if c in CAT]; numeric=[c for c in cols if c not in CAT]
    prep=ColumnTransformer([
        ('cat',Pipeline([('missing',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),categorical),
        ('num',SimpleImputer(strategy='median'),numeric)
    ])
    return Pipeline([('preparacion',prep),('rf',RandomForestRegressor(n_estimators=180,min_samples_leaf=3,max_features=0.8,random_state=42,n_jobs=-1))])

def evaluate(frame, scenario, cutoff, cfg, results, preds):
    target='objetivo_dias'; train=frame.loc[frame['fecha_corte']<cutoff].copy(); test=frame.loc[frame['fecha_corte']>=cutoff].copy()
    train=train.loc[train[target].notna()&(train[target]>=0)]
    test=test.loc[test[target].notna()&(test[target]>=0)]
    if len(train)<30 or len(test)<10:
        print(f'{scenario}: partición insuficiente ({len(train)} entrenamiento; {len(test)} prueba)');return
    model_ids=set(train.idmodelo.dropna());test['tipo_modelo']=np.where(test.idmodelo.isin(model_ids),'conocido','no_visto')
    for label,columns in cfg.items():
        model=estimator(columns)
        model.fit(train[columns],train[target]); p=np.maximum(model.predict(test[columns]),0)
        baseline=float(train[target].median()); b=np.full(len(test),baseline)
        for idx,part in [('todos',np.ones(len(test),dtype=bool)),('conocidos',test.tipo_modelo.eq('conocido').to_numpy()),('no_vistos',test.tipo_modelo.eq('no_visto').to_numpy())]:
            if not part.any(): continue
            obs=test[target].to_numpy()[part]
            for name,est in [('MEDIANA',b[part]),(label,p[part])]:
                results.append(dict(escenario=scenario,subgrupo=idx,modelo=name,n_entrenamiento=len(train),n_prueba=int(part.sum()),mae_dias=mean_absolute_error(obs,est),medae_dias=median_absolute_error(obs,est),fecha_corte=cutoff.date().isoformat()))
        for ot,actual,estimated,group in zip(test.idordentrabajo,test[target],p,test.tipo_modelo):
            preds.append(dict(escenario=scenario,modelo=label,idordentrabajo=ot,tipo_modelo=group,real_dias=actual,predicho_dias=estimated))
    print(f'{scenario}: {len(train)} entrenamiento, {len(test)} prueba; modelos no vistos en prueba: {(test.tipo_modelo=="no_visto").sum()}')

def main():
    a=argparse.ArgumentParser(); a.add_argument('--csv',default='A70_dataset_consolidado_bombas.csv');a.add_argument('--salida',default='resultados_P1_P2');a.add_argument('--corte',default='2025-01-01');args=a.parse_args()
    d=pd.read_csv(args.csv);d.columns=d.columns.str.strip()
    required=set(BASE+A47+A57+A69+['idordentrabajo','tiene_checklist','fecharecepcion','inicio_observado','ultimo_checklist','fin_observado','dias_ciclo_observado'])
    missing=sorted(required-set(d.columns));
    if missing: raise ValueError('Faltan columnas: '+str(missing))
    if d.idordentrabajo.duplicated().any():raise ValueError('A70 contiene OT duplicadas')
    for c in ['fecharecepcion','inicio_observado','ultimo_checklist','fin_observado']:
        d[c]=pd.to_datetime(d[c],errors='coerce',format='mixed')
    cutoff=pd.Timestamp(args.corte)
    output=Path(args.salida);output.mkdir(parents=True,exist_ok=True)
    results=[]; preds=[]
    # Recepción. Se excluyen totalmente variables de checklist y A69 cuya disponibilidad a recepción no está demostrada.
    p1=d.loc[d.fecharecepcion.notna() & d.fin_observado.gt(d.fecharecepcion)].copy()
    p1['fecha_corte']=p1.fecharecepcion
    p1['objetivo_dias']=(p1.fin_observado-p1.fecharecepcion).dt.total_seconds()/86400
    p1['mes_recepcion']=p1.fecharecepcion.dt.month
    evaluate(p1,'P1_recepcion',cutoff,{'P1_IDENTIDAD':BASE+['mes_recepcion']},results,preds)
    # P2 es solo aproximación: NO usarlo en producción como diagnóstico formalmente validado.
    p2=d.loc[d.tiene_checklist.eq(1)&d.ultimo_checklist.ge(d.fecharecepcion)&d.ultimo_checklist.le(d.inicio_observado)&d.fin_observado.gt(d.ultimo_checklist)].copy()
    p2['fecha_corte']=p2.ultimo_checklist
    p2['objetivo_dias']=(p2.fin_observado-p2.ultimo_checklist).dt.total_seconds()/86400
    evaluate(p2,'P2_PROXY_NO_VALIDADO',cutoff,{'P2_M1':BASE+A47,'P2_M2':BASE+A47+A57,'P2_M3':BASE+A47+A57+A69},results,preds)
    pd.DataFrame(results).to_csv(output/'metricas.csv',index=False)
    pd.DataFrame(preds).to_csv(output/'predicciones.csv',index=False)
    pd.DataFrame([dict(ot_totales=len(d),p1_elegibles=len(p1),p2_proxy_elegibles=len(p2),sin_checklist=int(d.tiene_checklist.eq(0).sum()),corte=str(cutoff.date()),advertencia='P2 no es operativo sin hito y snapshot validado del diagnostico')]).to_csv(output/'auditoria.csv',index=False)
    print('\nMETRICAS:')
    print(pd.DataFrame(results).to_string(index=False,float_format=lambda v:f'{v:.2f}'))
    print('\nATENCION: P1 estima fin de movimientos, NO fecha efectiva de despacho. P2 proxy no acredita datos as-of. Archivos:',output.resolve())
if __name__=='__main__': main()
