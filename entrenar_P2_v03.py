"""P2 v0.3: comparación retrospectiva de diagnóstico hasta fechachecklistvalido.
ATENCIÓN: fechacreacion <= hito no garantiza ausencia de ediciones posteriores.
M3 (A69) NO se entrena por falta de snapshot temporal de componentes.
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
A47=['registros_totales','componentes_distintos','registros_fabricar','unidades_fabricar',
     'registros_reponer','unidades_reponer','registros_limpieza','unidades_limpieza',
     'registros_compuestos','unidades_compuestas']
A57=['criterios_distintos','registros_criterio_no_resuelto','registros_fabricacion_criterio',
     'registros_compra_criterio','registros_limpieza_general','registros_recuperacion_menor',
     'registros_recuperacion_estandar','registros_recuperacion_mayor','registros_recuperacion_critica',
     'unidades_recuperacion_mayor','unidades_recuperacion_critica']

def make_model(cols):
    cat=[c for c in cols if c in BASE];num=[c for c in cols if c not in BASE]
    prep=ColumnTransformer([
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('hot',OneHotEncoder(handle_unknown='ignore'))]),cat),
        ('num',SimpleImputer(strategy='constant',fill_value=0),num)
    ])
    return Pipeline([('prep',prep),('rf',RandomForestRegressor(n_estimators=180,min_samples_leaf=3,
                                                              max_features=0.8,random_state=42,n_jobs=-1))])

def features(d, catalog=None):
    """Recalcular A47 y A57 desde las líneas fechadas, nunca A70 ya agregado."""
    d=d.copy()
    d['rep']=d['reparacion'].fillna('').astype(str).str.strip().str.lower().str.replace(r'\s+',' ',regex=True)
    d['qty']=pd.to_numeric(d['cantidad'],errors='coerce').clip(lower=0).fillna(0)
    d['cid']=pd.to_numeric(d['idcriterioevaluacion'],errors='coerce')
    d.loc[d.cid.le(0),'cid']=np.nan
    if catalog is not None:
        # Claves de catálogo inequívocas: tanto alias como nombre; sin asignaciones arbitrarias.
        cat=pd.concat([
            catalog[['idcriterioevaluacion','alias']].rename(columns={'alias':'key'}),
            catalog[['idcriterioevaluacion','nombrecriterioevaluacion']].rename(columns={'nombrecriterioevaluacion':'key'})
        ],ignore_index=True)
        cat['key']=cat.key.astype('string').str.strip().str.lower().str.replace(r'\s+',' ',regex=True)
        cat=cat.dropna(subset=['key']);cat=cat[cat.key.ne('')]
        valid=cat.groupby('key').idcriterioevaluacion.nunique()
        cat=cat.loc[cat.key.isin(valid[valid.eq(1)].index)].drop_duplicates('key')
        lookup=cat.set_index('key').idcriterioevaluacion
        raw=d.criterio.astype('string').str.strip().str.lower().str.replace(r'\s+',' ',regex=True)
        d['cid']=d['cid'].fillna(raw.map(lookup))

    # Para criterios sin ID directo, se permite mapear alias/nombre único desde el catálogo solo si se provee.
    d['clasificacion']=np.select([
        d.rep.isin(['fabricar','fabricar.','fabricación']),
        d.rep.eq('reponer'),d.rep.eq('limpieza'),d.rep.str.contains('+',regex=False)
    ],['FABRICAR','REPONER','LIMPIEZA','COMPUESTA'],default='OTRAS')
    base=d.groupby('idordentrabajo').agg(registros_totales=('idchecklistinftec','size'),
        componentes_distintos=('idcomponente',lambda v:v.dropna().loc[v.dropna()>0].nunique()),
        criterios_distintos=('cid','nunique'),
        registros_criterio_no_resuelto=('cid',lambda v:int((v.isna() | (v<=0)).sum()))).reset_index()
    for cat,prefix in [('FABRICAR','fabricar'),('REPONER','reponer'),('LIMPIEZA','limpieza'),('COMPUESTA','compuesto')]:
        part=d.loc[d.clasificacion.eq(cat)].groupby('idordentrabajo').agg(n=('idchecklistinftec','size'),q=('qty','sum'))
        base=base.merge(part.rename(columns={'n':('registros_compuestos' if prefix=='compuesto' else f'registros_{prefix}'), 'q':('unidades_compuestas' if prefix=='compuesto' else f'unidades_{prefix}')}),left_on='idordentrabajo',right_index=True,how='left')
    mapping={
        'registros_fabricacion_criterio':[2,3,4,13], 'registros_compra_criterio':[5,6,15],
        'registros_limpieza_general':[7], 'registros_recuperacion_menor':[8],
        'registros_recuperacion_estandar':[9], 'registros_recuperacion_mayor':[10],
        'registros_recuperacion_critica':[11]
    }
    for col,ids in mapping.items():
        s=d.loc[d.cid.isin(ids)].groupby('idordentrabajo').size().rename(col)
        base=base.merge(s,left_on='idordentrabajo',right_index=True,how='left')
    for col,code in [('unidades_recuperacion_mayor',10),('unidades_recuperacion_critica',11)]:
        s=d.loc[d.cid.eq(code)].groupby('idordentrabajo').qty.sum().rename(col)
        base=base.merge(s,left_on='idordentrabajo',right_index=True,how='left')
    return base.fillna(0)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--a70',default='A70_dataset_consolidado_bombas.csv')
    p.add_argument('--a72',default='A72')
    p.add_argument('--a74',default='A74')
    p.add_argument('--salida',default='resultados_P2_v03')
    p.add_argument('--a55',default=None,help='Catálogo de criterios para resolver códigos textuales inequívocos')
    p.add_argument('--corte',default='2025-01-01')
    args=p.parse_args(); out=Path(args.salida);out.mkdir(parents=True,exist_ok=True)
    a70=pd.read_csv(args.a70);a72=pd.read_csv(args.a72);a74=pd.read_csv(args.a74)
    for df,name in [(a70,'A70'),(a72,'A72')]:
        if df.idordentrabajo.duplicated().any():raise ValueError(name+' tiene OT repetidas')
    if a74.idchecklistinftec.duplicated().any():raise ValueError('A74 tiene idchecklistinftec duplicados')
    for col in ['fecharecepcion','fechachecklistvalido','fin_observado']:
        a72[col]=pd.to_datetime(a72[col],errors='coerce',format='mixed')
    a74['fechacreacion']=pd.to_datetime(a74.fechacreacion,errors='coerce',format='mixed')
    a74['fechachecklistvalido']=pd.to_datetime(a74.fechachecklistvalido,errors='coerce',format='mixed')
    if (a74.fechacreacion>a74.fechachecklistvalido).any(): raise ValueError('A74 contiene líneas posteriores a validación')
    if not set(a74.idordentrabajo).issubset(set(a72.idordentrabajo)): raise ValueError('A74 tiene OT fuera de A72')
    identity=a70[['idordentrabajo']+BASE].copy()
    d=a72.merge(identity,on='idordentrabajo',validate='one_to_one')
    d=d.loc[d.fechachecklistvalido.ge(d.fecharecepcion)&d.fechachecklistvalido.le(d.fin_observado)].copy()
    catalog=pd.read_csv(args.a55) if args.a55 else None
    feats=features(a74,catalog);d=d.merge(feats,on='idordentrabajo',how='left',validate='one_to_one')
    d['tiene_lineas_al_hito']=d.registros_totales.notna().astype(int)
    for c in A47+A57:d[c]=d[c].fillna(0)
    d['objetivo_dias']=(d.fin_observado-d.fechachecklistvalido).dt.total_seconds()/86400
    d['fecha_hito']=d.fechachecklistvalido
    cutoff=pd.Timestamp(args.corte)
    train=d.loc[(d.fecha_hito<cutoff)&(d.fin_observado<cutoff)&d.objetivo_dias.ge(0)].copy()
    test=d.loc[(d.fecha_hito>=cutoff)&d.objetivo_dias.ge(0)].copy()
    # No se pueden usar etiquetas de OTs cuyos resultados ocurrieron después del corte.
    excluded=d.loc[(d.fecha_hito<cutoff)&(d.fin_observado>=cutoff)].copy()
    if train.empty or test.empty:raise ValueError('Entrenamiento o prueba vacíos')
    known=set(train.idmodelo.dropna());test['tipo_modelo']=np.where(test.idmodelo.isin(known),'conocido','no_visto')
    metricas=[];pred=[]
    models={'MEDIANA':None,'P2_M1':BASE+A47,'P2_M2':BASE+A47+A57}
    for name,cols in models.items():
        if cols is None:estimated=np.repeat(train.objetivo_dias.median(),len(test))
        else:
            estimator=make_model(cols); estimator.fit(train[cols],train.objetivo_dias)
            estimated=np.maximum(0,estimator.predict(test[cols]))
        actual=test.objetivo_dias.to_numpy()
        for group in ['todos','conocido','no_visto']:
            mask=np.ones(len(test),dtype=bool) if group=='todos' else test.tipo_modelo.eq(group).to_numpy()
            if not mask.any():continue
            metricas.append({'modelo':name,'grupo':group,'n_train':len(train),'n_test':int(mask.sum()),
                            'mae_dias':mean_absolute_error(actual[mask],estimated[mask]),
                            'medae_dias':median_absolute_error(actual[mask],estimated[mask])})
        for i,(_,r) in enumerate(test.iterrows()):
            pred.append({'modelo':name,'idordentrabajo':int(r.idordentrabajo),'tipo_modelo':r.tipo_modelo,
                         'fecha_hito':r.fecha_hito,'real_dias':actual[i], 'predicho_dias':float(estimated[i]),
                         'error_absoluto_dias':abs(actual[i]-estimated[i])})
    pd.DataFrame(metricas).to_csv(out/'metricas_P2_v03.csv',index=False)
    pd.DataFrame(pred).to_csv(out/'predicciones_P2_v03.csv',index=False)
    pd.DataFrame([{'a70_ot':len(a70),'a72_ot':len(a72),'a74_lineas':len(a74),'a74_ot':a74.idordentrabajo.nunique(),'catalogo_a55_usado':bool(args.a55),
        'p2_hito_en_ventana':len(d),'p2_con_lineas_al_hito':int(d.tiene_lineas_al_hito.sum()),
        'train':len(train),'test':len(test),'fuera_por_resultado_futuro':len(excluded),
        'test_modelos_no_vistos':int(test.tipo_modelo.eq('no_visto').sum()),
        'corte':str(cutoff.date()),
        'P2_M3':'NO ENTRENADO: A69 carece de fecha por componente; posible fuga temporal',
        'advertencia':'Fechacreacion no certifica versiones de registros editados despues del hito; resultado retrospectivo'}]).to_csv(out/'auditoria_P2_v03.csv',index=False)
    print('P2: train',len(train),'test',len(test),'resultados futuros excluidos',len(excluded))
    print(pd.DataFrame(metricas).to_string(index=False,float_format=lambda x:f'{x:.2f}'))
    print('M3 no entrenado: A69 carece de fotografia temporal. Resultados:',out)

if __name__=='__main__':main()
