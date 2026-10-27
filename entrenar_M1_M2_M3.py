"""Experimento retrospectivo M1, M2 y M3 con A70.
Uso: python entrenar_M1_M2_M3.py --csv A70_dataset_consolidado_bombas.csv
Dependencias: pandas numpy scikit-learn
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = [
    'idmodelo','idsubfamilia','idfamilia',
    'registros_totales','componentes_distintos',
    'registros_fabricar','unidades_fabricar','registros_reponer','unidades_reponer',
    'registros_limpieza','unidades_limpieza','registros_compuestos','unidades_compuestas',
]
A57 = [
    'criterios_distintos','registros_criterio_no_resuelto',
    'registros_fabricacion_criterio','registros_compra_criterio',
    'registros_limpieza_general','registros_recuperacion_menor',
    'registros_recuperacion_estandar','registros_recuperacion_mayor',
    'registros_recuperacion_critica','unidades_recuperacion_mayor',
    'unidades_recuperacion_critica',
]
A69 = [
    'componentes_ot','componentes_con_checklist','componentes_sin_checklist',
    'porcentaje_componentes_diagnosticados','lineas_checklist_relacionadas',
]
MODELS = {'M1_A47':BASE, 'M2_A47_A57':BASE+A57, 'M3_A47_A57_A69':BASE+A57+A69}
TARGETS = ['dias_ejecucion_observados','dias_ciclo_observado']
CATEGORICAL = ['idmodelo','idsubfamilia','idfamilia']

def build_model(columns, seed):
    categories = [v for v in CATEGORICAL if v in columns]
    numeric = [v for v in columns if v not in categories]
    transformer = ColumnTransformer([
        ('categoricas',Pipeline([
            ('imputar',SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(handle_unknown='ignore')),
        ]),categories),
        ('numericas',SimpleImputer(strategy='median'),numeric),
    ])
    return Pipeline([
        ('preprocesar',transformer),
        ('modelo',RandomForestRegressor(n_estimators=150,min_samples_leaf=3,
                                        max_features=0.8,n_jobs=-1,random_state=seed)),
    ])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',default='A70_dataset_consolidado_bombas.csv')
    parser.add_argument('--salida',default='resultados_ML')
    args=parser.parse_args()
    src=Path(args.csv)
    data=pd.read_csv(src)
    required=set(BASE+A57+A69+TARGETS+['idordentrabajo','tiene_checklist'])
    missing=sorted(required-set(data.columns))
    if missing: raise ValueError('Columnas faltantes: '+', '.join(missing))
    if data.idordentrabajo.duplicated().any(): raise ValueError('Hay OT repetidas en A70')
    df=data.loc[data.tiene_checklist.eq(1)].copy()
    print(f'A70: {len(data)} OT; con checklist: {len(df)}; modelos: {df.idmodelo.nunique()}')
    last=pd.to_datetime(df['ultimo_checklist'],errors='coerce')
    start=pd.to_datetime(df['inicio_observado'],errors='coerce')
    end=pd.to_datetime(df['fin_observado'],errors='coerce')
    print(f'ADVERTENCIA temporal: checklist ultimo posterior al inicio: {(last>start).sum()}, posterior al fin: {(last>end).sum()}')
    print('No se usan fechas, identificador OT ni objetivos como variables de entrada.')
    out=Path(args.salida);out.mkdir(parents=True,exist_ok=True)
    metrics=[]; predictions=[]
    for target in TARGETS:
        valid=df.loc[df[target].notna() & (df[target]>0)].reset_index(drop=True).copy()
        for scenario in ['conocidos','no_vistos']:
            if scenario=='conocidos':
                splits=ShuffleSplit(n_splits=5,test_size=0.25,random_state=42).split(valid)
            else:
                splits=GroupShuffleSplit(n_splits=5,test_size=0.25,random_state=42).split(valid,groups=valid.idmodelo)
            for fold,(tr,te) in enumerate(splits,1):
                if scenario=='conocidos':
                    known=set(valid.iloc[tr].idmodelo)
                    te=np.array([idx for idx in te if valid.iloc[idx].idmodelo in known],dtype=int)
                    if not len(te): continue
                train=valid.iloc[tr];test=valid.iloc[te]
                if scenario=='no_vistos':
                    assert set(train.idmodelo).isdisjoint(set(test.idmodelo))
                baseline=float(train[target].median())
                truth=test[target].to_numpy()
                base_mae=mean_absolute_error(truth,np.full(len(test),baseline))
                metrics.append(dict(objetivo=target,escenario=scenario,fold=fold,modelo='MEDIANA',
                                    n_entrenamiento=len(train),n_prueba=len(test),mae_dias=base_mae))
                for name,cols in MODELS.items():
                    model=build_model(cols,42+fold)
                    model.fit(train[cols],train[target])
                    pred=model.predict(test[cols])
                    score=mean_absolute_error(truth,pred)
                    metrics.append(dict(objetivo=target,escenario=scenario,fold=fold,modelo=name,
                                        n_entrenamiento=len(train),n_prueba=len(test),mae_dias=score))
                    predictions.extend(dict(objetivo=target,escenario=scenario,fold=fold,
                                            modelo=name,idordentrabajo=ot,
                                            real_dias=real,predicho_dias=estimate)
                                       for ot,real,estimate in zip(test.idordentrabajo,truth,pred))
    metric_df=pd.DataFrame(metrics)
    aggregate=metric_df.groupby(['objetivo','escenario','modelo'],as_index=False).agg(
        mae_promedio_dias=('mae_dias','mean'),mae_desviacion=('mae_dias','std'),
        pruebas=('fold','count'),tamano_prueba_promedio=('n_prueba','mean'))
    metric_df.to_csv(out/'metricas_por_particion.csv',index=False)
    aggregate.to_csv(out/'resumen_metricas.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'predicciones_prueba.csv',index=False)
    print('\nRESULTADOS (MAE promedio de 5 particiones, en días):')
    print(aggregate.to_string(index=False,float_format=lambda x:f'{x:.2f}'))
    print('\nResultados escritos en:',out.resolve())
    print('IMPORTANTE: experimento retrospectivo, NO estimación posdiagnóstico validada temporalmente.')

if __name__=='__main__':main()
