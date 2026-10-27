import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. CREAR UN DATASET SIMULADO (Histórico del Taller)
# Esto simula lo que extraerías de tus tablas mct_checklistinftec y mct_ot_tiempos_reales
datos_historicos = {
    'pieza': [
        'Carcasa Principal', 'Tapa Lateral', 'Eje Principal', 'Carcasa Principal', 
        'Impulsor', 'Tapa Lateral', 'Eje Principal', 'Carcasa Principal'
    ],
    'material': [
        'Acero Fundido', 'Bronce', 'Acero Forjado', 'Hierro Nodular', 
        'Acero Inoxidable', 'Acero Fundido', 'Acero Forjado', 'Acero Fundido'
    ],
    'estado_llegada': [
        'Cavitación severa', 'Desgaste por fricción', 'Fisura superficial', 'Fisura estructural', 
        'Cavitación moderada', 'Desgaste leve', 'Desgaste en asiento', 'Abrasión extrema'
    ],
    'solucion_reparacion': [
        'Relleno de soldadura y mecanizado', 'Mecanizado y encamisado', 'Torneado y temple', 'Biselado y soldadura pesada', 
        'Relleno y pulido superficial', 'Rectificado plano', 'Relleno y rectificado de asiento', 'Aplicación de recubrimiento duro'
    ],
    'peso_kg': [150.0, 45.0, 80.0, 200.0, 30.0, 50.0, 85.0, 170.0],
    'horas_hombre_real': [25.5, 8.5, 16.0, 38.0, 10.0, 6.0, 18.5, 22.0] # Target (Lo que queremos predecir)
}

df = pd.DataFrame(datos_historicos)

# 2. SEPARAR VARIABLES DE ENTRADA (X) Y LA VARIABLE A PREDECIR (y)
X = df[['pieza', 'material', 'estado_llegada', 'solucion_reparacion', 'peso_kg']]
y = df['horas_hombre_real']

# 3. PREPROCESADOR PARA DATOS CATEGÓRICOS (Texto -> Números con One-Hot Encoding)
preprocesador = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['pieza', 'material', 'estado_llegada', 'solucion_reparacion'])
    ],
    remainder='passthrough' # Mantiene el peso_kg intacto
)

# 4. CREAR EL PIPELINE CON RANDOM FOREST
modelo_ml = Pipeline(steps=[
    ('preprocessor', preprocesador),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. ENTRENAR EL MODELO
# (En un dataset real de producción usarías train_test_split, aquí entrenamos con todo para el ejemplo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
modelo_ml.fit(X_train, y_train)

# 6. SIMULAR LA LLEGADA DE UNA NUEVA PIEZA AL TALLER
nueva_pieza = pd.DataFrame({
    'pieza': ['Carcasa Principal'],
    'material': ['Acero Fundido'],
    'estado_llegada': ['Cavitación severa'],
    'solucion_reparacion': ['Relleno de soldadura y mecanizado'],
    'peso_kg': [155.0]
})

# 7. REALIZAR LA PREDICCIÓN
horas_estimadas = modelo_ml.predict(nueva_pieza)

print("="*50)
print(" 🛠️ REPORTE DE PREDICCIÓN DE TALLER (MACHINE LEARNING)")
print("="*50)
print(f"Pieza ingresada: {nueva_pieza['pieza'].values[0]}")
print(f"Material: {nueva_pieza['material'].values[0]} ({nueva_pieza['peso_kg'].values[0]} kg)")
print(f"Estado de llegada: {nueva_pieza['estado_llegada'].values[0]}")
print(f"Solución planificada: {nueva_pieza['solucion_reparacion'].values[0]}")
print("-"*50)
print(f"⏱️ TIEMPO ESTIMADO POR EL MODELO: {horas_estimadas[0]:.2f} horas-hombre")
print("="*50)