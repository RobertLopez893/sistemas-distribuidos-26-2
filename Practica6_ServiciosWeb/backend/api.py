from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Instanciamos la API
app = FastAPI(title="MLOps API Gateway")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modelos de Datos
class PeticionEntrenamiento(BaseModel):
    dataset: str
    modelo: str


# Memoria Compartida
leaderboard_global = []


# --- UTILIDAD: Simulador del Nodo de Datos ---
def obtener_datos(tipo_dataset: str):
    muestras = 10000 if tipo_dataset == "sintetico_10k" else 100000
    X, y = make_classification(n_samples=muestras, n_features=20, n_informative=15, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)


# --- RUTAS DE LA API (Endpoints) ---
@app.post("/api/entrenar")
def entrenar_modelo(peticion: PeticionEntrenamiento):
    """Recibe la orden del Frontend, entrena y guarda el resultado."""
    
    # Obtener Datos
    X_train, X_test, y_train, y_test = obtener_datos(peticion.dataset)
    
    # Instanciar Modelo
    if peticion.modelo == "SVM":
        clf = SVC()
    elif peticion.modelo == "Random Forest":
        clf = RandomForestClassifier(n_estimators=50)
    else:
        clf = LogisticRegression(max_iter=1000)
        
    # Entrenamiento Intensivo
    start = time.time()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    t_exec = time.time() - start
    
    # Registro en el Tablero
    resultado = {
        "modelo": peticion.modelo, 
        "accuracy": round(acc, 4), 
        "tiempo": round(t_exec, 2)
    }
    
    # Aquí FastAPI maneja la concurrencia por nosotros en peticiones simples
    leaderboard_global.append(resultado)
    leaderboard_global.sort(key=lambda x: x["accuracy"], reverse=True)
    
    # Retornamos un JSON al Frontend
    return {"status": "EXITO", "resultado": resultado}


@app.get("/api/leaderboard")
def obtener_leaderboard():
    """Devuelve la tabla de posiciones actual en formato JSON."""
    return {"leaderboard": leaderboard_global}
