# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import itertools
import logging


# pgmpy imports
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("corrosion_backend")

app = FastAPI(title="Corrosion BN Inference API", version="1.0")

# ---------------------------
# CORS middleware (local dev)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Dataset path helper (robust)
# ---------------------------
# The dataset may be in backend/ or project root; try a few places.
CANDIDATE_PATHS = [
    Path(__file__).parent / "corrosion_dataset.csv",              # backend/corrosion_dataset.csv
    Path(__file__).parent.parent / "corrosion_dataset.csv",       # project_root/corrosion_dataset.csv
    Path(__file__).parent.parent.parent / "corrosion_dataset.csv" # one level up if nested
]

def find_dataset_path() -> Optional[Path]:
    for p in CANDIDATE_PATHS:
        if p.exists():
            return p
    return None

DATA_PATH = find_dataset_path()
if DATA_PATH:
    logger.info(f"Dataset found at: {DATA_PATH}")
else:
    logger.info("Dataset not found in candidate paths; dataset endpoints will return 404 until CSV is placed.")


# ---------------------------
# Build the Bayesian network
# ---------------------------
def build_bn() -> BayesianNetwork:
    model = BayesianNetwork([
        ("CO2", "Corrosion"),
        ("H2S", "Corrosion"),
        ("Low_pH", "Corrosion"),
        ("Coating_damage", "Corrosion"),
        ("CP_effective", "Corrosion"),
        ("Corrosion", "Sensor_signal"),
    ])

    # Prior probabilities for root nodes (binary: 0=absent/false, 1=present/true)
    cpd_CO2 = TabularCPD("CO2", 2, [[0.7], [0.3]])            # P(CO2=0)=0.7, P(CO2=1)=0.3
    cpd_H2S = TabularCPD("H2S", 2, [[0.85], [0.15]])
    cpd_Low_pH = TabularCPD("Low_pH", 2, [[0.8], [0.2]])
    cpd_Coating = TabularCPD("Coating_damage", 2, [[0.9], [0.1]])
    cpd_CP = TabularCPD("CP_effective", 2, [[0.6], [0.4]])

    # Corrosion CPD: parents order ['CO2','H2S','Low_pH','Coating_damage','CP_effective']
    probs = []
    for vals in itertools.product([0, 1], repeat=5):
        risk_score = sum(vals)
        # CP_effective reduces risk (if CP_effective==1)
        if vals[-1] == 1:
            risk_score = max(0, risk_score - 1)
        p = min(0.95, 0.05 + 0.2 * risk_score)
        probs.append([1 - p, p])  # P(Corrosion=0), P(Corrosion=1)
    arr = np.array(probs).T.tolist()
    cpd_Corrosion = TabularCPD(
        "Corrosion", 2, arr,
        evidence=["CO2", "H2S", "Low_pH", "Coating_damage", "CP_effective"],
        evidence_card=[2, 2, 2, 2, 2],
    )

    # Sensor_signal (3 states: 0=good,1=ambiguous,2=bad), depends on Corrosion
    cpd_Sensor = TabularCPD(
        "Sensor_signal", 3,
        # columns correspond to Corrosion=0 and Corrosion=1
        [[0.85, 0.1], [0.10, 0.2], [0.05, 0.7]],
        evidence=["Corrosion"],
        evidence_card=[2]
    )

    model.add_cpds(cpd_CO2, cpd_H2S, cpd_Low_pH, cpd_Coating, cpd_CP, cpd_Corrosion, cpd_Sensor)
    if not model.check_model():
        raise RuntimeError("BN model failed validation")
    return model

MODEL = build_bn()
INFER = VariableElimination(MODEL)


# ---------------------------
# Pydantic request models
# ---------------------------
class InferRequest(BaseModel):
    CO2_ppm: Optional[float] = None
    H2S_ppm: Optional[float] = None
    pH: Optional[float] = None
    temperature_C: Optional[float] = None
    flow_m_s: Optional[float] = None
    inhibitor_eff: Optional[float] = None
    CP_voltage: Optional[float] = None

    # optional explicit discrete/categorical fields (if you already computed them)
    CO2_present: Optional[int] = None  # 0/1
    H2S_present: Optional[int] = None  # 0/1
    Low_pH: Optional[int] = None       # 0/1
    Coating_damage: Optional[int] = None  # 0/1
    CP_effective: Optional[int] = None    # 0/1
    Sensor_signal: Optional[int] = None   # 0/1/2


class BatchInferRequest(BaseModel):
    items: List[InferRequest]


# ---------------------------
# Mapping helpers
# ---------------------------
def map_to_discrete(req: InferRequest) -> Dict[str, Optional[int]]:
    """
    Heuristic mapping from continuous inputs to discrete BN variables.
    If user supplied discrete flags (e.g., CO2_present), they take precedence.
    Returns dict with keys: CO2, H2S, Low_pH, Coating_damage, CP_effective, Sensor_signal
    """
    # defaults
    CO2 = 0
    H2S = 0
    LP = 0
    CD = 0
    CP = 1  # assume CP effective by default (1==effective)
    SS = None  # sensor signal unknown

    # If explicit discrete values provided, use them
    if req.CO2_present is not None:
        CO2 = 1 if req.CO2_present else 0
    elif req.CO2_ppm is not None:
        # threshold - tune for your system. Here we pick 150 ppm as 'present'
        CO2 = 1 if float(req.CO2_ppm) > 150.0 else 0

    if req.H2S_present is not None:
        H2S = 1 if req.H2S_present else 0
    elif req.H2S_ppm is not None:
        H2S = 1 if float(req.H2S_ppm) > 5.0 else 0

    if req.Low_pH is not None:
        LP = 1 if req.Low_pH else 0
    elif req.pH is not None:
        LP = 1 if float(req.pH) < 6.0 else 0

    if req.Coating_damage is not None:
        CD = 1 if req.Coating_damage else 0
    # else: unknown -> assume 0 (no damage). If you have a coating sensor, set this field.

    if req.CP_effective is not None:
        CP = 1 if req.CP_effective else 0
    elif req.CP_voltage is not None:
        # CP more negative is usually better. We consider CP effective if voltage <= -0.6 V
        CP = 1 if float(req.CP_voltage) <= -0.6 else 0

    if req.Sensor_signal is not None:
        SS = int(req.Sensor_signal)

    return {
        "CO2": int(CO2),
        "H2S": int(H2S),
        "Low_pH": int(LP),
        "Coating_damage": int(CD),
        "CP_effective": int(CP),
        "Sensor_signal": SS,
    }


# ---------------------------
# Inference endpoint
# ---------------------------
@app.post("/api/infer")
def infer(req: InferRequest):
    """
    Run inference for Corrosion given either continuous or discrete inputs.
    Returns P_corrosion and evidence used.
    """
    try:
        disc = map_to_discrete(req)
        evidence: Dict[str, Any] = {
            "CO2": disc["CO2"],
            "H2S": disc["H2S"],
            "Low_pH": disc["Low_pH"],
            "Coating_damage": disc["Coating_damage"],
            "CP_effective": disc["CP_effective"],
        }
        # include Sensor_signal if provided
        if disc["Sensor_signal"] is not None:
            evidence["Sensor_signal"] = int(disc["Sensor_signal"])

        # run inference for 'Corrosion'
        q = INFER.query(["Corrosion"], evidence=evidence)
        # q is a DiscreteFactor; q.values[1] corresponds to P(Corrosion=1)
        p_corrosion = float(q.values[1])
        return {
            "P_corrosion": round(p_corrosion, 6),
            "evidence": evidence
        }
    except Exception as e:
        logger.exception("BN inference failed, falling back to heuristic")
        # fallback: return heuristic probability using same formula as synthetic generator
        try:
            CO2_ppm = float(req.CO2_ppm) if req.CO2_ppm is not None else 0.0
            H2S_ppm = float(req.H2S_ppm) if req.H2S_ppm is not None else 0.0
            pH = float(req.pH) if req.pH is not None else 7.0
            temp = float(req.temperature_C) if req.temperature_C is not None else 50.0
            flow = float(req.flow_m_s) if req.flow_m_s is not None else 1.0
            inhibitor = float(req.inhibitor_eff) if req.inhibitor_eff is not None else 1.0
            cpv = float(req.CP_voltage) if req.CP_voltage is not None else -0.8

            risk = (
                0.003 * CO2_ppm +
                0.05 * (1.0 if H2S_ppm > 5.0 else 0.0) +
                0.5 * (1.0 if pH < 6.0 else 0.0) +
                0.02 * max(0.0, temp - 50.0) +
                0.8 * (1.0 if flow < 0.3 else 0.0) +
                1.2 * (1.0 if inhibitor < 0.3 else 0.0) +
                0.7 * (1.0 if cpv > -0.6 else 0.0)
            )
            p = 1.0 / (1.0 + np.exp(-(risk - 1.5)))
            return {"P_corrosion": round(float(p), 6), "evidence": {}, "fallback": True, "error": str(e)}
        except Exception as ex2:
            logger.exception("Fallback heuristic failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)} ; fallback failed: {str(ex2)}")


@app.post("/api/batch_infer")
def batch_infer(req: BatchInferRequest):
    results = []
    for item in req.items:
        r = infer(item)
        results.append(r)
    return {"results": results}


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------------------
# Dataset preview endpoint
# ---------------------------
@app.get("/api/dataset_preview")
def dataset_preview(n: int = 10):
    """
    Returns a preview of the dataset (first n rows) and column names.
    Place corrosion_dataset.csv in backend/ or project root (auto-detected).
    """
    dp = find_dataset_path()
    if not dp:
        raise HTTPException(status_code=404, detail="dataset not found on server; place corrosion_dataset.csv in backend/ or project root")
    try:
        df = pd.read_csv(dp)
        rows = df.head(n).fillna("").to_dict(orient="records")
        return {"rows": rows, "columns": list(df.columns), "n_rows": len(df)}
    except Exception as e:
        logger.exception("Failed to read dataset")
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {str(e)}")


# ---------------------------
# Parameter learning (optional)
# ---------------------------
def prepare_df_for_learning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert continuous columns in the CSV to discrete columns matching BN variables:
     - CO2 -> CO2 (0/1)
     - H2S -> H2S (0/1)
     - pH  -> Low_pH (0/1)
     - Coating_damage -> Coating_damage (0/1) if present
     - CP_voltage -> CP_effective (0/1)
     - Sensor_signal -> Sensor_signal (0/1/2) if present
    This function tries to be conservative; you should adapt thresholds to your domain.
    """
    df2 = df.copy()
    # normalize column names to lower for lookup
    cols = {c.lower(): c for c in df2.columns}

    def get_col(*cands):
        for c in cands:
            k = c.lower()
            if k in cols:
                return cols[k]
        return None

    # map continuous columns
    co2_col = get_col("CO2_ppm", "co2", "co2_ppm")
    h2s_col = get_col("H2S_ppm", "h2s")
    ph_col = get_col("pH", "ph")
    cpv_col = get_col("CP_voltage", "cp_voltage", "cp")
    coating_col = get_col("Coating_damage", "coating_damage", "coating", "coat_damage")
    sensor_col = get_col("Sensor_signal", "sensor_signal", "sensor", "signal")

    # create discrete columns expected by the BN
    if co2_col:
        df2["CO2"] = df2[co2_col].fillna(0).apply(lambda v: 1 if float(v) > 150.0 else 0)
    else:
        df2["CO2"] = 0

    if h2s_col:
        df2["H2S"] = df2[h2s_col].fillna(0).apply(lambda v: 1 if float(v) > 5.0 else 0)
    else:
        df2["H2S"] = 0

    if ph_col:
        df2["Low_pH"] = df2[ph_col].fillna(7).apply(lambda v: 1 if float(v) < 6.0 else 0)
    else:
        df2["Low_pH"] = 0

    if coating_col:
        # assume numeric or binary indicator; treat truthy/non-zero as damaged
        def cd_map(v):
            try:
                return 1 if (float(v) != 0 and (not np.isnan(float(v)))) else 0
            except Exception:
                return 1 if str(v).strip().lower() in ("1", "true", "yes", "y") else 0
        df2["Coating_damage"] = df2[coating_col].fillna(0).apply(cd_map)
    else:
        df2["Coating_damage"] = 0

    if cpv_col:
        df2["CP_effective"] = df2[cpv_col].fillna(-0.8).apply(lambda v: 1 if float(v) <= -0.6 else 0)
    else:
        df2["CP_effective"] = 1  # assume effective if unknown

    if sensor_col:
        # try to cast to int and clip to [0,2]
        def sensor_map(v):
            try:
                iv = int(float(v))
                if iv < 0: return 0
                if iv > 2: return 2
                return iv
            except Exception:
                s = str(v).strip().lower()
                if s in ("good", "ok", "0"): return 0
                if s in ("ambiguous", "warn", "1"): return 1
                return 2
        df2["Sensor_signal"] = df2[sensor_col].fillna(0).apply(sensor_map)
    else:
        # unknown sensor -> set to NaN (we won't learn sensor CPD if absent)
        df2["Sensor_signal"] = np.nan

    # Keep only columns the estimator needs
    keep_cols = ["CO2", "H2S", "Low_pH", "Coating_damage", "CP_effective", "Corrosion", "Sensor_signal"]
    # If 'Corrosion' isn't present in dataset, try alt names
    if "Corrosion" not in df2.columns:
        alt = get_col("Corrosion", "corrosion", "label")
        if alt:
            df2["Corrosion"] = df2[alt]
    # Ensure Corrosion is 0/1 if present (try to coerce)
    if "Corrosion" in df2.columns:
        def corrosion_map(v):
            try:
                return 1 if float(v) != 0 else 0
            except Exception:
                s = str(v).strip().lower()
                return 1 if s in ("1", "true", "yes", "y", "corroded") else 0
        df2["Corrosion"] = df2["Corrosion"].apply(corrosion_map)
    else:
        # If label is missing, we cannot learn Corrosion CPD supervisedly; keep structure learning out-of-scope
        logger.info("No Corrosion label found in dataset; learning will only update priors for root nodes.")
    return df2


@app.post("/api/learn_cpds")
def learn_cpds(equivalent_sample_size: int = 10):
    """
    Learn CPDs from the detected CSV dataset using BayesianEstimator and update the in-memory BN MODEL.
    This will attempt to map continuous columns to discrete variables and estimate CPDs for nodes present in the data.
    """
    dp = find_dataset_path()
    if not dp:
        raise HTTPException(status_code=404, detail="dataset not found on server; place corrosion_dataset.csv in backend/ or project root")

    try:
        df = pd.read_csv(dp)
        df_prepared = prepare_df_for_learning(df)

        # keep only discrete columns (drop rows with NaN in required columns for a node)
        estimator = BayesianEstimator(MODEL, df_prepared)
        updated_cpds = []
        updated_nodes = []

        for node in MODEL.nodes():
            # Only estimate CPDs for nodes present in df_prepared
            if node not in df_prepared.columns:
                logger.debug(f"Skipping learning for '{node}' (not present in prepared dataframe)")
                continue
            try:
                cpd = estimator.estimate_cpd(node, prior_type='BDeu', equivalent_sample_size=equivalent_sample_size)
                updated_cpds.append(cpd)
                updated_nodes.append(node)
                logger.info(f"Estimated CPD for node: {node}")
            except Exception as e:
                logger.exception(f"Failed to estimate CPD for node {node}: {e}")

        if updated_cpds:
            # Replace CPDs in the global MODEL
            # Remove old CPDs for updated nodes and add the new ones
            # (Easiest is to keep all CPDs, then add new ones — pgmpy will keep the latest by name)
            for cpd in updated_cpds:
                try:
                    MODEL.add_cpds(cpd)
                except Exception:
                    # in some cases add_cpds raises; try removing existing and re-adding
                    try:
                        MODEL.remove_cpds(*[n for n in updated_nodes])
                        MODEL.add_cpds(*updated_cpds)
                    except Exception as e2:
                        logger.exception("Failed to replace CPDs cleanly")
                        raise HTTPException(status_code=500, detail=f"Failed to update model CPDs: {str(e2)}")

            # Validate model after update
            if not MODEL.check_model():
                raise RuntimeError("Learned CPDs produced an invalid model")
            # Recreate inference engine
            global INFER
            INFER = VariableElimination(MODEL)
            return {"status": "ok", "updated_nodes": updated_nodes}
        else:
            return {"status": "no_updates", "reason": "no matching nodes found in dataset to estimate CPDs"}

    except Exception as e:
        logger.exception("Failed during parameter learning")
        raise HTTPException(status_code=500, detail=f"Parameter learning failed: {str(e)}")


# ---------------------------
# If run directly, allow uvicorn to pick it up
# ---------------------------
if __name__ == "__main__":
    # Running via `python main.py` won't auto-reload; prefer uvicorn command:
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
