import json
import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import sys

# Calcola percorso base (cartella dello script) per riferimenti ai file
base_dir = os.path.dirname(os.path.abspath(__file__))

# --- 1. DEFINIZIONE PATH (assoluti) ---
# Usiamo percorsi assoluti basati sulla cartella dello script in modo che
# il file venga trovato anche se lo script è eseguito da una working directory diversa.
train_file_path = os.path.join(base_dir, 'train.jsonl')
test_file_path = os.path.join(base_dir, 'test.jsonl')
full_data_path = os.path.join(base_dir, 'train_completo.jsonl')

# Assicura che la cartella corrente dello script sia nel path per import locali
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# --- Importa la funzione di split dal file esterno ---
# Import locale: abbiamo già inserito `base_dir` in sys.path più sopra
try:
    from split_dataset import split_data
except ImportError:
    print("ATTENZIONE: File 'split_dataset.py' non trovato.")
    print("Assicurati che sia nella stessa cartella di questo script.")
    # Potresti voler uscire se lo split è fondamentale
    # exit()

# --- 2. CONTROLLO E SPLIT DATASET (se necessario) ---
# Controlliamo se i file di split (train.jsonl e test.jsonl) esistono GIA'.
# Se uno dei due (o entrambi) manca, eseguiamo lo split.
#
if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
    print(f"'{train_file_path}' o '{test_file_path}' non trovati.")
    print("Esecuzione split del dataset...")
    
    # Chiamiamo la funzione dal file split_dataset.py
    # Assicurati che 'split_data' sia importabile
    try:
        split_success = split_data(
            full_data_path=full_data_path,
            train_output_path=train_file_path,
            test_output_path=test_file_path
        )
        
        if not split_success:
            print("ERRORE: Fallimento nella creazione dei file di split. Uscita.")
            exit() # Interrompe lo script se non può creare i file
        else:
            print("File di split creati con successo.")
    except NameError:
        print(f"ERRORE: La funzione 'split_data' non è stata trovata.")
        print("Impossibile creare i file. Controlla 'split_dataset.py'.")
        exit()
else:
    print(f"Trovati '{train_file_path}' e '{test_file_path}' esistenti. Split saltato.")


# --- 3. CARICAMENTO DATI ---
train_data = []
#print(f"\nCaricamento dati da '{train_file_path}'...") #printinutile
try:
    with open(train_file_path, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
    #print(f"Caricate {len(train_data)} battaglie di training.") #printinutile

    # Ispezione primo elemento (opzionale)
    if train_data:
        first_battle = train_data[0]
        battle_for_display = first_battle.copy()
        battle_for_display['battle_timeline'] = battle_for_display.get('battle_timeline', [])[:2]
        #print("\n--- Struttura prima battaglia (troncata): ---") #printinutile
        #print(json.dumps(battle_for_display, indent=4)) #printinutile
        #if len(first_battle.get('battle_timeline', [])) > 2: #printinutile
        #    print("    ...") #printinutile

except FileNotFoundError:
    print(f"ERRORE: Impossibile trovare '{train_file_path}'.")
    print("Assicurati che 'train_completo.jsonl' esista per generarlo.")
    exit()

# Caricamento test data
test_data = []
#print(f"\nCaricamento dati da '{test_file_path}'...") #printinutile
try:
    with open(test_file_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    #print(f"Caricate {len(test_data)} battaglie di test.") #printinutile
except FileNotFoundError:
    print(f"ERRORE: Impossibile trovare '{test_file_path}'.")
    exit()


# --- 4. FUNZIONE DI FEATURE ENGINEERING ---
def feature_extractor(data):
    df_rows = []
    
    # Liste predefinite per consistenza
    all_types = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting',
                 'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost',
                 'dragon', 'notype']
    
    stat_keys = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
    boost_keys = ['atk', 'def', 'spa', 'spd', 'spe'] 

    for battle in tqdm(data, desc="Estrazione feature"):
        row = {}

        # === ID e TARGET ===
        row['battle_id'] = battle.get('battle_id', f"battle_{len(df_rows)}")
        row['player_won'] = battle.get('player_won', None)

        # === 1️⃣ FEATURE STATICHE
        p1_team = battle.get('p1_team_details', [])
        p1_stats = {s: [] for s in stat_keys}
        p1_types = []

        for p in p1_team:
            for s in stat_keys:
                p1_stats[s].append(p.get(s, 0))
            p1_types.extend([t.lower() for t in p.get('types', []) if t and t.lower() in all_types])

        # Statistiche aggregate P1
        for s, values in p1_stats.items():
            row[f'p1_team_{s}_mean'] = np.mean(values) if values else 0
            row[f'p1_team_{s}_sum'] = np.sum(values) if values else 0

        # Conteggio Tipi P1
        for t in all_types:
            row[f'p1_type_{t}_count'] = p1_types.count(t)
        row['p1_type_diversity'] = len(set(p1_types))

        
        # === 2️⃣ FEATURE STATICHE
        # Non stimiamo più il team di P2. Usiamo solo i dati che conosciamo: il lead.
        p2_lead = battle.get('p2_lead_details', {})
        p2_lead_stats = {}
        p2_lead_types = []

        # Statistiche P2 Lead
        for s in stat_keys:
            stat_val = p2_lead.get(s, 0)
            row[f'p2_lead_{s}'] = stat_val
            p2_lead_stats[s] = stat_val # Per usarlo nel bilanciamento

        # Tipi P2 Lead
        p2_lead_types = [t.lower() for t in p2_lead.get('types', []) if t and t.lower() in all_types]
        for t in all_types:
            row[f'p2_lead_type_{t}_count'] = p2_lead_types.count(t)
        row['p2_lead_type_diversity'] = len(set(p2_lead_types))

        
        # === 3️⃣ FEATURE DI BILANCIAMENTO (P1-Team vs P2-Lead) ===
        # Confrontiamo la media dell'intero team P1 con il solo lead di P2.
        p1_total_sum = 0
        p2_lead_total_sum = 0
        
        for s in stat_keys:
            p1_mean = row[f'p1_team_{s}_mean']
            p2_lead_stat = p2_lead_stats[s]
            
            row[f'diff_p1_mean_vs_p2_lead_{s}'] = p1_mean - p2_lead_stat
            
            p1_total_sum += row[f'p1_team_{s}_sum']
            p2_lead_total_sum += p2_lead_stat

        # Rapporto di forza (Totale stats P1 vs Totale stats P2 Lead)
        row['team_vs_lead_power_ratio'] = p1_total_sum / (1 + p2_lead_total_sum)
        row['type_diversity_diff'] = row['p1_type_diversity'] - row['p2_lead_type_diversity']

        
        # === 4️⃣ FEATURE DINAMICHE (Timeline) ===
        
        timeline = battle.get('battle_timeline', [])
        
        # Inizializziamo i valori di default (per battaglie con timeline vuota)
        p1_hp_list = [1.0]
        p2_hp_list = [1.0]
        p1_status_count = 0
        p2_status_count = 0
        p1_move_cats = {'PHYSICAL': 0, 'SPECIAL': 0, 'STATUS': 0}
        p2_move_cats = {'PHYSICAL': 0, 'SPECIAL': 0, 'STATUS': 0}
        
        last_p1_hp = 1.0
        last_p2_hp = 1.0
        last_p1_status = 'nostatus'
        last_p2_status = 'nostatus'
        last_p1_boosts = {b: 0 for b in boost_keys}
        last_p2_boosts = {b: 0 for b in boost_keys}
        
        p2_seen_names = set()
        turns_seen = {}

        if timeline:
            # Svuotiamo le liste se la timeline non è vuota
            p1_hp_list = []
            p2_hp_list = []
            
            for turn in timeline[:30]: # Assicuriamoci di non superare i 30 turni
                p1_state = turn.get('p1_pokemon_state', {})
                p2_state = turn.get('p2_pokemon_state', {})

                # --- Aggregati HP ---
                p1_hp_list.append(p1_state.get('hp_pct', 1.0))
                p2_hp_list.append(p2_state.get('hp_pct', 1.0))

                # --- Conteggio Status ---
                if p1_state.get('status') != 'nostatus':
                    p1_status_count += 1
                if p2_state.get('status') != 'nostatus':
                    p2_status_count += 1
                
                # --- Conteggio Categorie Mosse 
                p1_move = turn.get('p1_move_details')
                p2_move = turn.get('p2_move_details')
                if p1_move and p1_move.get('category') in p1_move_cats:
                    p1_move_cats[p1_move.get('category')] += 1
                if p2_move and p2_move.get('category') in p2_move_cats:
                    p2_move_cats[p2_move.get('category')] += 1

                # --- Tracciamento P2 Visti
                name = p2_state.get('name')
                if name:
                    lower_name = name.lower()
                    if lower_name not in p2_seen_names:
                        turns_seen[lower_name] = turn['turn']
                    p2_seen_names.add(lower_name)

            # --- Snapshot all'ultimo turno ---
            if timeline: # Assicuriamoci che non fosse vuota
                last_turn = timeline[-1]
                last_p1_state = last_turn.get('p1_pokemon_state', {})
                last_p2_state = last_turn.get('p2_pokemon_state', {})
                
                last_p1_hp = last_p1_state.get('hp_pct', p1_hp_list[-1])
                last_p2_hp = last_p2_state.get('hp_pct', p2_hp_list[-1])
                last_p1_status = last_p1_state.get('status', 'nostatus')
                last_p2_status = last_p2_state.get('status', 'nostatus')
                last_p1_boosts = last_p1_state.get('boosts', last_p1_boosts)
                last_p2_boosts = last_p2_state.get('boosts', last_p2_boosts)


        # --- Assegnazione Feature Dinamiche ---

        # Aggregati Timeline (HP, Status, Mosse)
        row['p1_hp_avg'] = np.mean(p1_hp_list) if p1_hp_list else 1.0
        row['p2_hp_avg'] = np.mean(p2_hp_list) if p2_hp_list else 1.0
        row['p1_hp_min'] = np.min(p1_hp_list) if p1_hp_list else 1.0
        row['p2_hp_min'] = np.min(p2_hp_list) if p2_hp_list else 1.0
        
        row['p1_status_turns'] = p1_status_count
        row['p2_status_turns'] = p2_status_count
        row['status_turns_diff'] = p1_status_count - p2_status_count
        
        for cat, count in p1_move_cats.items():
            row[f'p1_move_{cat}_count'] = count
        for cat, count in p2_move_cats.items():
            row[f'p2_move_{cat}_count'] = count
            
        # Feature 'p2_avg_turn_first_seen' (dalla tua versione, è ottima)
        row['p2_avg_turn_first_seen'] = np.mean(list(turns_seen.values())) if turns_seen else 30

        # Snapshot Stato Finale (HP, Status, Boosts)
        row['last_p1_hp'] = last_p1_hp
        row['last_p2_hp'] = last_p2_hp
        row['last_hp_diff'] = last_p1_hp - last_p2_hp
        
        # One-hot encoding dello status finale (gestito da fillna(0) dopo)
        row[f'last_p1_status_{last_p1_status}'] = 1
        row[f'last_p2_status_{last_p2_status}'] = 1
        
        # Boost finali e differenziale
        for b in boost_keys:
            p1_b_val = last_p1_boosts.get(b, 0)
            p2_b_val = last_p2_boosts.get(b, 0)
            row[f'last_p1_boost_{b}'] = p1_b_val
            row[f'last_p2_boost_{b}'] = p2_b_val
            row[f'last_boost_diff_{b}'] = p1_b_val - p2_b_val

        df_rows.append(row)

    # fillna(0) è fondamentale. Crea le colonne per tutti i tipi e status
    # (es. 'p1_type_fire_count', 'last_p1_status_frz')
    # e le imposta a 0 se non erano presenti in una specifica battaglia.

    return pd.DataFrame(df_rows).fillna(0)


# --- 5. CREAZIONE DATAFRAME ---
#print("\nProcessamento dati di training...") #printinutile
train_df = feature_extractor(train_data)

#print("\nProcessamento dati di test...") #printinutile
test_df = feature_extractor(test_data)

'''print("\nAnteprima features di training (prime 5 righe, prime 5 colonne):") #printinutile
print(train_df.iloc[:, :5].head())'''


# --- 6. PREPARAZIONE PER IL MODELLO ---
#print("\nPreparazione dati per il modello...") #printinutile
# Definiamo features (X) e target (y)
features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]

if 'player_won' not in train_df.columns:
    print("ERRORE: 'player_won' non trovato in train_df. Impossibile addestrare.")
    exit()

X_train = train_df[features]
y_train = train_df['player_won']

# Assicurati che X_test abbia le stesse colonne, nello stesso ordine
X_test = test_df[features]


# --- 7. TRAINING DEL MODELLO (xgboost) ---
from xgboost import XGBClassifier

# XGBoost classifier - configura iperparametri di base
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# --- 8. VALUTAZIONE MODELLO ---
if 'player_won' in test_df.columns:
    y_test = test_df['player_won']
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nACCURATEZZA MODELLO: {accuracy:.4f}")
else:
    print("\n'player_won' non presente nel test set. Accuratezza non calcolabile.")

