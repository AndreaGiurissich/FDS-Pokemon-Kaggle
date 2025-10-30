import json
import os
from sklearn.model_selection import train_test_split

def split_data(full_data_path='train_completo.jsonl',
               train_output_path='train.jsonl',
               test_output_path='test.jsonl',
               test_size=0.3,
               random_state=42):
    """
    Carica un dataset completo da un file .jsonl, lo divide in set di 
    addestramento e test, e li salva in due nuovi file .jsonl.
    
    Restituisce True se ha successo, False altrimenti.
    """
    
    full_data = []
    print(f"\nLoading full data from '{full_data_path}'...")
    
    # Controlla se il file completo esiste prima di provare
    if not os.path.exists(full_data_path):
        print(f"ERROR: Full data file not found at '{full_data_path}'.")
        print("Please make sure 'train_completo.jsonl' is in the correct location.")
        return False
        
    try:
        with open(full_data_path, 'r') as f:
            for line in f:
                full_data.append(json.loads(line))
        print(f"Loaded {len(full_data)} battles from {full_data_path}.")
    
    except Exception as e:
        print(f"ERROR: Failed to load '{full_data_path}': {e}")
        return False

    # Esegui lo split
    if full_data:
        train_split, test_split = train_test_split(
            full_data,
            test_size=test_size,
            random_state=random_state
        )

        # Salva i due file
        try:
            with open(train_output_path, 'w') as f_train:
                for item in train_split:
                    f_train.write(json.dumps(item) + '\n')

            with open(test_output_path, 'w') as f_test:
                for item in test_split:
                    f_test.write(json.dumps(item) + '\n')
            
            print(f"\nSplit complete: {len(train_split)} train battles, {len(test_split)} test battles.")
            print(f"Saved to '{train_output_path}' and '{test_output_path}'.")
            return True  # Successo
            
        except IOError as e:
            print(f"ERROR writing split files: {e}")
            return False
    else:
        print("No data loaded. Split aborted.")
        return False

# Questa parte permette di eseguire il file anche da solo
# (es. python split_dataset.py) per forzare lo split
if __name__ == "__main__":
    print("Running split_dataset.py as a standalone script...")
    split_data()