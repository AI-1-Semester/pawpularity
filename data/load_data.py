import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

def load_train_data():
    train_data = pd.read_csv("./data/train/train.csv")
    return train_data

def load_test_data():
    test_data = pd.read_csv("./data/test/test.csv")
    return test_data

def load_pawpularity_data():
    pawpularity_train_data = load_train_data()
    
    # Prepare the data
    df = pd.DataFrame(pawpularity_train_data)
    x = df.drop(['Id', 'Pawpularity', 'Subject Focus'], axis=1)
    y = df['Pawpularity']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}

def load_occlusion_data():
    occlusion_train_data = load_train_data()

    # Prepare the data
    df = pd.DataFrame(occlusion_train_data)
    x = df[['Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Info', 'Blur']]
    y = df['Occlusion']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}

def load_humanpred_data():
    humanpred_train_data = load_train_data()

    correlated_data = humanpred_train_data.drop(columns=['Id', 'Pawpularity', 'Action', 'Accessory', 'Near', 'Collage', 'Eyes', 'Face', 'Info', 'Subject Focus', 'Blur', 'Group'], axis=1)
    # Prepare the data
    df = pd.DataFrame(correlated_data)
    x = df.drop('Human', axis=1)
    y = df['Human']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}

def load_stacking_data():
    stacking_train_data = load_train_data()
    
    # Prepare the data
    df = pd.DataFrame(stacking_train_data)
    x = df.drop(['Id', 'Pawpularity'], axis=1)
    y = df['Pawpularity']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}

def load_clustering_data(n_samples=100, n_features=2, centers=3, random_state=42):
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
    n_train = int(0.8 * n_samples)
    x_train, x_test = X[:n_train], X[n_train:]
    return {
        'x_train': pd.DataFrame(x_train, columns=[f'feature_{i}' for i in range(n_features)]),
        'x_test': pd.DataFrame(x_test, columns=[f'feature_{i}' for i in range(n_features)]),
        'y_train': None,
        'y_test': None
    }