import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

def load_clustering_data(sample_size = 20):
    clustering_train_data = load_train_data()
    
    df = pd.DataFrame(clustering_train_data)
    x = df.drop(['Id', 'Pawpularity'], axis = 1)
    y = df['Pawpularity']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    x_train = x_train.sample(n=sample_size, random_state=42)
    x_test = x_test.sample(n=sample_size, random_state=42)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return {
        'x_train': pd.DataFrame(x_train_scaled, columns=x.columns),
        'x_test': pd.DataFrame(x_test_scaled, columns=x.columns),
        'y_train': y_train,
        'y_test': y_test
    }
    