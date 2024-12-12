import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
pd.options.mode.chained_assignment = None
from rich.progress import track
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt



def preprocess_true_distance(steps, df_merged):

    # Calculate delta values for future steps
    df_merged['delta_x'] = df_merged['x'].shift(-steps) #- df_merged['player_position_x']
    df_merged['delta_y'] = df_merged['x'].shift(-steps)# - df_merged['player_position_y']
    # Remove rows with NaN values resulting from shift
    if steps > 0:
        df_merged = df_merged.iloc[:-steps]
    elif steps < 0:
        df_merged = df_merged.iloc[-steps:]
    else:
        pass  # steps = 0, no shift needed
    #print(df_merged['delta_X_from_O'].unique())
    return df_merged

def correlate_from_O(df_merged):

    # Drop rows with NaN values in 'delta_X' and 'delta_Y'
    y = df_merged[['delta_x', 'delta_y']]
    
    # Align 'df' data to match 'y' dimensions by dropping the last 'future_steps' rows
    features = df_merged[[f'hidden_{i}' for i in range(0, 256)]]

    current_pos = df_merged[['x', 'y']]
    current_pos_array = np.array(current_pos)

    X = np.array(features)
    Y = np.array(y)

    X_train, X_test, y_train, y_test, current_pos_train, current_pos_test = train_test_split(X, Y,current_pos_array, test_size = 0.25, shuffle=True,random_state=42) 

    model = Ridge(alpha=0.01)
    model.fit(X_train, y_train)

    return model, X_test, y_test, current_pos_test

if __name__ == "__main__":
    range1 = np.arange(-201, -20, 10)
    range2 = np.arange(-20, 20, 1)  # Note: the end is 21 to include 20
    range3 = np.arange(20, 201, 10)
    #range3 = np.arange(20,21)

    # Combine the ranges
    dt_list = np.concatenate((range1, range2, range3))

    l = [1, 6, 50, 200]
    all_rmse =[]
    all_naive = []
    for n in l:
        df = pd.read_csv(f"CSV_files/hidden_data{n}.csv", low_memory=False)
        print(len(df))
        #df_merged = preprocess_true_distance(10, df)
        #model, X_test1, y_test1, current_pos_test = correlate_from_O(df_merged)
        #print(len(df_merged))
        RMSE_ridge = []
        naive = []
        all_data = []
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
        # store the models before running the code!!!!
        for i in track(dt_list):
            df_merged = preprocess_true_distance(i, df)
            model, X_test, y_test, _ = correlate_from_O(df_merged)
            all_data.append([model])
            RMSE_ridge.append(root_mean_squared_error(y_true=y_test,y_pred=model.predict(X_test)) )
            naive.append(root_mean_squared_error(y_true=y_test,y_pred=np.full_like(y_test, 10)) )
        all_rmse.append(RMSE_ridge)
        all_naive.append(naive)

    plt.figure(1)
    plt.title("Displacement from origin")
    #plt.plot(dt_list, np.array(RMSE_linear), label="Linear")
    plt.plot(dt_list, np.array(all_rmse[0]), label="Ridge 1")
    plt.plot(dt_list, np.array(all_rmse[1]), label="Ridge 6")
    plt.plot(dt_list, np.array(all_rmse[2]), label="Ridge 50")
    plt.plot(dt_list, np.array(all_rmse[3]), label="Ridge 200")
    plt.plot(dt_list, np.array(all_naive[3]), label="Naive")
    #plt.plot(future_list, np.array(naive)/np.array(RMSE_linear), label="Naive")
    plt.legend()
    plt.ylabel("RMSE")
    plt.xlabel("Future steps")
    plt.show()