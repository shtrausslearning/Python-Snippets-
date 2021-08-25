# Cython + Python Implementation of Decision Tree Regression

# Evaluation Function

from sklearn.model_selection import train_test_split as tts
from sklearn import datasets

def eval_tts(ldf,feature='target', ratio=0.3,model=None):
    
    # Split feature/target variable
    y = ldf[feature].copy()
    X = ldf.copy()
    del X[feature]     # remove target variable
    
    X = X.values; y=y.values
    X_train,X_test,y_train,y_test = tts(X,y,test_size=ratio,
                                            random_state=32)
    ym_train = model.fit(X_train,y_train) 
    
    # Plot Training & Model Data (Linear
    fig = go.Figure()
    x_tr = [i for i in range(0,y_train.shape[0])]
    fig.add_traces(go.Scatter(x=x_tr,y=y_train,mode='lines'))
    fig.add_traces(go.Scatter(x=x_tr,y=model.predict(X_train),mode='lines'))
    fig.update_layout(template='plotly_white',
                      margin=dict(l=20, r=20, t=120, b=20),
                      title='Training | Model & Truth')
    fig.update_yaxes(range=[0,50])
    fig.update_layout(showlegend=False,height=300)
    fig.show()
    
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

df_boston = sklearn_to_df(datasets.load_boston())
y_truth = df_boston['target']
# display(df_boston.head())

model = CDTRegressor(max_depth=10,min_size=10)
%time y_model = eval_tts(df_boston,feature='target',ratio=0.3,model=model)
