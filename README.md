# CS:GO Round Winner Classification
> A simple tabular ML project based on a video game called Counter Strike: Global Offensive

#### -- Project Status: [Completed]

### Data
The dataset that is used in this project can be obtained from [Christian Lillelund's CS:GO Round Winner Classification](https://www.kaggle.com/christianlillelund/csgo-round-winner-classification) dataset, which is available on Kaggle. The dataset contains 122411 round snapshots gathered from professional-level CS:GO matches. The features present in the dataset are the remain- ing time in the round, the scores (or round wins) of each team, map, bomb plant status, total health of each team, armor statuses of each team, number of players alive on each team, the number of defuse kits of CTs, and one-hot encoded values for the weapons present in each team. Fortunately, the data was already preprocessed or cleaned by the author, hence why there are no missing values in the entirety of the dataset.
### Methods Used
* Basic EDA
* Data Preprocessing
* Data Transformation
* Feature Engineering
* Logistic Regression, Decision Trees, Random Forest

### Technologies
* Python
* Pandas
* Matplotlib, Seaborn
* Scikit-learn

## Project Description
Counter-Strike: Global Offensive, or CS:GO for short, is a tactical shooter game with a lot of intricacies. As an avid fan of the game, I wanted to implement a machine learning model which can predict the outcome of a round in a game of CS:GO. Valve, the developers of CS:GO, has actually implemented this system in the game in a recent update where the winning probabilities on different timestamps of each team were displayed at the end of each round. But, of course, the source code or methods which they use to create their prediction system was not shared with the public. I hope to recreate similar system to Valve’s where a model can predict which team will win based on a given set of features. I will also be using different data preprocessing and feature engineering techniques in the hopes of improving the model’s accuracy. 


## Domain Knowledge
### CS:GO and Its Objectives
CS:GO is a tactical first-person shooter game where two teams battle to win a particular objective. It is usually played on a PC and can be downloaded via the Valve client. 10players are divided into two different teams, namely the Terrorist and Counter-Terrorist. They will be referred to as T and CT for the rest of the paper. Both will fight for a maximum of30 rounds and the first team to accumulate 16 round wins will be crowned the champion. After 15 rounds, the players in each team will be swapped to provide balance to the game.

The simplest objective that can be achieved to grant a round win is by eliminating the players on the opposing team. However, each side also has its own special objectives.Ts, for example, can plant a bomb at a designated bombsite in a map and wait for it to explode to get the round win. CTs can win the round in two other different ways: wait for the round timer to expire, or defuse a planted bomb. These intricate objectives naturally place the Ts as the “attackers” and CTs as the “defenders”. In other words, Ts will try to plant the bomb while CTs will try to stop them from doing so.

### Maps
There are currently 8 different playable maps in CS:GO. Each map has a unique layout and usually has a tendency to benefit a certain side. For example, the de_dust2 map is considered by many to be T-sided or, in simpler terms, Ts have a much higher chance of winning a round in de_dust2.

### Economy
Economy is perhaps one of the most integral and defining aspects of CS:GO as a competitive FPS. Each player in the game will be given a certain amount of money before a round starts. Money can be earned by winning rounds (losing rounds give lesser money) and eliminating players. This money can then be used to buy guns, armor, and utilities (or grenade). Needless to say, the team with more money will have an advantage since they can buy better weapons. However, there is a certain price limit in which players can buy full armor, utility, and guns, which is capped at around $5500 (this figure differs between Ts, CTs, and weapon choices). Therefore, a player with max money ($16000) will not have any advantage over a player with $5500. When players have a bad economy (money less than $5500), they even sometimes opt to buy lower quality and cheap weapons so that they can reach this $5500 "full-buy" range on the next round.

### Armor and Helmet System
Having armor and a helmet can single-handedly be the difference between winning or losing a gunfight. However, the mechanics in which these items are implemented are quite unique in CS:GO. Helmets cannot be depleted and prevent aim-punches (the sudden movement of the crosshair and screen) when shot in the head. It also prevents several weapons from doing one-hit kills (by blocking some amount of damage) when getting shot in the head. Armor has the same effects as helmets, but on the body. However, armor can be depleted at a very slow rate. Damages to armor are incredibly minuscule that it can sustain up to 3 rounds on average. It should also be noted that lower armor does not mean blocking less damage. A player with 100 armor has no advantage over a player with 50 armor. However, having 0 armor almost guarantees to lose a gunfight due to the aforementioned reasons.

## Code
### Data Exploration
```python
plt.figure(figsize=(10,8))
sns.countplot(x="map", hue="round_winner", data=df)
plt.show()
```

```python
plt.figure(figsize=(10,8))
ax = sns.barplot(x=df['round_winner'].unique(), y=df['round_winner'].value_counts())
plt.show()
```

### Data Preprocessing
Let's **drop single-valued columns** (filled with only zeroes) from the dataset, which are mostly present in the one-hot encoded columns of the weapons. Since these guns are never used by each of the corresponding sides, removing them from these columns is reasonable for the occasion of this dataset.
```python
single_cols=[]
for i, col in enumerate(df.columns):
    if (df[col].nunique() == 1):
        single_cols.append(df.columns[i])
        
for col in single_cols:
    df.drop(col, axis = 1, inplace=True)
```
I also **encoded non-numeric columns** both with label encoding and one-hot encoding methods. The three non-numeric columns in our dataset are the `map` (8 different string values), `bomb-planted` status (boolean values), and the `round_winner` (either T or CT). Label encoding is applied to the bomb-plant status and round winner columns since they are binary-valued quantities and ordinality is not an issue here. One-hot encoding, on the other, will be better suited to be applied to the map column since it is a non-ordinal feature.
```python
non_numeric = [col for col in df.columns if df[col].dtype != 'float64']

label_encoder = LabelEncoder() 
df['bomb_planted'] = label_encoder.fit_transform(df['bomb_planted'])
df['round_winner'] = label_encoder.fit_transform(df['round_winner'])

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(df[['map']]))
df = df.join(OH_cols)
df.drop(['map'], axis=1, inplace = True)
```
I initially **scaled or normalized** our data using sklearn’s `StandardScaler`, since I also did some experiments with logistic regression. However, the logistic regression model performs very poorly and so I opted to use decision trees and random forest classifiers, which do not require scaling. After several trials, normalizing the data seems to marginally worsen the two models’ performance. As a result, I did not normalize or scale the data.

### Feature Engineering

The first technique that I implement for our feature engineering is **Principal Component Analysis**. PCA is commonly used, in feature engineering, to identify "important" features that describe or accounts for most of the variance present in the data. It can also be used to find the correlation between features, which may aid in creating or engineering new ones.

```python
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)
```
Unfortunately, the PCA results are quite underwhelming. When plotting the explained variance ratio of the Principal Components on a scree plot, almost all the features account for the same percentage of variation in the data. PC1, for example, only explains 9% of the variations in the dataset. Nevertheless, I tried integrating PCA by joining the values of PC1 (data points projected into the PC1 Eigenvector) into our dataset.

I also **created features using the domain knowledge of the game**. Due to the aforementioned armor mechanics in CS:GO, I decided to create a binary-valued feature that tracks the presence of armor on each team. Therefore, `t_has_armor` will have a value of 1 when `t_armor` is greater than 0, likewise with `ct_has_armor`.

```python
X['ct_has_armor'] = [0 if row == 0 else 1 for row in X['ct_armor']]
X['t_has_armor'] = [0 if row == 0 else 1 for row in X['t_armor']]
```
Having man advantages in CS:GO is also detrimental in winning rounds, especially in professional-level play. Therefore, I experimented with a feature called advantage, where the difference between `t_players_alive` and `ct_players_alive` will be tracked.

```python
X['advantage'] = X['t_players_alive'] - X['ct_players_alive']
X['advantage']
```

The last feature engineering technique I implemented was **K-means Clustering**. Unsupervised learning algorithms, such as K-means, can be used to identify patterns (or clusters) in the data that might be invisible to a supervised learning algorithm. The sklearn’s KMeans algorithm was run with a range of clusters from 2 to 12. For each number of cluster, I measured how the resulting clusters improved the performance of the Random Forest classifier. The best number of cluster is then picked.
```python
def find_cluster(X):
    best = 2;
    acc = 0;
    for i in range(10):
        kmeans = KMeans(n_clusters=i+2, random_state=1)
        X["Cluster"] = kmeans.fit_predict(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        rf_model = RandomForestClassifier(random_state=1)
        rf_model.fit(X_train,y_train)
        score = rf_model.score(X_test, y_test)
        if (score > acc):
            acc = score;
            best = i+2
    return best
    
n = find_cluster(X)
kmeans = KMeans(n_clusters = n, random_state=1)
X["Cluster"] = kmeans.fit_predict(X)
```
### Model Training & Evaluation

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

```python
def evaluate(y_pred, y_test):
    print("accuracy: ", accuracy_score(y_test, y_pred))
    print("precision: ", precision_score(y_test, y_pred))
    print("recall: ", recall_score(y_test, y_pred))
    print("F1 score: ", f1_score(y_test, y_pred))
    conf_mat = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(conf_mat, ['CT','T'], columns=['CT','T'])
    plt.figure(figsize = (10,8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues',cbar=False)
    plt.show()
```


```python
dt_model = DecisionTreeClassifier(random_state=1)
dt_model.fit(X_train,y_train)
y_pred = dt_model.predict(X_test)
evaluate(y_pred, y_test)

rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
evaluate(y_pred, y_test)
```
## Results

From the results, we can see that K-Means clustering and the `has_armor` feature produces each improves the accuracy, albeit only by a marginal scale, of the baseline model. Since both these features are presumably independent, integrating both of them into the data even improves the results even further to a high of 88.41% accuracy on the Random Forest Classifier.
As expected, including PCA did not yield a more accurate model. The `advantage` feature, unfortunately, did not improve the Random Forest Classifier but did manage to significantly boost the performance of the Decision Tree Classifier.

## Potential Improvements
Unfortunately, 88.41% was the best result I can obtain from our experiments. I have attempted and tried different approaches to get a result of 90% but these never came to fruition. I have several hypotheses on why a higher accuracy model was not able to be created.
Firstly, CS:GO, in its nature, is incredibly dynamic and unpredictable because its gameplay heavily emphasizes aim (or a player’s ability to shoot accurately) more than anything. The best CS:GO players in the world can, from time to time, "outplay" opponents and turn the tides of the round even with mediocre weapons due to their extraordinarily accurate aim. Even if a side seems to be winning, rounds can be virtually thrown at any second, which is why CS:GO matches are extremely entertaining to watch in the first place. One way to work around this problem is by tracking players’ performance such as accuracy or headshot percentage, which actually already exists in the current version of CS:GO.
CS:GO rounds also rely on momentum to a great extent. In other words, events that happen in the past will have a direct influence on the present. For example, a side that has won five rounds in a row will usually have a much higher chance of winning the next round due to less mental pressure and, arguably, "hotter aim". The dataset, unfortunately, has no way of detecting this highly influential momentum in the game.
