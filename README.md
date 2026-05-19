# Glass Feature Pair Analysis

## About
This project aims to analyse a glass identification dataset to find out the 2 most useful features which can be used to best differentiate and classify different glass types. To achieve this, this program makes use of a K-Nearest Neighbour Machine Learning algorithm with LOOCV, to find out the most accurate feature pair. Moreover, this program creates all (36) possible combinations of scatter graphs for supplementary visual analysis.

Additionally, during Exploratory Data Analysis (EDA) when collecting results, a pattern emerged when viewing table data and KNN performance for each feature pair. This led to the discovery of a post-analysis hypothesis, that a high number of zeros in features like Iron may be associated with lower KNN performance. This hypothesis was observed to be consistent with the KNN accuracy scores, and further inspection suggests it may align with known discussions in data science regarding feature sparsity, although this is not formally proven in this project.

## Attribution
This dataset was from the US Irvine Machine Learning Repository, created by B. German

Their DOI is: [10.24432/C5WW2P](https://archive.ics.uci.edu/dataset/42/glass+identification)

The website: [Glass Identification](https://archive.ics.uci.edu/dataset/42/glass+identification)

## Methodology
[Feature Pair Graph Visualisation](https://github.com/ProAstroShan/Glass-feature-classifier/blob/ce22d077bbcf3ef3a9a6a0959924b1805caafdc9/Feature%20Graph%20Images/Scatter-Graph-Program)
- Data is imported and formatted as a Pandas dataframe, and further cleaned to be column-based data, making it more suitable for graph plotting
- 36 Pairs of features are created using itertools, with each feature acting as the x-axis and y-axis respectively
- pd.factorize is used to colour datapoints based on the glass types, to allow identification of clusters

[KNN and LOOCV](https://github.com/ProAstroShan/Glass-feature-classifier/blob/ce22d077bbcf3ef3a9a6a0959924b1805caafdc9/LOOCV-KNN%20predictor/Program)
- Data is imported using the built-in python file system, cleaned, and formatted to be row-based data, making it more suitable for machine learning purposes
- All data is normalised within their features, preventing one feature from completely overshadowing another
- LOOCV is used, so every iteration, 1 row of data is used as a sample test data and the other rows as training data, to obtain extensive results and no randomness in final results
- Manual KNN is carried out to show what is going on behind the scenes, where sample data is being matched with training data with the shortest euclidean distance between them for prediction, where k=3 is used to prevent the algorithm from being too sensitive, while not underfitting

## Analysis

### Important Trends

#### Refractive Index
- Appeared in top 4 feature pairs
- Consistently produced strong class separation
- Suggests potentially high discriminative power

#### Iron
- Appeared in bottom 4 feature pairs
- Consistently weak in separating glass classes
- Likely contributes limited useful structure for KNN

### Best Vs Worst feature pairs

#### Best-performing pairs
These are feature pairs that rank highly and are supported by their graphs

Refractive Index + Potassium (Rank 1)
![Refractive Index vs. Potassium](Feature%20Graph%20Images/All%20Feature%20Graph/Refractive%20Index%20vs.%20Potassium.png)

Refractive Index + Magnesium (Rank 2)
![Refractive Index vs. Magnesium](Feature%20Graph%20Images/All%20Feature%20Graph/Refractive%20Index%20vs.%20Magnesium.png)

Observation: Clear separation of data points across multiple glass types based on colours, and clusters are well defined and visually distinctive

#### Worst-performing pairs
These are feature pairs that rank low and are supported by their graphs

Silicon + Iron (Rank 36)
![Silicon vs. Iron](Feature%20Graph%20Images/All%20Feature%20Graph/Silicon%20vs.%20Iron.png)

Sodium + Iron (Rank 35)
![Sodium vs. Iron](Feature%20Graph%20Images/All%20Feature%20Graph/Sodium%20vs.%20Iron.png)

Observation: Little to no separation of data points between glass types, and clusters either overlap with each other, or are not visually distinctive and are mostly homogenous throughout the graph

### Anomalous feature pairs
These are feature pairs, whose ranks are not necessarily supported, and even contradictory to their graphs.

Refractive Index + Calcium (Rank 3)
![Refractive Index vs. Calcium](Feature%20Graph%20Images/All%20Feature%20Graph/Refractive%20Index%20vs.%20Calcium.png)

The graph shows visual overlap, which suggests clusters are not well separated, making it expected to rank lower. However, this feature pair still obtains a high ranking.

Sodium + Silicon (Rank 32)
![Sodium vs. Silicon](Feature%20Graph%20Images/All%20Feature%20Graph/Sodium%20vs.%20Silicon.png)

The graph appears to be visually well separated, which would suggest a higher ranking. However, this feature pair obtains a low ranking.

### Glass table data observations -- Zeros in feature data

#### Abstract
- Some features do not contain any datapoints with zeros, which are Refractive Index, Sodium, Aluminium and Calcium.
- Some features contain some but infrequent zeros, which are Magnesium, Potassium and Silicon
- Some features contain many zeros, which are Barium and Iron
- Generally, there seems to be no consistent relationship between number of zeros in feature data and KNN performance. For example, features like Magnesium and Barium both appear in both high and low ranking feature pairs.
- **However, two consistent exceptions are Refractive Index, which does not contain any zeros and generally ranks high, and Iron, which contains many zeros and generally ranks low.**

#### Food for thought
- In forensic science and glass manufacturing, it is known that elements like sodium and silicon are major structural components of glass, while elements like iron and barium are either trace elements or impurities, potentially explaining why these features have many zeros
- After Z-Score normalisation, it would be expected that the effects of zeros would not significantly affect performance, but as shown here its impact can still be observed
- This observation may be related to limitations in the KNN algorithm, where sparse features may distort geometric distance calculations
- This observation may also relate to the phenomena of "Curse of Dimensionality" and "Feature Sparsity"
- The statement "The greater the number of zeros in feature data, the poorer its performance during KNN" appears to be a commonly discussed hypothesis, but this project does not formally prove it
- So within the scope of this project, the results are consistent with this hypothesis, but it is important to note that it does not **prove** it

### Conclusion of analysis
To conclude the analysis, the best feature pair to best differentiate and classify glass types is Refractive Index and Potassium, as they have the highest performance in KNN, while being strongly supported by their scatter graph. This also suggests that higher performance in KNN generally correlates to stronger geometric separation between glass types. Finally, this project provides evidence consistent with the hypothesis on feature sparsity, but does not formally prove it.

## Running of algorithms
- Download "glass1.csv" and use it in your programming environment
- Run "graph_creator.py", and images obtained can be found in "Feature Graph Images"
- Run "feature_predictor.py", and output can be found in feature_rankings.txt
