# Glass Feature Pair Analysis
## About
This project aims to analyse a glass identification dataset to find out the 2 most useful features which can be used to best differentiate and classify different glass types. To achieve this, this program makes use of a K-Nearest Neighbour Machine Learning algorithm with LOOCV, to find out the most accurate feature pair. Moreover, this program creates all (36) possible combinations of scatter graphs for supplementary visual analysis.

## Atribution
This dataset was from the US IRvine machine learning repository, created by B. German 

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
- Manual KNN is carried out to show what is going on behine the scenes, where sample data is being matched with training data with the shortest euclidean distance between them for prediction, where a k=3 is used to prevent athe algorith from being too sensitive, while not underfitting

## Analysis
