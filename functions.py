import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso
# function to plot a boxplot and a histogram along the same scale.

def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

    # function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )



   
    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot

def mape_score(targets, predictions):
    return np.mean(np.abs(targets - predictions) / targets) * 100
    
def model_performance_regression(model,predictors,target):
    pred = model.predict(predictors)
    r2 = r2_score(target,pred)
    adjr2 = adj_r2_score(predictors, target,pred)
    rmse = np.sqrt(mean_squared_error(target,pred))
    mae = mean_absolute_error(target,pred)
    mape = mape_score(target, pred)  #
    
    ## Creating a dataframe of the matrics
    df_per = pd.DataFrame(
    {
        "RMSE": rmse,
        "MAE": mae,
        "R-Squared": r2,
        "Adj R2 Squared": adjr2,
        "MAPE": mape,
    }, index=[0],)
    return df_per

def tunealpha(model, features, target, start, stop, num=10):
    alpha_values = np.linspace(start, stop, num=num)
    for alpha in alpha_values:
        estimator = model(alpha=alpha, random_state=0)
        estimator = estimator.fit(features,target)
        print(f'Alpha is {alpha}')
        print(model_performance_regression(estimator,features,target))

def adj_r2_score(predictors, target,predictions):
    r2 = r2_score(target,predictions)
    n = predictors.shape[0]
    k = predictors.shape[1]
    return 1-((1-r2)*(n-1)/(n-k-1))