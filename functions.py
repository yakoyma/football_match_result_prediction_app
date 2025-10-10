"""
===============================================================================
This file contains all the functions for the project
===============================================================================
"""
# Libraries
import matplotlib.pyplot as plt


from sklearn.metrics import (roc_auc_score,
                             log_loss,
                             accuracy_score,
                             balanced_accuracy_score,
                             multilabel_confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay)



def display_pie_chart(dataset, var, figsize):
    """This function displays a pie chart with the proportions and
    count values.

    Args:
        dataset (pd.DataFrame): the Pandas dataset
        var (str): the variable (column of the dataset) to use
        title (str): the title of the chart
        figsize (tuple): the size of the chart
    """

    # Create a series with counted values
    dataviz = dataset[var].value_counts().sort_values(ascending=False)

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('equal')
    ax.pie(
        x=list(dataviz),
        labels=list(dataviz.index),
        autopct='%1.1f%%',
        pctdistance=0.5,
        labeldistance=1.05,
        textprops=dict(color='black', size=12, weight='bold')
    )
    plt.title(f'{var} variable categories', size=18, weight='bold')
    plt.axis('equal')
    plt.grid(False)
    plt.show()


def display_barplot(dataset, var, figsize):
    """This function displays a barplot.

    Args:
        dataset (pd.DataFrame): the Pandas dataset
        var (str): the variable (column of the dataset) to use
        figsize (tuple): the size of the chart
    """

    # Create the dataset for visualisation
    dataviz = dataset[var].value_counts().sort_values(ascending=False)

    # Set up the figure
    ax = dataviz.plot.bar(figsize=figsize)
    for p in ax.patches:
        ax.annotate(
            str(p.get_height()),
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha='center',
            va='bottom'
        )
    ax.set_xlabel(var)
    ax.set_ylabel('Count')
    ax.set_title(f'Plot of variable {var}')
    ax.legend(loc='best')
    ax.grid(True)
    plt.show()


def evaluate_multiclass_classification(y_test, y_pred, y_proba, labels):
    """This function evaluates the result of a Multiclass Classification.

    Args:
        y_test (ndarray): the test labels
        y_pred (ndarray): the predicted labels
        y_proba (ndarray): the predicted probabilities
        labels (ndarray): list of unique labels for Confusion Matrix Plot
    """

    if y_proba is not None:
        print('\n\nROC AUC: {:.3f}'.format(roc_auc_score(
            y_true=y_test, y_score=y_proba, multi_class='ovr')))
        print('Log loss: {:.3f}'.format(log_loss(
            y_true=y_test, y_pred=y_proba)))
    print('Accurcay: {:.3f}'.format(
        accuracy_score(y_true=y_test, y_pred=y_pred)))
    print('Balanced Accurcay: {:.3f}'.format(
        balanced_accuracy_score(y_true=y_test, y_pred=y_pred)))
    print('Multilabel Confusion Matrix:\n{}'.format(
        multilabel_confusion_matrix(y_true=y_test, y_pred=y_pred)))
    print('Classification Report:\n{}'.format(
        classification_report(y_true=y_test, y_pred=y_pred)))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred,
        display_labels=labels,
        xticks_rotation='vertical',
        cmap=plt.cm.Blues
    )
    display.ax_.set_title('Plot of the Confusion Matrix')
    plt.grid(False)
    plt.show()
