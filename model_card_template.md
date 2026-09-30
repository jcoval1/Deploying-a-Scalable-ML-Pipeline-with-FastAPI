# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a logistic regression classifier created using scikit-learn. It predicts whether an individual's salary is greater than $50,000 or less than or equal to $50,000.

## Intended Use

The model is intended to predict an individual's salary using demographic and employment-related census data. It was developed as part of an API deployment project.

## Training Data

The model was trained using the provided census.csv dataset. The data was split into training and test sets using train_test_split with a random state of 42. Categorical features were processed using one-hot encoding.

## Evaluation Data

The model was evaluated using the test portion of the census.csv dataset. The test data was processed using the encoder fitted to the training data.

## Metrics

The model was evaluated using precision, recall, and F1 score. The model achieved a precision of 0.7202, recall of 0.2631, and F1 score of 0.3854.

## Ethical Considerations

The dataset includes demographic features such as race and sex. Patterns in the data may show different model performance among demographic groups, so predictions should not be used in decision making.

## Caveats and Recommendations

The model has low recall and performance varies across categorical data slices. Future improvements could include hyperparameter tuning.
