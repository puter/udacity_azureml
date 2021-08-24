# Optimizing an ML Pipeline in Azure - Chad Puterbaugh

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
This dataset contains marketing information for prospective customers. We seek to predict whether the customer will respond to a marketing campaign.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
Using hyperdrive, the best performing model was a logistic regression model with 2500 max iterations and a C value of 1.536. Its accuracy was .916

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
The pipeline archecture starts with a call to register a tabular dataset. The data is retrieved using `TabularDatasetFactory.from_delimited_files()` which is made available through `azureml.core`. The dataset can be found at https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv . The dataset was then converted into a pandas dataframe. The dataset itself consists of 32950 observations of 21 features. Each observation represents the marketing data and result of the marketing campaign for each person recorded. The features include:

Attribute Information:

Input variables:
# bank client data:
  1 - age (numeric)
  2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
  3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
  4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
  5 - default: has credit in default? (categorical: 'no','yes','unknown')
  6 - housing: has housing loan? (categorical: 'no','yes','unknown')
  7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
  8 - contact: contact communication type (categorical: 'cellular','telephone')
  9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
  10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
  11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
  12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
  14 - previous: number of contacts performed before this campaign and for this client (numeric)
  15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
  16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
  17 - cons.price.idx: consumer price index - monthly indicator (numeric)
  18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
  19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
  20 - nr.employed: number of employees - quarterly indicator (numeric)

# Output variable (desired target):
  21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

https://archive.ics.uci.edu/ml/datasets/bank+marketing

The dataframe was then cleaned and prepared for training. The job feature was dropped entirely. `marital`, `default`, `housing`, `loan`, `poutcome` and `y` were all simplified and converted in place with a boolean 1 or 0 so that the logistic regression could handle the data mathematically. `married` was converted to 1, all others 0. For each of the others, a 'yes' was converted to a 1, and all other responses 0. Contact and dummies were 1-hot encoded using the `get_dummies` method. This method converts each categorical response into its own column, and essentially pivots the data in place using assigning a 1 if that category was true for the row, 0 if not. Similarly, this is done so the values can be mathematically used for inputs. Lastly, and time-based values were converted into integers for the same reason (e.g. months into mapped month of the year). The data cleanup also split the `y` feature (the tag) from the remaining `x` dataframe representing predictors for the regression. The resulting `x` and `y` dataframes were split into test and training groups using `train_test_split`. The training `x` and `y` dataframes were used to fit a logistic regression binary classifier, and the test `x` and `y` dataframes were used for validation. The hyperparamaters for the logistic regression training were fed dynamically using hyperdrive in successive runs. Each run was scored for its classification accuracy, and the highest accuracy results were saved off into the outputs folder. 

**What are the benefits of the parameter sampler you chose?**
The paramater sampler is a configuration setting provided to the hyperdrive instance which cycles through a variety of hyperparameters. In each run, the config provides a method for hyperdrive to select a new set of hyperparameters, and hyperdrive then has logging instruction for various model outputs that we are interested in understanding. In each run, the parameter is logged as well as the output. From there it is a matter of selecting which is the output variable we are interested in maximizing, and hyperdrive then selects the optimal set of paramaters that achieved this result from the run at its disposal. 

I chose a random parameter sampler. I chose the random sampler to get a sense as to initial values that might work with this sort of classification problem. On future experiments with this data, I would use the outputs from the random parameter sampler and feed them into another, such as grid or bayesian so as to limit the scope with a more efficient sweep of the hyperparameter's now restricted domains. This strategy would allow my process to get some notional direction of where the solution space is, and then allow me to more efficiently explore a solution space on a subsequent run. 

Exploring `--max_iter` allowed the model to determine the ballpark iterations needed for the logistic regression to converge on correct weights. For the parameter sampler, I chose a discrete selection of various orders of magnitude to arrive a convergence that made sense. I didn't want the numbers to grow too large and be wasteful, nor too small as to not allow the weights to converge. For `--C`, I chose a uniform distribution between .001 and 4 so as to understand in rough terms the right penalty scale for regularization. Regularization is attempting to penalize the model for being too specific to training data and growing too complex. In future runs, I might choose a bayseian sampler having some directional guidance on where the appropriate hyperparameters are. My parameter sampler attributes were:

  + 'C': uniform(0.001, 4.0), # Uniform distribution to apply various penalties to regularization
  + 'max_iter': choice([1000, 1500, 2000, 2500, 5000])})

https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#define-the-search-space

**What are the benefits of the early stopping policy you chose?**
The stopping policy I chose is called a bandit policy, which means it evaluates the output criteria against previous runs and stops a current run if it is not within a threshold of the most successful run at this point in the experiment (prevents wasteful successive runs). Compute time was of the essence for this experiment using free resources. The evaluation interval of 1 means that the policy is applied to every run. Delay interval of 5 was chosen so that at least the first half of the runs were evaluated. The slack factor of .1 terminates any run within a very small degredation than the current candidate model (1/1+.1) 
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#bandit-policy

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
The best performing model was a Voting Ensemble model with accuracy of .918. The voting power of the component models is also listed. For example, MaxAbsScaler, LightGBM was given .33333 voting weight in the ensemble, its training algorithm was assigned the hyperparameter of 20 as the minimum data in each leaf. The full complement of component models in the ensemble were: XGBoost Classifier, SGD, and LightGBM, sometimes represented multiple times with differing hyperparameters. Summary of automl config parameters detailed inline in the notebook file and here: 

 + experiment_timeout_minutes=30 # To not allow experiments to run indefinitely. 
 + task="classification" # To establish the type of algorithms to apply. We are interested in classifying customers as likely to buy. THis is a classification problem
 + primary_metric="accuracy" # Accuracy was a choice, I could have also chosen AUC. Accuracy maximizes a particular model's predictive power, but may not be as generally robust as if we had optimized for AUC
 + training_data=x_tabular_dataset # This is the registered dataset upon which the model was trained
 + label_column_name="y" # This is the output column used to build the model
 + n_cross_validations=5 # Instructions to ensure that the data is segmented 5 times for validation purposes. This process helps ensure the model is more robust, and still fully utilizes all the data for training
 + compute_target=compute_cluster # This is the compute cluster assigned to build the model and run the experiment

Other outputs and insights generated by AutoML:

The top 4 features important to the model were: Duration, nr_employed, emp_var_rate and cons_conf_idx
The AUC_weighted was also provided as .949. This is the metric that measures the true positive rate across a variety of threasholds. closer to 1 is better. 



## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
There was a small difference in the architecture of the two pipelines. I wanted to reuse as much of the code as I could, so I had to alter the training code to return the y_df in the results. The automl workflow when run remotely required that I registered the tabular dataset. That dataset also had to include the predictor. The automl workflow took the full timeline to run because I didn't restrict the solution space. AutoML performed much more accurately than the hyperdrive model. Hyperdrive only tried one classification algorithm, whereas automl explored the full solution space, and could arrive at a better model choice. The accuracy was only very marginally better, and to me not worth the extra execution time. 

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
Future experiments could include taking learnings from automl in terms of better performing models and potentially allow it to run for longer or on a more powerful machine. I was constrained by budget for this experiement. Allowing automl to iterate over a larger solution space potentially could yield a model that otherwise I would not have been able to discover myself. 
