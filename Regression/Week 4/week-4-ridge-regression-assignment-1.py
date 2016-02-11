
# coding: utf-8

# # Regression Week 4: Ridge Regression (interpretation)

# In this notebook, we will run ridge regression multiple times with different L2 penalties to see which one produces the best fit. We will revisit the example of polynomial regression as a means to see the effect of L2 regularization. In particular, we will:
# * Use a pre-built implementation of regression (GraphLab Create) to run polynomial regression
# * Use matplotlib to visualize polynomial regressions
# * Use a pre-built implementation of regression (GraphLab Create) to run polynomial regression, this time with L2 penalty
# * Use matplotlib to visualize polynomial regressions under L2 regularization
# * Choose best L2 penalty using cross-validation.
# * Assess the final fit using test data.
# 
# We will continue to use the House data from previous notebooks.  (In the next programming assignment for this module, you will implement your own ridge regression learning algorithm using gradient descent.)

# # Fire up graphlab create

# In[150]:

import graphlab as gl
import graphlab
gl.canvas.set_target('ipynb')


# # Polynomial regression, revisited

# We build on the material from Week 3, where we wrote the function to produce an SFrame with columns containing the powers of a given input. Copy and paste the function `polynomial_sframe` from Week 3:

# In[151]:

def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = feature ** (power)
    return poly_sframe


# Let's use matplotlib to visualize what a polynomial regression looks like on the house data.

# In[152]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[153]:

sales = graphlab.SFrame('kc_house_data.gl/')


# As in Week 3, we will use the sqft_living variable. For plotting purposes (connecting the dots), you'll need to sort by the values of sqft_living first:

# In[154]:

sales = sales.sort('sqft_living')


# In[155]:

l2_small_penalty = 1.5e-5


# Let us revisit the 15th-order polynomial model using the 'sqft_living' as the input. Generate polynomial features up to degree 15 using `polynomial_sframe()` and fit a model with these features. After fitting the model, print out the learned weights.
# 
# Hint: make sure to add 'price' column to the new SFrame before calling `graphlab.linear_regression.create()`. To make sure GraphLab Create doesn't introduce any regularization, make sure you include the option `l2_penalty=1.5e-5` in this call.  Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set=None` in this call.

# In[156]:

poly_data = polynomial_sframe(sales['sqft_living'],15)


# In[157]:

poly_feature = poly_data.column_names()


# In[158]:

poly_data['price'] = sales['price']


# In[159]:

model = graphlab.linear_regression.create(poly_data,target='price', features = poly_feature,
                                              validation_set=None,verbose=False, l2_penalty=l2_small_penalty)


# In[160]:

model.get('coefficients')


# ***QUIZ QUESTION:  What's the learned value for the coefficient of feature `power_1`?***

# # Observe overfitting

# Recall from Week 3 that the polynomial fit of degree 15 changed wildly whenever the data changed. In particular, when we split the sales data into four subsets and fit the model of degree 15, the result came out to be very different for each subset. The model had a *high variance*. We will see in a moment that ridge regression reduces such variance. But first, we must reproduce the experiment we did in Week 3.

# First, split the data into split the sales data into four subsets of roughly equal size and call them `set_1`, `set_2`, `set_3`, and `set_4`. Use the `.random_split` function and make sure you set `seed=1`. 

# In[161]:

(semi_split1, semi_split2) = sales.random_split(.5,seed=1)
(set_1, set_2) = semi_split1.random_split(0.5, seed=1)
(set_3, set_4) = semi_split2.random_split(0.5, seed=1)
l2_small_penalty=1e-9


# Next, fit a 15th degree polynomial on `set_1`, `set_2`, `set_3`, and `set_4`, using 'sqft_living' to predict prices. Print the weights and make a plot of the resulting model.
# 
# Hint: To make sure GraphLab Create doesn't introduce any regularization, make sure you include the option `l2_penalty=0.0` when calling `graphlab.linear_regression.create()`.  Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set = None` in this call.

# In[162]:

poly_data_1 = polynomial_sframe(set_1['sqft_living'],15)
poly_feature_1 = poly_data_1.column_names()
poly_data_1['price'] = set_1['price']
model_1 = graphlab.linear_regression.create(poly_data_1,target='price', features = poly_feature_1,
                                              validation_set=None,verbose=False, l2_penalty=l2_small_penalty)
print model_1.get('coefficients')


# In[163]:

poly_data_2 = polynomial_sframe(set_2['sqft_living'],15)
poly_feature_2 = poly_data_2.column_names()
poly_data_2['price'] = set_2['price']
model_2 = graphlab.linear_regression.create(poly_data_2,target='price', features = poly_feature_2,
                                              validation_set=None,verbose=False, l2_penalty=l2_small_penalty)
print model_2.get('coefficients')


# In[164]:

poly_data_3 = polynomial_sframe(set_3['sqft_living'],15)
poly_feature_3 = poly_data_3.column_names()
poly_data_3['price'] = set_3['price']
model_3 = graphlab.linear_regression.create(poly_data_3,target='price', features = poly_feature_3,
                                              validation_set=None,verbose=False, l2_penalty=l2_small_penalty)
print model_3.get('coefficients')


# In[165]:

poly_data_4 = polynomial_sframe(set_4['sqft_living'],15)
poly_feature_4 = poly_data_4.column_names()
poly_data_4['price'] = set_4['price']
model_4 = graphlab.linear_regression.create(poly_data_4,target='price', features = poly_feature_4,
                                              validation_set=None,verbose=False, l2_penalty=l2_small_penalty)
print model_4.get('coefficients')


# The four curves should differ from one another a lot, as should the coefficients you learned.
# 
# ***QUIZ QUESTION:  For the models learned in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?***  (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)

# # Ridge regression comes to rescue

# Generally, whenever we see weights change so much in response to change in data, we believe the variance of our estimate to be large. Ridge regression aims to address this issue by penalizing "large" weights. (Weights of `model15` looked quite small, but they are not that small because 'sqft_living' input is in the order of thousands.) In GraphLab Create, adding an L2 penalty is a matter of adding an extra argument.
# 
# With the extra argument `l2_penalty=1e5`, fit a 15th-order polynomial model on `set_1`, `set_2`, `set_3`, and `set_4`. Other than the extra parameter, the code should be the same as the experiment above. Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set = None` in this call.

# In[166]:

l2_large_penalty=1.23e2


# In[167]:

poly_data_1_1 = polynomial_sframe(set_1['sqft_living'],15)
poly_feature_1_1 = poly_data_1_1.column_names()
poly_data_1_1['price'] = set_1['price']
model_1_1 = graphlab.linear_regression.create(poly_data_1_1,target='price', features = poly_feature_1_1,
                                              validation_set=None,verbose=False, 
                                              l2_penalty=l2_large_penalty)
print model_1_1.get('coefficients')


# In[168]:

poly_data_2_2 = polynomial_sframe(set_2['sqft_living'],15)
poly_feature_2_2 = poly_data_2_2.column_names()
poly_data_2_2['price'] = set_2['price']
model_2_2 = graphlab.linear_regression.create(poly_data_2_2,target='price', features = poly_feature_2_2,
                                              validation_set=None,verbose=False, 
                                              l2_penalty=l2_large_penalty)
print model_2_2.get('coefficients')


# In[169]:

poly_data_3_3 = polynomial_sframe(set_3['sqft_living'],15)
poly_feature_3_3 = poly_data_3_3.column_names()
poly_data_3_3['price'] = set_3['price']
model_3_3 = graphlab.linear_regression.create(poly_data_3_3,target='price', features = poly_feature_3_3,
                                              validation_set=None,verbose=False, 
                                              l2_penalty=l2_large_penalty)
print model_3_3.get('coefficients')


# In[170]:

poly_data_4_4 = polynomial_sframe(set_4['sqft_living'],15)
poly_feature_4_4 = poly_data_4_4.column_names()
poly_data_4_4['price'] = set_4['price']
model_4_4 = graphlab.linear_regression.create(poly_data_4_4,target='price', features = poly_feature_2_2,
                                              validation_set=None,verbose=False, 
                                              l2_penalty=l2_large_penalty)
print model_4_4.get('coefficients')


# These curves should vary a lot less, now that you introduced regularization.
# 
# ***QUIZ QUESTION:  For the models learned with regularization in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?*** (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)

# # Selecting an L2 penalty via cross-validation

# Just like the polynomial degree, the L2 penalty is a "magic" parameter we need to select. We could use the validation set approach as we did in the last module, but that approach has a major disadvantage: it leaves fewer observations available for training. **Cross-validation** seeks to overcome this issue by using all of the training set in a smart way.
# 
# We will implement a kind of cross-validation called **k-fold cross-validation**. The method gets its name because it involves dividing the training set into k segments of roughtly equal size. Similar to the validation set method, we measure the validation error with one of the segments designated as the validation set. The major difference is that we repeat the process k times as follows:
# 
# Set aside segment 0 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
# Set aside segment 1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
# ...<br>
# Set aside segment k-1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set
# 
# After this process, we compute the average of the k validation errors, and use it as an estimate of the generalization error. Notice that  all observations are used for both training and validation, as we iterate over segments of data. 
# 
# To estimate the generalization error well, it is crucial to shuffle the training data before dividing them into segments. GraphLab Create has a utility function for shuffling a given SFrame. We reserve 10% of the data as the test set and shuffle the remainder:

# In[171]:

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=0)
test_shuffled = graphlab.toolkits.cross_validation.shuffle(test, random_seed=0)


# Once the data is shuffled, we divide it into equal segments. Each segment should receive `n/k` elements, where `n` is the number of observations in the training set and `k` is the number of segments. Since the segment 0 starts at index 0 and contains `n/k` elements, it ends at index `(n/k)-1`. The segment 1 starts where the segment 0 left off, at index `(n/k)`. With `n/k` elements, the segment 1 ends at index `(n*2/k)-1`. Continuing in this fashion, we deduce that the segment `i` starts at index `(n*i/k)` and ends at `(n*(i+1)/k)-1`.

# With this pattern in mind, we write a short loop that prints the starting and ending indices of each segment, just to make sure you are getting the splits right.

# In[172]:

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print i, (start, end)


# Let us familiarize ourselves with array slicing with SFrame. To extract a continuous slice from an SFrame, use colon in square brackets. For instance, the following cell extracts rows 0 to 9 of `train_valid_shuffled`. Notice that the first index (0) is included in the slice but the last index (10) is omitted.

# In[173]:

train_valid_shuffled[0:10] # rows 0 to 9


# Now let us extract individual segments with array slicing. Consider the scenario where we group the houses in the `train_valid_shuffled` dataframe into k=10 segments of roughly equal size, with starting and ending indices computed as above.
# Extract the fourth segment (segment 3) and assign it to a variable called `validation4`.

# In[174]:

validation4 = train_valid_shuffled[5818:7758]


# To verify that we have the right elements extracted, run the following cell, which computes the average price of the fourth segment. When rounded to nearest whole number, the average should be $549,213.

# In[175]:

print int(round(validation4['price'].mean(), 0))


# After designating one of the k segments as the validation set, we train a model using the rest of the data. To choose the remainder, we slice (0:start) and (end+1:n) of the data and paste them together. SFrame has `append()` method that pastes together two disjoint sets of rows originating from a common dataset. For instance, the following cell pastes together the first and last two rows of the `train_valid_shuffled` dataframe.

# In[176]:

n = len(train_valid_shuffled)
first_two = train_valid_shuffled[0:2]
last_two = train_valid_shuffled[n-2:n]
print first_two.append(last_two)


# Extract the remainder of the data after *excluding* fourth segment (segment 3) and assign the subset to `train4`.

# In[177]:

part_1 = train_valid_shuffled[0:5818]
part_2 = train_valid_shuffled[7758:19396]
train4 = part_1.append(part_2)


# To verify that we have the right elements extracted, run the following cell, which computes the average price of the data with fourth segment excluded. When rounded to nearest whole number, the average should be $539,008.

# In[178]:

print int(round(train4['price'].mean(), 0))


# Now we are ready to implement k-fold cross-validation. Write a function that computes k validation errors by designating each of the k segments as the validation set. It accepts as parameters (i) `k`, (ii) `l2_penalty`, (iii) dataframe, (iv) name of output column (e.g. `price`) and (v) list of feature names. The function returns the average validation error using k segments as validation sets.
# 
# * For each i in [0, 1, ..., k-1]:
#   * Compute starting and ending indices of segment i and call 'start' and 'end'
#   * Form validation set by taking a slice (start:end+1) from the data.
#   * Form training set by appending slice (end+1:n) to the end of slice (0:start).
#   * Train a linear model using training set just formed, with a given l2_penalty
#   * Compute validation error using validation set just formed

# In[185]:

def k_fold_cross_validation(k, l2_penalty, data, features_list):
    n = len(data)
    error = 0
    Residual_sum_square = 0
    for i in xrange(k): #Xrange create list of k for iteration
        #Slice segment into validation
        start = (n*i)/k
        end = (n*(i+1)/k-1)
        validation_set = data[start:end+1]
        #training set
        train_part1 = data[0:start]
        train_part2 = data[end+1:n]
        train_set = train_part1.append(train_part2)
        #Modeling with l2_penalty
        valid_model = graphlab.linear_regression.create(train_set,target='price', 
                                                        features = features_list,
                                              validation_set=None,verbose=False, 
                                              l2_penalty=l2_penalty)
        #print valid_model
        #print validation_set
        predictions = valid_model.predict(validation_set)
        error_set = RSS_function(predictions, validation_set['price'])
        Residual_sum_square = Residual_sum_square + error_set
        total_error = (Residual_sum_square/k)
    return total_error
        
def RSS_function(predictions, output):
    residual = output - predictions
    residual_square = residual ** 2 
    RSS = residual_square.sum()
    return RSS
        
        


# Once we have a function to compute the average validation error for a model, we can write a loop to find the model that minimizes the average validation error. Write a loop that does the following:
# * We will again be aiming to fit a 15th-order polynomial model using the `sqft_living` input
# * For `l2_penalty` in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] (to get this in Python, you can use this Numpy function: `np.logspace(1, 7, num=13)`.)
#     * Run 10-fold cross-validation with `l2_penalty`
# * Report which L2 penalty produced the lowest average validation error.
# 
# Note: since the degree of the polynomial is now fixed to 15, to make things faster, you should generate polynomial features in advance and re-use them throughout the loop. Make sure to use `train_valid_shuffled` when generating polynomial features!

# In[186]:

import numpy as np #For l2_penalty


# In[187]:

#for l2_pen in logspace:
test_set_error = k_fold_cross_validation(10, 316.227766017, test_data, test_feature)
print test_set_error
#  print l2_pen
# print test_set_error
print (np.logspace(1,7,num=13))


# In[182]:

test_data = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
test_feature = test_data.column_names()
test_data['price'] = train_valid_shuffled['price']
logspace = np.logspace(1, 7, num=13)
lowest_penalty = 1e20;
for i in (np.logspace(1, 7, num=13)):
    test_set_error = k_fold_cross_validation(10, i, test_data, test_feature)
    print 'Current l2_penalty is', i
    print 'The RSS with', i, 'is', test_set_error
#    if(test_set_error<lowest_penalty):
#        lowest_penalty = test_set_error


# ***QUIZ QUESTIONS:  What is the best value for the L2 penalty according to 10-fold validation?***

# You may find it useful to plot the k-fold cross-validation errors you have obtained to better understand the behavior of the method.  

# In[183]:

# Plot the l2_penalty values in the x axis and the cross-validation error in the y axis.
# Using plt.xscale('log') will make your plot more intuitive.
#x = np.logspace(1, 7, num=13)
#plt.xscale('log')


# Once you found the best value for the L2 penalty using cross-validation, it is important to retrain a final model on all of the training data using this value of `l2_penalty`.  This way, your final model will be trained on the entire dataset.

# In[189]:

#Retrain a nice model
data_set = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
data_feature = data_set.column_names()
data_set['price']= train_valid_shuffled['price'] #Add price column
final_model = graphlab.linear_regression.create(data_set,target='price', 
                                                        features = data_feature,
                                              validation_set=None,verbose=False, 
                                              l2_penalty=1000)
print final_model
print 'RSS of final_model on trainning data',k_fold_cross_validation(10,100,data_set,data_feature)


# ***QUIZ QUESTION: Using the best L2 penalty found above, train a model using all training data. What is the RSS on the TEST data of the model you learn with this L2 penalty? ***

# In[192]:

#Compute RSS on the test 
#Model created on validation data, to visualize the l2_penalty
test_data_set = polynomial_sframe(test_shuffled['sqft_living'], 15)
test_data_feature = test_data_set.column_names()
test_data_set['price']= test_shuffled['price'] #Add price column
#final_test_model = graphlab.linear_regression.create(test_data_set,target='price', 
                                                        #features = test_data_feature,
                                                      #validation_set=None,verbose=False, 
                                                      #l2_penalty=1000)
#print final_test_model
print 'RSS of final_model on trainning data',k_fold_cross_validation(10,100,test_data_set,test_data_feature)


# In[ ]:



