import numpy as np
#Set gradient to Zero approach
#It generally take longer than gradient descent 
#as gradient HAVE to approach zero 
list_input_feature = [0,1,2,3,4]
list_output = [1,3,7,13,21]
#Input feature and output is a column of data
#The training column will train a linear regression model
#To output the best fitted line
# Y= AX + B
#Assume input_feature and output are already array
def gradient_linear_regression(input_feature, output):
    [your code here]
    #Starter variables to defind the regression
    slope = 0
    intercept = 0
    step_size = 0.05
    tolerance = 0.01
    #Test for debugging
    magnitude = 10
    #Predict based on current slope and intercept
    #Iterate through and descent until it reaches tolerance
    while (magnitude > tolerance):
        #Make prediction
        prediction = (input_feature * slope) + intercept
        prediction_errors = prediction - output
        #Compute for new intercept
        intercept_derivative = prediction_errors.sum([])
        intercept_adjustment = step_size * intercept_derivative
        intercept = intercept - adjustment
        #Computer for new slope
        #Inner product
        slope_derivative = np.dot(prediction_errors,input_feature)
        slope_adjustment = step_size * slope_derivative
        slope = slope - slope_adjustment
        #Compute for magnitude of the gradient
        gradient = ((intercept_derivative**2 +(slope_derivative**2)) **(1/2.0))
return(intercept, slope)

#Set gradient to Zero approach
#It generally take longer than gradient descent 
#as gradient HAVE to approach zero 
def close_form_regression(input_feature, output):
    n = input_feature.size()
    sum_output = (output.sum())
    sum_input = (input_feature.sum())
    product_output_input = (input_feature) * (output)
    dot_product = (product_output_input.sum())
    square_input = (input_feature) ** 2 
    square_input_sum = ((square_input).sum())
    #Calculate for slope
    numerator = (dot_product) * ((sum_input * sum_output)/n)
    denomenator = (square_input_sum) - (square_input_sum/n)
    slope = numerator / denomenator
    
    
return (intercept, slope)


#This function uses intecept and slope to predict price
def get_regression_predictions(input_feature, intercept, slope)
    [your code here]
    #Y= AX + B
    predicted_output = float((input_feature * slope) + (intercept))
return(predicted_output)


#RSS is the sum of the square prediction errors
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    [your code here]
    prediction = get_regression_predictions(input_feature, intercept, slope)
    RSS = (prediction - output) ** 2
return(RSS)

#Inverse by doing the operation in get prediction
def inverse_regression_predictions(output, intercept, slope):
    [your code here]
    estimated_input = float((output - intercept) / slope )
return(estimated_input)