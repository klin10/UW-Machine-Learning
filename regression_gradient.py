#Test Data
import numpy as np
#Set gradient to Zero approach
#It generally take longer than gradient descent 
#as gradient HAVE to approach zero 
list_input_feature = [0,1,2,3,4]
list_output = [1,3,7,13,21]
input_feature = np.asarray(list_input_feature)
output = np.asarray(list_output)

def gradient_linear_regression(input_feature, output):
    #Starter variables to defind the regression
    slope = 0
    intercept = 0
    step_size = 0.05
    tolerance = 0.01
    #Test for debugging
    coverge = True;
    #count = 0
    #Predict based on current slope and intercept
    #Iterate through and descent until it reaches tolerance
    while coverge:
        #Make prediction
        prediction = (input_feature * slope) + intercept
        prediction_errors = (prediction - output)
        #Compute for new intercept
        intercept_derivative = prediction_errors.sum()
        intercept_adjustment = step_size * intercept_derivative
        intercept = intercept - intercept_adjustment
        print intercept
        #Computer for new slope
        #Inner product
        slope_derivative = np.dot(prediction_errors,input_feature)
        slope_adjustment = step_size * slope_derivative
        slope = slope - slope_adjustment
        #print slope
        #print count
        #count +=1
        #Compute for magnitude of the gradient
        magnitude = ((intercept_derivative**2 +(slope_derivative**2)) **(1/2.0))
        if(magnitude < tolerance):
            coverge = False
    return(intercept, slope)
    
gradient_linear_regression(input_feature,output)