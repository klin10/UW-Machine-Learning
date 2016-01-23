import numpy as np
#Set gradient to Zero approach
#It generally take longer than gradient descent 
#as gradient HAVE to approach zero 
list_input_feature = [0,1,2,3,4]
list_output = [1,3,7,13,21]
input_feature = np.asarray(list_input_feature)
output = np.asarray(list_output)

def close_form_regression(input_feature, output):
    n = len(input_feature)
    print n
    sum_output = (output.sum())
    print sum_output
    sum_input = (input_feature.sum())
    print sum_input
    product_output_input = (input_feature) * (output)
    dot_product = (product_output_input.sum())
    print "dot_product is %d:" %(dot_product) 
    square_input = (input_feature) ** 2 
    square_input_sum = ((square_input).sum())
    #Calculate for slope
    numerator = (dot_product) - ((sum_input * sum_output)/n)
    print numerator
    denomenator = (square_input_sum) - ((sum_input ** 2)/n)
    print denomenator
    slope = numerator / denomenator
    intercept = (((sum_output / n)-((slope * sum_input)/n)))
    print slope
    print intercept
    return (intercept, slope)
    
close_form_regression(input_feature, output)
    