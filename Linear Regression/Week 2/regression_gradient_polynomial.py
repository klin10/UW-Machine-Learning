def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the ‘features’ list into the SFrame ‘features_sframe’

    # this will convert the features_sframe into a numpy matrix with GraphLab Create >= 1.7!!
    features_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the target to the variable ‘output_sarray’
    
    # this will convert the SArray into a numpy array:
    output_array = output_sarray.to_numpy() # GraphLab Create>= 1.7!!
    return(features_matrix, output_array)
    
#Helper function to predict the outcome
def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return(predictions)
#Helper function to compute derivative of the feature gradient    
def feature_derivative(errors, feature):
    derivative = 2(np.dot(feature,errors))
    return(derivative)
    
#Feature by feature update
#Iterates through the feature matrix
#to descent down each weight in the feature matrix
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        prediction = predict_outcome(feature_matrix, initial_weights)
        # compute the errors as predictions - output:
        errors = prediction - output
        gradient_sum_squares = 0 # initialize the gradient
        #update each weight individually:
        for i in range(len(weights)):
            # compute the derivative for weight[i]:
            # Core formula for the feature by feature update
            derivative = feature_derivative(errors,feature_matrix[:, i])
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares = square_derivative + (derivative * derivative)
            # Adjust the weight coeffiecent by using the derivative
            weight[i] = weight[i] - step_size * derivative
        gradient_magnitude = sqrt(gradient_sum_squares)
        #Convergence check
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)