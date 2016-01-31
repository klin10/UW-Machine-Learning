def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features # this is how you combine two lists
    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features] #added
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = data_sframe['price']
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy()
    return(feature_matrix, output_array)
    
    
#Helper function to predict the outcome
def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return(predictions)
#Helper function to compute derivative of the feature gradient    
def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative = 2*(np.dot(feature,errors))
    return(derivative)
    
    
#Feature by feature update
#Iterates through the feature matrix
#to descent down each weight in the feature matrix
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        prediction = predict_outcome(feature_matrix, weights)
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
            weight[i] = weight[i] - (step_size * derivative)
        gradient_magnitude = sqrt(gradient_sum_squares)
        #Convergence check
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)
    
    
    def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        prediction = predict_output(feature_matrix, initial_weights)
        # compute the errors as predictions - output:
        errors = prediction - output
        gradient_sum_squares = 0 # initialize the gradient
        #update each weight individually:
        for i in range(len(weights)):
            # compute the derivative for weight[i]:
            # Core formula for the feature by feature update
            derivative = feature_derivative(errors,feature_matrix[:, i])
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares = gradient_sum_squares + (derivative * derivative)
            # Adjust the weight coeffiecent by using the derivative
            weights[i] = weights[i] - (step_size * derivative)
        gradient_magnitude = sqrt(gradient_sum_squares)
        #Convergence check
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)