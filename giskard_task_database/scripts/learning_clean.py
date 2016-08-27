import numpy as np
import myutil
import matplotlib.pyplot as plt
##For visualizing trees:
from sklearn.externals.six import StringIO
################################
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn import cross_validation
from pymongo import MongoClient
# from sklearn.metrics import explained_variance_score #

class ActionLearner:
    """Class for learning model(s) from demonstrations in Gazebo"""
    OUTLIER_METHOD = 'none' #'none', 'MCDEllipticEnv', 'MCDchi', 'OCSVMiqr' 'iqr' 'std'
    OUTLIER_CONTAMINATION = 0.2 #For MCDEllipticEnv
    SUPPORTFRACTION = 0.8 #For MCDchi
    OUTLIER_IQRRANGE = 1.5 #For iqr and for OCSVMiqr
    OUTLIER_STDRANGE = 2 #For std
    MOTION_PHASES = ["moveabovepan", "tiltbottle", "tiltback"]
    TRAINING_PERC = 1.0 #0.9 #how much of the available data is used for training (the rest is put aside as test set)
    
    OUT_FIGS = "/home/yfang/Projects/summerpas/generated_figures/" #location where produces and sames figures are stored

    def __init__(self, dbname, paramcollname, perfcollname, predictors, dependents, learningmodel):
        """
        Prepare ActionModel based on data read from mongodb
        
        dbname: which database the demonstration data is located in
        paramcollname: which collection the parameter of the demonstration data is located in
        perfcollname: which collection the performance of the demonstration data is located in
        predictors: list of strings with the names of the variables (columnnames) that are used to predict the other variables
        dependents: list of strings with the names of the variables (columnnames) that are being predicted. If None, assumes that all numeric variables that are not predictors should be predicted
        learning model: which type of model should be used for learning
        datatype: for what we want to learn a model, "parameters" or "performance"
        """   
        #Open appropriate Mongo databases and collections  
        client = MongoClient('localhost', 27018) #open connection
        db = client[dbname] #get database
        paramcoll = db[paramcollname] #access the action model collection (will create one if it doesn't exist yet)
        perfcoll = db[perfcollname]
        
        #Read data into dictionaries
        overall_doc = {}
        overall_doc["phases"]={} #have to initialize this dictionary
        #read in param data (add to empty dictionary)
        paramdata, _, phase_names = self.read_param_data(overall_doc, perfcoll, paramcoll, incl_spilled=False)
        #read in perf data (add to paramdata)
        data, _, _ = self.read_perf_data(paramdata, perfcoll, incl_spilled=False)
        self.add_features_to_data(data)
        
        self.datadict=data #data stored in arrays that are named using keys. WARNING! data has not been converted to float yet
        ##create numpy array for the (dict) data read from MongoDB
        alldata_array, alldata_names = self.dictdata_to_dictarrays(data, phase_names)

        #Get data separated into dependents and predictors (and unused variables)
        predictors_data, dependents_data, predictor_names, dependent_names, notused_keys = self.separate_datavariables(alldata_array, alldata_names, predictors, dependents)
        print "These variables were not used as predictor or dependent: {}".format(str(notused_keys))
        
        #put data into their respective matrices
        predictors_data = self.transpose_dictdata(predictors_data) #each row is all the values for that fieldname, so need to transpose so that each row is one observation
        dependents_data = self.transpose_dictdata(dependents_data)
        self.all_data = self.transpose_dictdata(alldata_array)
        self.all_datanames = alldata_names

        #Create model
        self.actionmodel = ActionModel(predictors_data, predictor_names, dependents_data, dependent_names, learningmodel)
        self.evaluationmodel = self.actionmodel.evaluate_all_models(self.actionmodel.learned_models, test_xdata=np.array([]), test_ydata=np.array([]), printsome=False, printall=False)
    
    def read_param_data(self, in_doc, perfcoll, paramcoll, incl_spilled=True):
            """
            Read data from parameter MongoDB collection and combine into a single dictionary
            If incl_spilled==False, only takes into account the episodes where pouring occurred and no spilling occurred. Default is set to False.
            
            WARNING: currently has no check for existing keys inside of the "phases" field; assumes param and perf collections do NOT have any same-named fields here
            """
            overall_doc = in_doc
            #Store the existing keys so that we don't write duplicate data (for example: both paramcoll and perfcoll contain the field subject_id; when both read to the same dictionary, this will result in an array twice the appropriate size
            existing_keys = overall_doc.keys()
            phase_names = []
            counter = 0
            for iperfdoc in perfcoll.find(): #go through all the documents
                if iperfdoc['pouring'] and (iperfdoc['spilled']==0 or incl_spilled): #if episodes contains pouring and either no spilling or we want spilling to be included
                    iparamdoc = paramcoll.find_one({"episode_id": iperfdoc['episode_id']})
#                     overall_doc.setdefault("spill", []).append(iperfdoc['spilled']) #TODO: there is some weird effect here that causes spill to be numeric and spilled to have string values
                    for phase in iparamdoc['phases']: #for each phase in the episode (==document)
                        if not (str(phase) in overall_doc["phases"]): #check whether this part of the dictionary already exists
                            overall_doc["phases"][str(phase)]={}
                            phase_names.append(str(phase))
                        for phasekey in iparamdoc['phases'][phase]:
                            #learned to append to dict values here: http://stackoverflow.com/questions/3199171/append-multiple-values-for-one-key-in-python-dictionary
                            overall_doc["phases"][str(phase)].setdefault(str(phasekey),[]).append(str(iparamdoc['phases'][phase][phasekey]))
                    
                    for ikey in iparamdoc:
                        if not(ikey=='phases' or ikey=='_id') and not(ikey in existing_keys): #Not interested in storing objectid, mongodb sepecific. Phases were already added above. So ignore these two keys and copy rest
                            overall_doc.setdefault(str(ikey),[]).append(str(iparamdoc[ikey]))
                    
                    for ikey in iperfdoc:
                        if not(ikey=='phases' or ikey=='_id') and not(ikey in iparamdoc): #Not interested in storing objectid, mongodb sepecific. Phases were already added above. So ignore these two keys and copy rest
                            overall_doc.setdefault(str(ikey),[]).append(str(iperfdoc[ikey]))      
                    counter += 1  
                ##WARNING! WHAT IF THERE ARE NAN VALS SOMEWHERE IN A SPECIFIC SUBJECT? I'd have to go back and delete it?!
#                 if docval and not('nan' in docval.values()): #if the dictionary is not empty 
#                     allvals.append(docval)
#                 elif 'nan' in docval.values():
#                     print str(iperfdoc['episode_id']) + " contains a NaN value and will be removed from the parameter set: " + str(docval)

            #Obsolete (but still interesting piece of code for handling dictionaries..)
            #merge dictionaries to contain a list of values if there are multiple entries for one key
#             data={k:[d.get(k) for d in allvals] for k in {k for d in allvals for k in d}} #Got this from http://stackoverflow.com/questions/2365921/merging-python-dictionaries, preconstructed :D
            print "read in " + str(counter) + " param collections to data"
            n_obs = len(overall_doc)
            return overall_doc, n_obs, phase_names
        
    def read_perf_data(self, in_doc, perfcoll, incl_spilled=True):
        """
        Read data from performance collection and combine into a single dictionary
        If incl_spilled==False, only takes into account the episodes where pouring occurred and no spilling occurred. Default is set to True.
        
        WARNING: currently has no check for existing keys inside of the "phases" field; assumes param and perf collections do NOT have any same-named fields here
        """
        #Store the existing keys so that we don't write duplicate data (for example: both paramcoll and perfcoll contain the field subject_id; when both read to the same dictionary, this will result in an array twice the appropriate size
        overall_doc = in_doc
        existing_keys = overall_doc.keys()
        phase_names = []
        counter = 0
        for iperfdoc in perfcoll.find(): #go through all the documents
            if iperfdoc['pouring'] and (iperfdoc['spilled']==0 or incl_spilled): #if episodes contains pouring and either no spilling or we want spilling to be included
                for phase in iperfdoc['phases']: #for each phase in the episode (==document)
                    if not (str(phase) in overall_doc["phases"]): #check whether this part of the dictionary already exists
                        overall_doc["phases"][str(phase)]={}
                        phase_names.append(str(phase))
                    for phasekey in iperfdoc['phases'][phase]:
                        overall_doc["phases"][str(phase)].setdefault(str(phasekey),[]).append(str(iperfdoc['phases'][phase][phasekey]))
                
                for ikey in iperfdoc:
                    if not(ikey=='phases' or ikey=='_id') and not(ikey in existing_keys): #Not interested in storing objectid, mongodb sepecific. Phases were already added above. So ignore these two keys and copy rest
                        overall_doc.setdefault(str(ikey),[]).append(str(iperfdoc[ikey]))      
                counter += 1  
        
        print "read in " + str(counter) + " perf collections to data"
        n_obs = len(overall_doc)
        return overall_doc, n_obs, phase_names
    
    def add_features_to_data(self, in_doc):
        """Adds features to the existing (read-in) data. 
        Currently used to add features poured_m3 and ontarget_m3
        
        NOTE: the function inserts the data directly into the argument given and therefore does not return anything (return None)
        """
        #Compute m3 per one unit
        #TODO: unit_volume assumes this is a sphere, need to put in a conditional for sauce
        unit_volume = np.power((np.array(in_doc["unit_size"], dtype=float)*2),3)
        
        #Get "poured" data array from dictionary
        poured_arr = np.array(in_doc["poured"], dtype=float)
        pouredm3_arr = poured_arr * unit_volume
        #put into the dictionary
        in_doc["pouredm3"]=pouredm3_arr
        print "added pouredm3 feature to data"
        
        #Get "ontarget" data array from dictionary
        ontarget_arr = np.array(in_doc["ontarget"], dtype=float)
        ontargetm3_arr = ontarget_arr * unit_volume
        #put into the dictionary
        in_doc["ontargetm3"]=ontargetm3_arr
        print "added ontargetm3 feature to data"
        
    @staticmethod
    def dictdata_to_dictarrays(data, phase_names):
        alldata_dict = {} #We will construct a dictionary with per key containing the entire matrix for data for one phase. Overall data such as subject, pouring, fill, spill, etc. will be copied and thus redundant
        alldata_names = {}
        for phase in phase_names:
            alldata_dict[phase] = []
            alldata_names[phase]=[]
             
        for key in data:
            #if this is phase data, will need to enter and create separate matrices
            if key=='phases':
                for phase in data[key]: #each phase contains for all parameters a list of values that were extracted from the demonstrations
                    for param in data[key][phase]:
                        if all(myutil.convertableToFloat(i) for i in data[key][phase][param]): #if all the data for this variable are valid floats (because we cannot calculate with other types of columns for now)
                            alldata_dict[phase].append(np.array(data[key][phase][param], dtype=float))
#                             print phase + "-" + param + " was " + str(len(data[key][phase][param]))
                            alldata_names[phase].append(param)
                        else:
                            print "Tried to use " + param + " but not all inputs were floats"
            elif all(myutil.convertableToFloat(i) for i in data[key]): #all other usable data will be copied into the phase data
                for phase in phase_names:
                    alldata_dict[phase].append(np.array(data[key], dtype=float))
#                     print key + " was " + str(len(data[key]))
                    alldata_names[phase].append(key)
        
        #convert the lists to np arrays
        for phase in alldata_dict:
            alldata_dict[phase]=np.array(alldata_dict[phase], dtype=float)
              
#         target = open("testdataprocessing.txt", 'w')
#         target.write(str(alldata_dict))
#         target.close()
        return alldata_dict, alldata_names
      
    def separate_datavariables(self, data, data_names, predictors, dependents):
        """
        Separates data into separate matrices (numpy arrays) for predictors and dependent variables.
        
        Parameters
        ----------
        data : dict
               Contains per key the variable matrix for one motion phase. The matrix contains per row a float array that are all the values for some constraint X in that phase.
        data_names : dict
               Contains per key a list with the names of the variables for one motion phase.
        predictors : dict
                     Contains per key a list of strings with the names of the variables that function as predictor for that phase. These names should match what's present in data_names parameter
        dependents : list
                     Contains per key a list of strings with the names of the variables that function as dependent variables for that phase. These names should match what's present in data_names parameter
        """
        predictors_data = {} #We will construct a dictionary with per key containing the entire matrix for data for one phase. Overall data such as subject, pouring, fill, spill, etc. will be copied and thus redundant
        dependents_data = {}
        predictor_names = {}
        dependent_names = {}
        not_used_vars = {}
        for phase in data:
            predictor_names[phase] = []
            dependent_names[phase]=[]        
            not_used_vars[phase]=[]
            
        for phase in data: #for each phase
            #look up the indices of the predictors and put them into the matrix
            pred_tmp = [] #temporarily storing data until can convert to np.array
            dep_tmp = []
            for pred in predictors[phase]:
                if pred in data_names[phase]:
                    idx_pred = data_names[phase].index(pred)
                    data_pred = data[phase][idx_pred]
                    pred_tmp.append(data_pred)
                    predictor_names[phase].append(pred)
#                     print phase + "-" + pred + ": "
#                     print data_pred
                else:
                    print "WARNING: predictor " + pred + " does not exist in phase " + phase
            for dep in dependents[phase]:
                if dep in data_names[phase]:
                    idx_dep = data_names[phase].index(dep)
                    data_dep = data[phase][idx_dep]
                    dep_tmp.append(data_dep)
                    dependent_names[phase].append(dep)
                else:
                    print "WARNING: dependent " + dep + " does not exist in phase " + phase
            for var in data_names[phase]: #check which vars are not used at all
                if var not in predictors[phase] and var not in dependents[phase]:
                    not_used_vars[phase].append(var)
            predictors_data[phase] = np.array(pred_tmp, dtype=float)
            dependents_data[phase] = np.array(dep_tmp, dtype=float)            
        return predictors_data, dependents_data, predictor_names, dependent_names, not_used_vars
    
    @staticmethod
    def transpose_dictdata(data):
        """ Transposes the datamatrices within the dictionary-style data. 
        The output of reading in the data and transforming the data up until separate_datavariables
        results in matrices for which each row contains all the values for a particular variable.
        This is to transpose it so that each row is one observation (for the learning algorithm)
        
        Parameters
        ----------
        data : dict
               Contains per key a data patrix.
        """
        trans_data ={}
        for phase in data:
            transposed_data = np.transpose(data[phase])
            trans_data[phase] = transposed_data
            
        return trans_data

    def estimate_constraint_range(self, evaluation_model, newx, fitthreshold, plotreplacement=False, verbose=False):
        """
        Given a predicted_vals for a constraint value, returns constraint range using errormetric if the associative power exceeds 
        fitthreshold for that variable, otherwise returns constraint range using q1 and q3 
        
        newx: numpy array representing one observation of predictor variables
        
        Returns: array containing 2 multidimensional arrays. The first array are the minimum values, the second array are the maximum. They are multidimensional because in principle can predict multiple observations to be predicted at the same time (so there would be an array per set of x)
        
        Note: learned_models is a dictionary with keys 'model errormetric modelxdata modelydata xnames ynames'
        """
        result = {}
        for phase in self.actionmodel.learned_models:
            lmodel = self.actionmodel.learned_models[phase]
            
            #Extract x-array from newx dictionary to match the learned order of predictors
            newx_list = [newx[key] for key in self.actionmodel.predictor_names[phase]]
            predicted_vals = lmodel['model'].predict(newx_list)
            assert len(predicted_vals) == 1
            predicted_range = [predicted_vals-lmodel['errormetric'], predicted_vals+lmodel['errormetric']] #create array of 2 arrays, where the first one contains the min values and the other the max values 
            result[phase] = np.array(predicted_range)
            
            #Plot graph to illustrate the ranges per y value
            if plotreplacement:
                fig, ax = plt.subplots(1,1)
                xaxis=xrange(0,len(predicted_vals[0]))
                ax.plot(xaxis, predicted_range[0][0],c='black', label="predictions")
                ax.plot(xaxis, predicted_range[1][0],c='black')
                ax.fill_between(xaxis, predicted_range[0][0], predicted_range[1][0], color='black', alpha=0.3)
            
            #Replace y-values of dependent variables that could not be predicted above fitthreshold by q1 and q3 for min-max values.
            for ix in range(0,len(predicted_vals)): #in principle we could predict the values for multiple x's (e.g. multiple observations) at the same time, but that code was not tested
                cur_y = predicted_vals[ix]
                for iy in range(0, len(cur_y)): #for every predicted variable, check whether we should use the predicted_vals or the median
    #                 explained_variance = np.mean(evaluation_model['cv']['r2_adj_sep'][lmodel['ynames'][iy]])
                    explained_variance = evaluation_model[phase]['train']['oob_r2adj'][lmodel['ynames'][iy]]
                    if explained_variance < fitthreshold: #If not a moderate amount of variance in this variable is explained by the predictors
                        q1, q3 = np.percentile(lmodel['modelydata'][:,iy], [25, 75])
                        median = np.percentile(lmodel['modelydata'][:,iy], 50)
                        rmse = lmodel['errormetric'][iy]
                        predicted_range[0][ix][iy] = q1
                        predicted_range[1][ix][iy] = q3
                        if verbose:
                            print "in " + phase + " replacing " + lmodel['ynames'][iy]+": " + str(explained_variance)
                    
            #Add adapted y values to graph to illustrate the ranges after replacements with quartiles
            if plotreplacement:
                ax.plot(xaxis, predicted_range[0][0],c='green', label="corrected predictions")
                ax.plot(xaxis, predicted_range[1][0],c='green')
                ax.fill_between(xaxis, predicted_range[0][0], predicted_range[1][0], color='green', alpha=0.3)
                ax.set_xlabel("dependent variables")
                ax.set_ylabel("values")
                ax.legend()
                plt.show()     
        return result
        
###############################################

class ActionModel:
    """Class for learning model(s) from demonstrations in Gazebo"""
    
    OUTLIER_METHOD = 'none' #'none', 'MCDEllipticEnv', 'MCDchi', 'OCSVMiqr' 'iqr' 'std'
    OUTLIER_CONTAMINATION = 0.2 #For MCDEllipticEnv
    SUPPORTFRACTION = 0.8 #For MCDchi
    OUTLIER_IQRRANGE = 1.5 #For iqr and for OCSVMiqr
    OUTLIER_STDRANGE = 2 #For std
    ALPHA_RIDGE = 0.5 #alpha parameter for ridge regression
    DTREG_MAXDEPTH = 5 #max depth for decision tree regressor
    RFREG_NEST = 100 #number of trees for random forest regressor
    RFREG_NFEAT = None #max number of features for random forest regressor (scikit-learn advises n_features for regression problems)
    MOTION_PHASES = ["moveabovepan", "tiltbottle", "tiltback"]
    CV_FOLDS = 5 #how many folds for crossvalidation for estimating scores
    
    TRAINING_PERC = 1.0 #0.9 #how much of the available data is used for training (the rest is put aside as test set)
    
    OUT_FIGS = "/home/yfang/Projects/summerpas/generated_figures/" #location where produces and sames figures are stored

    def __init__(self, predictor_data, predictor_names, dependent_data, dependent_names, modeltype):
        """
        Construct a model of type [modeltype] based on predictor and dependent data
        
        predictor_data: dict
                        Each key contains for one motion phase a numpy 2D array with one observation in each row
        predictor_names: dict
                        For each motion phase contains a list with names of the variables matching columns of predictor_data
        dependent_data: dict
                        Each key contains for one motion phase a numpy 2D array with one observation in each row
        dependent_names: dict
                        For each motion phase contains a list with names of the variables matching columns of dependent_data
        modeltype: which type of model should be used to fit the data
        
        """
        assert predictor_data.keys() == dependent_data.keys() #sanity check. They were constructed from the same datamatrix. In our current design they should always contain the same motionphases
        
        self.predictor_data = predictor_data
        self.dependent_data = dependent_data
        self.predictor_names = predictor_names
        self.dependent_names = dependent_names
        self.modeltype = modeltype.lower()
        self.learned_models ={}
        
        for phase in predictor_data:
            #for each phase, learn model
            self.learned_models[phase] = self.learnModel(predictor_data[phase], dependent_data[phase], predictor_names[phase], dependent_names[phase], self.modeltype, verbose=False)
        
    @staticmethod
    def rmse(predictions, targets):
        """calculate root mean squared error of model"""
#         return np.sqrt(((predictions-targets) **2).mean()) //equivalent to function available below
        result = [] 
        targets = np.asarray(targets)
        
        #store rmse for every list in predictions and targets (could be multiple in case of multiple y)
        if isinstance(predictions[0], np.ndarray):
            for i in xrange(len(predictions[0])):
                ipred = predictions[:,i]
                itarget = targets[:,i]
                result.append(metrics.mean_squared_error(itarget, ipred)**0.5)
        else:
            result.append(metrics.mean_squared_error(targets, predictions)**0.5)
        
        return result
        
    def learnModel(self, xdata, ydata, xnames, ynames, modeltype, verbose=False):
        """creates a multivariate linear model (multiple predictors, multiple dependent variables) or 
        regression tree using the predictor(s) to predict all other numeric columns
                
        NOTE: all the models so far support multiple response variables EXCEPT 'gamr'. For the others, y contains all response variables implicitly.
        Returns dictionary containing model, errormetric, modelxdata, modelydata, xnames, ynames
        """
        #x is a 2d array (for possibly multiple xs) and y is also a 2d array (for possibly multiple ys)
        x = xdata.astype(np.float) #convert all elements to float. assumes that all colxs of interest are numeric, will throw error if contains a string
        y = ydata.astype(np.float)
        
        if(modeltype == 'olslr'):
            lmodel = linear_model.LinearRegression()
        elif(modeltype == 'rlr'): #NOTE: also tried RidgeCV but somehow alpha level selected is really high (10 out of sequence 0.1, 0.5, 1.0, 10, 20) and the residual sum of squares actually becomes slightly larger
            lmodel = linear_model.Ridge(alpha=self.ALPHA_RIDGE)
        elif(modeltype == 'dtreg'):
            lmodel = tree.DecisionTreeRegressor(max_depth=self.DTREG_MAXDEPTH)
        elif(modeltype=='rforestreg'):
            lmodel = RandomForestRegressor(n_estimators=self.RFREG_NEST, max_features=self.RFREG_NFEAT, oob_score=True)
        else:
            print "model " + modeltype + " not supported"
            return None
            
        #Fit model using entire traindata set
        lmodel.fit(x,y)
        y_pred = lmodel.predict(x)
        res_errormetric = ActionModel.rmse(y_pred, y)
            
        ##Print some extra information if desired
        if verbose:
            # The coefficients
            if modeltype=='olslr' or modeltype=='rlr':
                print('Coefficients: \n', lmodel.coef_)
            # The mean square error TODO: warning, since I didn't make separate test-train sets, error estimation may be off
            print("Residual sum of squares: %.4f" % np.mean((y_pred - y) ** 2))
            print "Root Mean Squared Error: %s" % str(res_errormetric)
     
        #Other variables are meant for keeping track of which training inputs the model was based on
        result = {'modeltype': modeltype, 'model': lmodel, 'errormetric':res_errormetric, 'modelxdata':x, 'modelydata':y, 'xnames':xnames, 'ynames':ynames}
        return result
    
    def useModel(self, xdata, verbose=False):
        """
        Uses the model to give a prediction given xdata. Note that for xdata, there should be one observation per row. If there is only
        one prediction, a single numpy array suffices.
        
        xdata: numpy 2d array with each row corresponding to one observation of predictor variables
        """
        y = self.learned_models['model'].predict(xdata)
        pervar_y = np.transpose(y)
        if verbose:
            print "Input data: {}".format(zip(self.predictor_names, xdata))
            for i,var in enumerate(self.dependent_names):
                print var + ": " + str(pervar_y[i])
        return pervar_y
    
    def compareParameters(self,xdata1,xdata2):
        """
        Uses the model to return the parameters that differ for xdata2 compared to xdata1.
        
        xdata1: numpy array corresponding with one observation of predictor variables
        xdata2: numpy array corresponding with one observation of predictor variables
        """
        y1 = self.useModel(xdata1)
        y2 = self.useModel(xdata2)
        diff = y2-y1
        print diff
        print "Input 1: {}\nInput 2: {}".format(zip(self.predictor_names, xdata1), zip(self.predictor_names,xdata2))
        for i,var in enumerate(self.dependent_names):
            if diff[i] > 0:
                print var + ": " + str(diff[i])
        return diff    
    
    def evaluate_all_models(self, learned_models, test_xdata=np.array([]), test_ydata=np.array([]), printsome=False, printall=False):
        """ Evaluate learned models for each phase """
        evaluation_models = {}
        for phase in learned_models:
            evaluation_models[phase] = self.evaluate_model(learned_models[phase], test_xdata, test_ydata, printsome=printsome, printall=printall)
        return evaluation_models
    
    def evaluate_model(self, learned_model, test_xdata=np.array([]), test_ydata=np.array([]), printsome=False, printall=False):
        """Evaluate models using R squared metric. Outputs train, CV and test results.
        
        Warning: CV scores are based on training data only, not on the testdata
        Warning: testscores might be empty, if no testset was given
        Note: learned_models contains dictionary with keys 'model errormetric modelxdata modelydata xnames ynames'
        """
        
        #Obtain relevant info from self.learned_models
        x=learned_model['modelxdata']
        y=learned_model['modelydata']
        lmodel = learned_model['model']
        y_pred = lmodel.predict(x)
        
        #Evaluation of model on trainset
        trainr2 = learned_model['model'].score(x, y)
        trainr2_adj = myutil.adjustedRS(trainr2, y.shape[0], x.shape[1])
        
        #Evaluation of model on testset
        ##Prepare testdata
        if not test_xdata.size==0: #if there are any samples in the testset
            test_x = test_xdata.astype(np.float)
            test_y = test_ydata.astype(np.float)
            y_real_pred = lmodel.predict(test_x)
            
            ##Testset
            testr2 = lmodel.score(test_x, test_y)
            testr2_adj = myutil.adjustedRS(testr2, test_y.shape[0], test_x.shape[1])
        else:
            print "No testset was given for evaluation"
         
        ##Scores separated per dependents column
        y_colnames = learned_model['ynames']
        #TODO: replaced explained_variance_score with r2_score, because we're using model.score in other places and that function actually just calls r2_score. See http://stackoverflow.com/questions/24378176/python-sci-kit-learn-metrics-difference-between-r2-score-and-explained-varian for an explanation of the difference between the two functions.
        #See difference in source code here (actually well-documented/commented) https://github.com/scikit-learn/scikit-learn/blob/bb39b49/sklearn/metrics/regression.py#L212
        traindata_scores = [r2_score(y[:,icol], y_pred[:,icol]) for icol in xrange(y.shape[1])] 
        
        ###train scores
        trainr2_sep = dict(zip(y_colnames, traindata_scores))
        adjusted_traindata_scores = [myutil.adjustedRS(score, y.shape[0], x.shape[1]) for score in traindata_scores]
        trainr2_adj_sep = dict(zip(y_colnames, adjusted_traindata_scores))
        
        if self.modeltype=='rforestreg':
            oob_y = lmodel.oob_prediction_
            traindata_scores_oob = [r2_score(y[:,icol], oob_y[:,icol]) for icol in xrange(y.shape[1])] 
            adjusted_traindata_scores_oob = [myutil.adjustedRS(score, y.shape[0], x.shape[1]) for score in traindata_scores_oob]
            trainr2_adj_sep_oob = dict(zip(y_colnames, adjusted_traindata_scores_oob))
            if printsome:
                print 'OOB train R2adj sep: ' + str(trainr2_adj_sep_oob)
            
        ###test scores
        if not test_xdata.size==0:
            testdata_scores = [r2_score(test_y[:,icol], y_real_pred[:,icol]) for icol in xrange(test_y.shape[1])]
            testr2_sep = dict(zip(y_colnames, testdata_scores))
            adjusted_testdata_scores = [myutil.adjustedRS(score, test_y.shape[0], test_x.shape[1]) for score in testdata_scores]
            testr2_adj_sep = dict(zip(y_colnames, adjusted_testdata_scores))
            testscores = {'r2':testr2, 'r2_adj':testr2_adj, 'r2_sep':testr2_sep, 'r2_adj_sep':testr2_adj_sep}
            if printsome or printall:
                print('Test R2 score: %s' % str(round(testr2, 4)))
                print('Test R2adj score: %s' % str(round(testr2_adj, 4)))
                if printall:
                    print 'Test R2 sep: ' + str(testr2_sep)
                    print 'Test R2adj sep: ' + str(testr2_adj_sep)
        else:
            testscores = None
        
        #Evaluate estimator performance using cross-validation
        cvr2_train = cross_validation.cross_val_score(lmodel, x, y, cv=self.CV_FOLDS, scoring='r2')
        cvr2_adj_train = np.array([myutil.adjustedRS(score ,y.shape[0], x.shape[1]) for score in cvr2_train ])
        #Estimator performance using cross-validation per column
        separate_scores = []
        avg_scores = []
        for icol in xrange(y.shape[1]):
            cur_y = y[:,icol]
            score = cross_validation.cross_val_score(lmodel, x, cur_y, cv=self.CV_FOLDS, scoring='r2')
            avg = np.mean(score)
            separate_scores.append(score)
            avg_scores.append(avg)
        cvr2_train_sep = dict(zip(y_colnames, avg_scores))
        adjusted_avg_scores = np.array([myutil.adjustedRS(score, y.shape[0], x.shape[1]) for score in avg_scores])
        cvr2_adj_train_sep = dict(zip(y_colnames, adjusted_avg_scores))
        #Put all the values together in dictionaries
        trainscores = {'oob_r2adj':trainr2_adj_sep_oob, 'r2':trainr2, 'r2_adj':trainr2_adj, 'r2_sep':trainr2_sep, 'r2_adj_sep':trainr2_adj_sep}
        cvscores = {'r2':cvr2_train, 'r2_adj':cvr2_adj_train, 'r2_sep':cvr2_train_sep, 'r2_adj_sep':cvr2_adj_train_sep}
        
        #If want to print, do here. Formatting float to string because R2adj could return None and that gives an error if expecting a float.
        if printsome or printall:
            print('Train R2 score: %s' % str(round(trainr2, 4))) # Explained variance score: 1 is perfect prediction
            print('Train R2adj score: %s' % str(round(trainr2_adj, 4)))
            print 'TrainCV R2 sep avg: '+ str(cvr2_train_sep)
            print 'TrainCV R2adj sep avg: ' + str(cvr2_adj_train_sep)
            if printall:    
                if not cvr2_train== None:
                    cvr2_mean = np.mean(cvr2_train)
                else:
                    cvr2_mean = None
                if not cvr2_adj_train== None:
                    cvr2_adj_mean = np.mean(cvr2_adj_train)
                else:
                    cvr2_adj_mean = None
                print('TrainCV R2 avg scores: %s' % str(round(cvr2_mean, 4)))
                print('TrainCV R2adj avg scores: %s' % str(round(cvr2_adj_mean, 4)))
                print 'TrainCV R2 scores: ' + str(cvr2_train)
                print 'TrainCV R2adj scores: ' + str(cvr2_adj_train)
                print 'Train R2 sep: ' + str(trainr2_sep) # Explained variance score: 1 is perfect prediction
                print 'Train R2adj sep: ' + str(trainr2_adj_sep)
                    
        result = {'model':lmodel, 'train':trainscores, 'test':testscores, 'cv':cvscores}
        return result
    
