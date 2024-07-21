


from __future__ import print_function
import sys
from itertools import product
import copy



import pandas as pd
#from ipfn import ipfn
import numpy as np
import streamlit as st 
import plotly.express as px


import datetime

from zipfile import ZipFile
from io import BytesIO

def convert_df(df):
    return df.to_csv().encode('utf-8')

st.write("""
# Iterative Proportional Fitting
""")

st.sidebar.header('Upload Input/Goal Features:')

uploaded_file1 = st.sidebar.file_uploader("Upload your input data", type=["csv", "xlsx"])
uploaded_file2 = st.sidebar.file_uploader("Upload your goals data", type=["csv", "xlsx"])


st.sidebar.header('Set the Max/Min Weighted Number:')
max_weigjted_number = st.sidebar.number_input("Maximun Weighted Number (Recommendation: 100000)",101,100000)

min_weigjted_number = st.sidebar.number_input("Minimun Weighted Number (Recommendation: 10)",1,100)


if st.sidebar.button('Submit'):


    if uploaded_file1 and uploaded_file2 and max_weigjted_number and min_weigjted_number is not None:
#############################################################
        
        #####
        #data cleaning for input
        
           
        if uploaded_file1.type == "text/csv":
            uploaded_file1.seek(0)
            df = pd.read_csv(uploaded_file1, low_memory=False)
            #df_target = pd.read_csv(uploaded_file2)
        elif uploaded_file1.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file1)
            #df_target = pd.read_excel(uploaded_file2)
        
        if uploaded_file2.type == "text/csv":
            #df = pd.read_csv(uploaded_file1)
            uploaded_file2.seek(0)
            df_target = pd.read_csv(uploaded_file2, low_memory=False)
        elif uploaded_file2.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            #df = pd.read_excel(uploaded_file1)
            df_target = pd.read_excel(uploaded_file2)

        
        #####
        #data cleaning for input
#        df = pd.read_excel(uploaded_file1)
#        df_target = pd.read_excel(uploaded_file2)
        ####        
        
        
        ####
        passcode = []
        if df.isnull().any().any():
            #text messenge show missing value exist!
            #1-1, 1
            data_not_pass = 1
        else:
            passcode.append(1)
            #continue to 1-2, 2
            if 'weightgroup' in list(df.columns):
                #weightgroup column exist in df
                passcode.append(1)
                #continue to 1-3, 3
                if 'weightgroup' in list(df_target.columns):
                    #weightgroup column exist in df_target
                    passcode.append(1)
                    #continue to 1-4, 4
                    if any(x.startswith('feature') for x in list(df.columns)):
                        #There's some column name start from ‘feature’
                        passcode.append(1)
                        #continue to 2-1, 5
                        if len([s for s in list(df.columns) if s.startswith('feature')]) >= 2:
                            #two or more features exist
                            passcode.append(1)
                            #continue to 2-2, 6
                            if sorted(list(range(1, df.filter(regex='^feature').shape[1]+1))) == sorted([float(item.split('feature')[1]) for item in list(df.filter(regex='^feature').columns)]):
                                #the order of the features is good
                                passcode.append(1)
                                #continue to 2-3, 7
                                if [str(a) for a in list(df.filter(regex='^feature').dtypes)] == ['int64']*(df.filter(regex='^feature').shape[1]):
                                    #all the content in feature columns are integer
                                    passcode.append(1)
                                    #continue to 2-4, 8
                                    if (df.filter(regex='^feature') == 0).any().any():
                                        #have 0 in the content of feature columns
                                        data_not_pass = 1
                                    else:
                                        #there’s no zero in the feature columns
                                        passcode.append(1)
                                        #continue to 2-5, 9
                                        
                                        check_feature_columns = []
                                        for i in range(df.filter(regex='^feature').shape[1]):
                                            check_feature_columns.append(int(list(range(1, max(list(df.filter(regex='^feature').iloc[0:,i])) + 1)) == sorted(list(dict.fromkeys(list(df.filter(regex='^feature').iloc[0:,i]))))))

                                        if all(item == 1 for item in check_feature_columns):
                                            #content of all the feature columns are good, no lack of information
                                            passcode.append(1)
                                            #continue to 2-6, 10
                                            
                                            feature_cloumns_in_df_target = []
                                            for i in range(df.filter(regex='^feature').shape[1]):
                                                feature_cloumns_in_df_target.append([list(df.filter(regex='^feature').columns)[i] + '_' + str(value) for value in sorted(list(dict.fromkeys(list(df.filter(regex='^feature').iloc[0:,i]))))])

                                            if [item for sublist in feature_cloumns_in_df_target for item in sublist] == list(df_target.filter(regex='^feature').columns):
                                                #the content and columns’ name in df fit with the feature columns in df_target
                                                passcode.append(1)
                                                #continue to 3-1, 11
                                                if str(df['weightgroup'].dtype) == 'int64':
                                                    #the data type of the weightgroup column in df is integer
                                                    passcode.append(1)
                                                    #continue to 3-2, 12
                                                    if str(df_target['weightgroup'].dtype) == 'int64':
                                                        #the data type of the weightgroup column in df_targetis is integer
                                                        passcode.append(1)
                                                        #continue to 3-3, 13
                                                        if sorted(list(dict.fromkeys(list(df['weightgroup'])))) == list(df_target['weightgroup']):
                                                            #the number of weightgroup does fit
                                                            passcode.append(1)
                                                            #continue to 3-4, 14
                                                            
                                                            df['count'] = 1
                                                            data_scarcity = []
                                                            for i in list(df.filter(regex='^feature').columns):
                                                                for j in list(dict.fromkeys(list(df['weightgroup']))): #wweightgroup number
                                                                    if list(set(list(range(1, max(list(df[i]))+1))) - set(list(pd.DataFrame(df.groupby(['weightgroup', i])['count'].sum()[j]).index))) == []:
                                                                        data_scarcity.append([i, j, 0, 0])
                                                                    else:
                                                                        data_scarcity.append([i, j, 1, list(set(list(range(1, max(list(df[i]))+1))) - set(list(pd.DataFrame(df.groupby(['weightgroup', i])['count'].sum()[j]).index)))])
                                                            
                                                            df_temp_0 = pd.DataFrame(data_scarcity)

                                                            if sum(list(df_temp_0[2])) == 0:
                                                                #enough data for each weightgroup and feature
                                                                passcode.append(1)
                                                            else:
                                                                #some content in some weightgroup is lack of data
                                                                data_not_pass = 1
                                                                
                                                                df_temp = df_temp_0[df_temp_0[2] == 1]
                                                                df_temp.columns = ['Feature', 'Weightgroup', 'X', 'Lack of Information']
                                                                df_temp = df_temp.drop('X', axis=1)
                                                                
                                                                #display df_temp to see which data is not enough


                                                        else:
                                                            #the number of weightgroup does not fit
                                                            data_not_pass = 1
                                                    else:
                                                        #the data type of the weightgroup column in df_targetis is not integer
                                                        data_not_pass = 1
                                                else:
                                                    #the data type of the weightgroup column in df is not integer
                                                    data_not_pass = 1
                                            else:
                                                #the content and columns’ name in df not fit with the feature columns in df_target
                                                data_not_pass = 1
                                        else:
                                            #miss the value in content. eg, Only 1, 3, 4 in feature2 column
                                            data_not_pass = 1
                                else:
                                    #not all the content in feature columns are integer
                                    data_not_pass = 1
                            else:
                                #False if the number of feature duplicated
                                data_not_pass = 1
                        else:
                            #text show there’s only one feature
                            data_not_pass = 1
                    else:
                        #text show no column name start from ‘feature’ #1-4
                        data_not_pass = 1
                else:
                    #text show theres no weightgroup column in goal data #1-3
                    data_not_pass = 1
            else:
                #text show theres no weightgroup column in input data #1-2
                data_not_pass = 1
        
        #####
        
        if sum(passcode) < 14:
            st.warning("Please check the input data and goal data again...")
            #st.subheader("Data Scarcity")
            #st.subheader(data_not_pass)
            #st.subheader(sum(passcode))
            
            if (data_not_pass == 1) and (sum(passcode) == 0):           #1-1, 1
                st.subheader("Missing values exist...")
                
            elif (data_not_pass == 1) and (sum(passcode) == 1):     #1-2, 2
                st.subheader("There’s no ‘weightgroup’ column in input data")
                
            elif (data_not_pass == 1) and (sum(passcode) == 2):     #1-3, 3
                st.subheader("There’s no ‘weightgroup’ column in goal data")
                
            elif (data_not_pass == 1) and (sum(passcode) == 3):     #1-4, 4
                st.subheader("There’s no column name start from ‘feature'")
                
            elif (data_not_pass == 1) and (sum(passcode) == 4):     #2-1, 5
                st.subheader("There's only one feature column")
                
            elif (data_not_pass == 1) and (sum(passcode) == 5):     #2-2, 6
                st.subheader("Having problems with features' name, please check.")
                
            elif (data_not_pass == 1) and (sum(passcode) == 6):     #2-3, 7
                st.subheader("The data in not all interger in the feature columns")
                
            elif (data_not_pass == 1) and (sum(passcode) == 7):     #2-4, 8
                st.subheader("In input data, items should start from 1 in the feature columns")
                
            elif (data_not_pass == 1) and (sum(passcode) == 8):     #2-5, 9
                st.subheader("Some items missing in the feature columns, such as only 1, 2, 4 in feature1 column")
                
            elif (data_not_pass == 1) and (sum(passcode) == 9):     #2-6, 10
                st.subheader("The feature does not fit between input data and goal data")
                
            elif (data_not_pass == 1) and (sum(passcode) == 10):    #3-1, 11
                st.subheader("The data type of the weightgroup column in the input data is not integer")
                
            elif (data_not_pass == 1) and (sum(passcode) == 11):    #3-2, 12
                st.subheader("The data type of the weightgroup column in the goal data is not integer")
                
            elif (data_not_pass == 1) and (sum(passcode) == 12):    #3-3, 13
                st.subheader("The weightgroup information does not fit between input data and goal data")
                
            elif (data_not_pass == 1) and (sum(passcode) == 13):    #4-1, 14
 #               def main():
                    
                st.subheader("Data Scarcity")
                
                #df_temp = df_temp_0[df_temp_0[2] == 1]
                #df_temp.columns = ['Feature', 'Weightgroup', 'X', Lack of Information']
                #df_temp = df_temp.drop('X', axis=1)
                
                st.table(df_temp)
 #               if __name__ == '__main__':
 #                   main()

        else:
        
            
            
            st.sidebar.success("Successfully Submitted, please wait ...")
            
            class ipfn(object):
        
                def __init__(self, original, aggregates, dimensions, weight_col='total',
                             convergence_rate=1e-5, max_iteration=500, verbose=0, rate_tolerance=1e-8):
                    """
                    Initialize the ipfn class
                    original: numpy darray matrix or dataframe to perform the ipfn on.
                    aggregates: list of numpy array or darray or pandas dataframe/series. The aggregates are the same as the marginals.
                    They are the target values that we want along one or several axis when aggregating along one or several axes.
                    dimensions: list of lists with integers if working with numpy objects, or column names if working with pandas objects.
                    Preserved dimensions along which we sum to get the corresponding aggregates.
                    convergence_rate: if there are many aggregates/marginal, it could be useful to loosen the convergence criterion.
                    max_iteration: Integer. Maximum number of iterations allowed.
                    verbose: integer 0, 1 or 2. Each case number includes the outputs of the previous case numbers.
                    0: Updated matrix returned.
                    1: Flag with the output status (0 for failure and 1 for success).
                    2: dataframe with iteration numbers and convergence rate information at all steps.
                    rate_tolerance: float value. If above 0.0, like 0.001, the algorithm will stop once the difference between the conv_rate variable of 2 consecutive iterations is below that specified value
                    For examples, please open the ipfn script or look for help on functions ipfn_np and ipfn_df
                    """
                    self.original = original
                    self.aggregates = aggregates
                    self.dimensions = dimensions
                    self.weight_col = weight_col
                    self.conv_rate = convergence_rate
                    self.max_itr = max_iteration
                    self.verbose = verbose
                    self.rate_tolerance = rate_tolerance
        
                @staticmethod
                def index_axis_elem(dims, axes, elems):
                    inc_axis = 0
                    idx = ()
                    for dim in range(dims):
                        if (inc_axis < len(axes)):
                            if (dim == axes[inc_axis]):
                                idx += (elems[inc_axis],)
                                inc_axis += 1
                            else:
                                idx += (np.s_[:],)
                    return idx
        
                def ipfn_np(self, m, aggregates, dimensions, weight_col='total'):
                    """
                    Runs the ipfn method from a matrix m, aggregates/marginals and the dimension(s) preserved.
                    For example:
                    from ipfn import ipfn
                    import numpy as np
                    m = np.array([[8., 4., 6., 7.], [3., 6., 5., 2.], [9., 11., 3., 1.]], )
                    xip = np.array([20., 18., 22.])
                    xpj = np.array([18., 16., 12., 14.])
                    aggregates = [xip, xpj]
                    dimensions = [[0], [1]]
                    IPF = ipfn(m, aggregates, dimensions)
                    m = IPF.iteration()
                    """
        
                    # Check that the inputs are numpay arrays of floats
                    inc = 0
                    for aggregate in aggregates:
                        if not isinstance(aggregate, np.ndarray):
                            aggregate = np.array(aggregate).astype(np.float)
                            aggregates[inc] = aggregate
                        elif aggregate.dtype not in [np.float, float]:
                            aggregate = aggregate.astype(np.float)
                            aggregates[inc] = aggregate
                        inc += 1
                    if not isinstance(m, np.ndarray):
                        m = np.array(m)
                    elif m.dtype not in [np.float, float]:
                        m = m.astype(np.float)
        
                    steps = len(aggregates)
                    dim = len(m.shape)
                    product_elem = []
                    tables = [m]
                    # TODO: do we need to persist all these dataframe? Or maybe we just need to persist the table_update and table_current
                    # and then update the table_current to the table_update to the latest we have. And create an empty zero dataframe for table_update (Evelyn)
                    for inc in range(steps - 1):
                        tables.append(np.array(np.zeros(m.shape)))
                    original = copy.copy(m)
        
                    # Calculate the new weights for each dimension
                    for inc in range(steps):
                        if inc == (steps - 1):
                            table_update = m
                            table_current = tables[inc].copy()
                        else:
                            table_update = tables[inc + 1]
                            table_current = tables[inc]
                        for dimension in dimensions[inc]:
                            product_elem.append(range(m.shape[dimension]))
                        for item in product(*product_elem):
                            idx = self.index_axis_elem(dim, dimensions[inc], item)
                            table_current_slice = table_current[idx]
                            mijk = table_current_slice.sum()
                            # TODO: Directly put it as xijk = aggregates[inc][item] (Evelyn)
                            xijk = aggregates[inc]
                            xijk = xijk[item]
                            if mijk == 0:
                                # table_current_slice += 1e-5
                                # TODO: Basically, this part would remain 0 as always right? Cause if the sum of the slice is zero, then we only have zeros in this slice.
                                # TODO: you could put it as table_update[idx] = table_current_slice (since multiplication on zero is still zero)
                                table_update[idx] = table_current_slice
                            else:
                                # TODO: when inc == steps - 1, this part is also directly updating the dataframe m (Evelyn)
                                # If we are not going to persist every table generated, we could still keep this part to directly update dataframe m
                                table_update[idx] = table_current_slice * 1.0 * xijk / mijk
                            # For debug purposes
                            # if np.isnan(table_update).any():
                            #     print(idx)
                            #     sys.exit(0)
                        product_elem = []
        
                    # Check the convergence rate for each dimension
                    max_conv = 0
                    for inc in range(steps):
                        # TODO: this part already generated before, we could somehow persist it. But it's not important (Evelyn)
                        for dimension in dimensions[inc]:
                            product_elem.append(range(m.shape[dimension]))
                        for item in product(*product_elem):
                            idx = self.index_axis_elem(dim, dimensions[inc], item)
                            ori_ijk = aggregates[inc][item]
                            m_slice = m[idx]
                            m_ijk = m_slice.sum()
                            # print('Current vs original', abs(m_ijk/ori_ijk - 1))
                            if abs(m_ijk / ori_ijk - 1) > max_conv:
                                max_conv = abs(m_ijk / ori_ijk - 1)
        
                        product_elem = []
        
                    return m, max_conv
        
                def ipfn_df(self, df, aggregates, dimensions, weight_col='total'):
                    """
                    Runs the ipfn method from a dataframe df, aggregates/marginals and the dimension(s) preserved.
                    For example:
                    from ipfn import ipfn
                    import pandas as pd
                    age = [30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50]
                    distance = [10,20,30,40,10,20,30,40,10,20,30,40]
                    m = [8., 4., 6., 7., 3., 6., 5., 2., 9., 11., 3., 1.]
                    df = pd.DataFrame()
                    df['age'] = age
                    df['distance'] = distance
                    df['total'] = m
                    xip = df.groupby('age')['total'].sum()
                    xip.loc[30] = 20
                    xip.loc[40] = 18
                    xip.loc[50] = 22
                    xpj = df.groupby('distance')['total'].sum()
                    xpj.loc[10] = 18
                    xpj.loc[20] = 16
                    xpj.loc[30] = 12
                    xpj.loc[40] = 14
                    dimensions = [['age'], ['distance']]
                    aggregates = [xip, xpj]
                    IPF = ipfn(df, aggregates, dimensions)
                    df = IPF.iteration()
                    print(df)
                    print(df.groupby('age')['total'].sum(), xip)"""
        
                    steps = len(aggregates)
                    tables = [df]
                    for inc in range(steps - 1):
                        tables.append(df.copy())
                    original = df.copy()
        
                    # Calculate the new weights for each dimension
                    inc = 0
                    for features in dimensions:
                        if inc == (steps - 1):
                            table_update = df
                            table_current = tables[inc].copy()
                        else:
                            table_update = tables[inc + 1]
                            table_current = tables[inc]
        
                        tmp = table_current.groupby(features)[weight_col].sum()
                        xijk = aggregates[inc]
        
                        feat_l = []
                        for feature in features:
                            feat_l.append(np.unique(table_current[feature]))
                        table_update.set_index(features, inplace=True)
                        table_current.set_index(features, inplace=True)
        
                        multi_index_flag = isinstance(table_update.index, pd.MultiIndex)
                        if multi_index_flag:
                            if not table_update.index.is_lexsorted():
                                table_update.sort_index(inplace=True)
                            if not table_current.index.is_lexsorted():
                                table_current.sort_index(inplace=True)
        
                        for feature in product(*feat_l):
                            den = tmp.loc[feature]
                            # calculate new weight for this iteration
        
                            if not multi_index_flag:
                                msk = table_update.index == feature[0]
                            else:
                                msk = feature
        
                            if den == 0:
                                table_update[weight_col] = np.clip(table_update[weight_col], a_max=max_weigjted_number, a_min=min_weigjted_number)
                                table_update.loc[msk, weight_col] =\
                                    table_current.loc[feature, weight_col] *\
                                    xijk.loc[feature]
                            else:
                                table_update[weight_col] = np.clip(table_update[weight_col], a_max=max_weigjted_number, a_min=min_weigjted_number)
                                table_update.loc[msk, weight_col] = \
                                    table_current.loc[feature, weight_col].astype(float) * \
                                    xijk.loc[feature] / den
        
        
                        table_update.reset_index(inplace=True)
                        table_current.reset_index(inplace=True)
                        inc += 1
                        feat_l = []
        
                    # Calculate the max convergence rate
                    max_conv = 0
                    inc = 0
                    for features in dimensions:
                        tmp = table_update.groupby(features)[weight_col].sum()
                        ori_ijk = aggregates[inc]
                        temp_conv = max(abs(tmp / ori_ijk - 1))
                        if temp_conv > max_conv:
                            max_conv = temp_conv
                        inc += 1
        
                    return table_update, max_conv
        
                def iteration(self):
                    """
                    Runs the ipfn algorithm. Automatically detects of working with numpy ndarray or pandas dataframes.
                    """
        
                    i = 0
                    conv = np.inf
                    old_conv = -np.inf
                    conv_list = []
                    m = self.original
        
                    # If the original data input is in pandas DataFrame format
                    if isinstance(self.original, pd.DataFrame):
                        ipfn_method = self.ipfn_df
                    elif isinstance(self.original, np.ndarray):
                        ipfn_method = self.ipfn_np
                        self.original = self.original.astype('float64')
                    else:
                        print('Data input instance not recognized')
                        sys.exit(0)
                    while ((i <= self.max_itr and conv > self.conv_rate) and (i <= self.max_itr and abs(conv - old_conv) > self.rate_tolerance)):
                        old_conv = conv
                        m, conv = ipfn_method(m, self.aggregates, self.dimensions, self.weight_col)
                        conv_list.append(conv)
                        i += 1
                    converged = 1
                    if i <= self.max_itr:
                        if (not conv > self.conv_rate) & (self.verbose > 1):
                            print('ipfn converged: convergence_rate below threshold')
                        elif not abs(conv - old_conv) > self.rate_tolerance:
                            print('ipfn converged: convergence_rate not updating or below rate_tolerance')
                    else:
                        print('Maximum iterations reached')
                        converged = 0
        
                    # Handle the verbose
                    if self.verbose == 0:
                        return m
                    elif self.verbose == 1:
                        return m, converged
                    elif self.verbose == 2:
                        return m, converged, pd.DataFrame({'iteration': range(i), 'conv': conv_list}).set_index('iteration')
                    else:
                        print('wrong verbose input, return None')
                        sys.exit(0)
        
            
            #df = pd.read_excel(uploaded_file1) # show上傳的data
            #df_target = pd.read_excel(uploaded_file2) # show上傳的data
            def main():
                st.title(" ")
                st.subheader("The difference table heat map ")
                
        
                #df = pd.read_excel(uploaded_file1) # show上傳的data
                #df_target = pd.read_excel(uploaded_file2) # show上傳的data



           
                if uploaded_file1.type == "text/csv":
                    uploaded_file1.seek(0)
                    df = pd.read_csv(uploaded_file1, low_memory=False)
                    #df_target = pd.read_csv(uploaded_file2)
                elif uploaded_file1.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    df = pd.read_excel(uploaded_file1)
                    #df_target = pd.read_excel(uploaded_file2)
                
                if uploaded_file2.type == "text/csv":
                    #df = pd.read_csv(uploaded_file1)
                    uploaded_file2.seek(0)
                    df_target = pd.read_csv(uploaded_file2, low_memory=False)
                elif uploaded_file2.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    #df = pd.read_excel(uploaded_file1)
                    df_target = pd.read_excel(uploaded_file2)



                
                
                df.columns = [col.lower() for col in df.columns]
                df.columns = df.columns.str.replace(r'\(.*\)', '', regex=True)
                df.columns = df.columns.str.strip()
                df.columns = df.columns.str.replace('code', '', regex=False)
                
                #df['weightgroup'] = df['weightgroup'].astype(str)                   ###############
                #df_target['weightgroup'] = df_target['weightgroup'].astype(str)     ###############
                
                df['weightgroup'] = [str(i).zfill(3) for i in df['weightgroup']]
                df_target['weightgroup'] = [str(i).zfill(3) for i in df_target['weightgroup']]
                
                if not any(col in df.columns for col in ["weight", "weight group", "weightgroup"]):
                    if "district" in df.columns:
                        df = df.rename(columns={"district": "weightgroup"})
                
                feature_new = [col for col in list(df.columns) if col.startswith('feature')]
                column_groups = {}
                
                for col in list(df_target.columns):
                    parts = col.split('_')
                    if len(parts) > 1:
                        group_name = parts[0]
                        if group_name not in column_groups:
                            column_groups[group_name] = []
                        column_groups[group_name].append(col)
                
                sub_feature_new = list(column_groups.values())
                
                df_weight = df
                df_weight['total'] = 1
                
                flattened_list = [item for sublist in sub_feature_new for item in sublist]
                df_target = df_target[['weightgroup', 'totalgoal'] + flattened_list]
        
                xip = df_target.groupby('weightgroup')['totalgoal'].sum()
                
                #######################################################   
                        
                df_0 = df
                df_0['weight'] = sum(list(df_target['totalgoal']))/len(df)
                b = pd.DataFrame()
                for p in range(len(feature_new)):
                    b = pd.concat([b, pd.DataFrame(df_0.groupby(['weightgroup', feature_new[p]])['total'].sum().unstack())], axis=1)
                b.columns = flattened_list
                b.reset_index(inplace = True)
                
                for q in flattened_list:
                  b[q] = b[q]/df_target['totalgoal']
                
                #######################################################
                
                for n in range(50):
                
                    for k in range(len(feature_new)):  #feature_new = ['feature1', 'feature2', ...]
                        
                        #xip = xip.astype('float')
                        xpj = df_weight.groupby(['weightgroup', feature_new[k]])['total'].sum()
                        xpj = xpj.astype('float')
                        #target_weight_by_feature = []
                      
                        #for m in range(7):
                      
                        for i in range(len(df_target['totalgoal'])):
                            for j in range(len(sub_feature_new[k])): #'0' here means the first feature
                                #target_weight_by_feature.append(df_target['totalgoal'][i]*float(df_target[sub_feature_new[k][j]][i]))
                                #xpj.loc[i + 1, j + 1] = df_target['totalgoal'][i]*float(df_target[sub_feature_new[k][j]][i])
                                xpj[df_target['weightgroup'][i]][j+1] = df_target['totalgoal'][i]*float(df_target[sub_feature_new[k][j]][i])
                                print(k,i,j)
                          
                        aggregates = [xip, xpj]
                        dimensions = [['weightgroup'], ['weightgroup', feature_new[k]]]
                          
                        IPF = ipfn(df_weight, aggregates, dimensions, max_iteration=1, convergence_rate=1e-7, rate_tolerance=1e-8)
                        df_weight = IPF.iteration()
                    
                    
                
                a = pd.DataFrame()
                for l in range(len(feature_new)):
                    a = pd.concat([a, pd.DataFrame(df_weight.groupby(['weightgroup', feature_new[l]])['total'].sum().unstack())], axis=1)
                a.columns = flattened_list
                a.reset_index(inplace = True)
                
                for m in flattened_list:
                  a[m] = a[m]/df_target['totalgoal']
                
                diff = a[flattened_list] - df_target[flattened_list]
                #diff.values
                #flattened_list
                fig = px.imshow(diff.values,
                                labels=dict(x="Sub Features", y="WeightGroups", color="Productivity"),
                                x=flattened_list,
                                y=list(df_target['weightgroup'])
                               )
                #fig.update_xaxes(side="top")
                #fig.show()
                st.plotly_chart(fig)
                
                diff_csv = convert_df(diff)
                weight_csv = convert_df(df_weight)
                
                diff_0 = b[flattened_list] - df_target[flattened_list]
                
                diff_0_csv = convert_df(diff_0)
                weight_0_csv = convert_df(df_0)
                
                def create_zip_file():

                    # Specify CSV file names
                    csv_file1 = 'diff_table_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.csv'
                    csv_file2 = 'weighted_table_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.csv'
                    csv_file3 = 'diff_table_none_ipf_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.csv'
                    csv_file4 = 'weighted_table_none_ipf_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.csv'
                
                    # Save DataFrames to CSV files
                    diff.to_csv(csv_file1, index=False)
                    df_weight.to_csv(csv_file2, index=False)
                    diff_0.to_csv(csv_file3, index=False)
                    df_0.to_csv(csv_file4, index=False)
                    
                    
                
                    # Create a zip file in memory and add CSV files to it
                    zip_buffer = BytesIO()
                    with ZipFile(zip_buffer, 'w') as zipf:
                        zipf.write(csv_file1)
                        zipf.write(csv_file2)
                        zipf.write(csv_file3)
                        zipf.write(csv_file4)
                
                    return zip_buffer
                
                #if st.button('Download Zip File'):
                zip_buffer = create_zip_file()
                st.markdown("Note: After you click any download buttom, the web page will be reset. Please choose to download multiple or single files.")
                
                #st.markdown(str(uploaded_file1.type))
                
                st.caption('Download All files')
                
                st.download_button(label='Download difference and weighted tables', data=zip_buffer.getvalue(), file_name= 'diff_and_weighted_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.zip', mime='application/zip')
                
                st.caption('Download single file')

                st.download_button(
                     label = "Download the difference table",
                     data = diff_csv,
                     file_name='diff_table_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.csv',
                     mime='text/csv',
                 )
                
                st.download_button(
                     label = "Download the weighted table",
                     data = weight_csv,
                     file_name='weighted_table_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.csv',
                     mime='text/csv',
                 )

                ################################
                ######################################################
                

                
                st.download_button(
                     label = "Download the difference table(without IPF process, just for a comparison)",
                     data = diff_0_csv,
                     file_name='diff_none_ipf.csv',
                     mime='text/csv',
                 )
                
                st.download_button(
                     label = "Download the weighted table(without IPF process, just for a comparison)",
                     data = weight_0_csv,
                     file_name='weighted_none_ipf.csv',
                     mime='text/csv',
                 )
                ######################################################
        
        
            
                
                
            if __name__ == '__main__':
                main()
                
                




#############################################################
    else:
      
        # 直接show既有的sales_data
        def main():
            st.title("Uploaded file example")
            st.subheader("Input file")
            st.caption('1. The file extension must be ".xlsx" or ".csv"')
            st.caption('2. The data must have "weightgroup" and "feature" columns, and the contents must be numbers')
            st.caption('3. Whether it is the content in weightgroup column or the feature columns, the starting value must be 1')
            st.caption('4. In addition to weightgroup and feature columns, it can have other columns such as id, date, etc')
            df_input = pd.read_excel('template_input.xlsx')
    
            st.table(df_input.head(10))
            
            def convert_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            df_input_csv = convert_to_csv(df_input)
    
            st.download_button(label = "Download the input file template", data = df_input_csv, file_name='input_template.csv')
    
            
            st.subheader("Goal file")
            st.caption('1. The file extension must be ".xlsx" or ".csv"')
            st.caption('2. The columns must contain "weightgroup", "totalgoal", and the feature part')
            st.caption('3. The rule of the feature part: there is a underline between feature and the content, such as feature1_3, feature4_2, etc')
            #st.caption('1. The extension should be ".xlsx"')
            df_goals = pd.read_excel('template_goals.xlsx')
            
            st.table(df_goals.head(10))
            
            df_goals_csv = convert_to_csv(df_goals)
            
            st.download_button(label = "Download the goal file template", data = df_goals_csv, file_name='goals_template.csv')
    
        if __name__ == '__main__':
            main()
else:
    st.sidebar.warning("Not Yet Submitted")
    
    def main():
        st.title("Uploaded file example")
        st.subheader("Input file")
        st.caption('1. The file extension must be ".xlsx" or ".csv"')
        st.caption('2. The data must have "weightgroup" and "feature" columns, and the contents must be numbers')
        st.caption('3. Whether it is the content in weightgroup column or the feature columns, the starting value must be 1')
        st.caption('4. In addition to weightgroup and feature columns, it can have other columns such as id, date, etc')
        df_input = pd.read_excel('template_input.xlsx')

        st.table(df_input.head(10))
        
        def convert_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        df_input_csv = convert_to_csv(df_input)

        st.download_button(label = "Download the input file template", data = df_input_csv, file_name='input_template.csv')

        
        st.subheader("Goal file")
        st.caption('1. The file extension must be ".xlsx" or ".csv"')
        st.caption('2. The columns must contain "weightgroup", "totalgoal", and the feature part')
        st.caption('3. The rule of the feature part: there is a underline between feature and the content, such as feature1_3, feature4_2, etc')
        #st.caption('1. The extension should be ".xlsx"')
        df_goals = pd.read_excel('template_goals.xlsx')
        
        st.table(df_goals.head(10))
        
        df_goals_csv = convert_to_csv(df_goals)
        
        st.download_button(label = "Download the goal file template", data = df_goals_csv, file_name='goals_template.csv')

    if __name__ == '__main__':
        main()
    
    
    
    

