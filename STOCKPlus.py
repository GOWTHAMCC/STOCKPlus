import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import numpy as np 
import pandas as pd
import altair as alt
from PIL import Image



nav = st.sidebar.radio("MENU",["Home","Prediction","Screener","Process Flow","About us"])

image = Image.open("C:/Users/fahad/Desktop/Fahad/Fahad/Fahad/PBL/Final/SG23/BMSITM.png")
st.image(image, caption='',width = 500)

if nav == "Home":
    #st.title(' TO THE OPEN DAY 2021 OF BMSIT & M')
    st.title("WELCOME!! This is  STOCKPlus")
    st.header('STOCK PRICE PREDICTION WEB APPLICATION')           
    #st.image("C:/Users/fahad/Desktop/Fahad/FahadFahad/PBL/Final/PBL Final/data/img.jpeg")
    from PIL import Image
    image = Image.open("C:/Users/fahad/Desktop/Fahad/Fahad/Fahad/PBL/Final/SG23/image.jpg")
    st.image(image, caption='Stock Price Prediction', width=800)
    
    st.header('Our application is the act of trying to determine the future price of a company stock. The successful prediction of a stocks future price could yield significant profit. It basically works with the past values of the company and gives the approximate range of the stock price.')    
    

if nav == "Prediction":

    st.header("MAKE PREDICTIONS OF ANY STOCK OF YOUR CHOICE.") 
    st.subheader("Enter the Company name")
    selected_stock = st.text_input("please enter the ticker name only")
    Z = st.slider("No. of days in Future", min_value=1, max_value=30)
    
    if  st.button("PREDICT"):
    
    
        ### Data Collection
        start="2020-01-01"
        today=date.today().strftime("%Y-%m-%d")
    
        #selected_stock=st.text_input('select the company stocks')   

        @st.cache
        def load_data(ticker):
            data=yf.download(ticker,start,today)
            data.reset_index(inplace=True)
            return data
    
    
        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!')
        
        st.subheader('Past Data of the Stock')
        st.write(data.tail())
    
        # Plot raw data
        def plot_raw_data():
             fig = go.Figure()
             fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
             fig.layout.update(title_text='Performance of the Stock with respect to Time', xaxis_rangeslider_visible=True)
             st.plotly_chart(fig)
            
             
	
        plot_raw_data()


        df1=data.reset_index()['Close']

        ### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 
        from sklearn.preprocessing import MinMaxScaler
 
        scaler=MinMaxScaler(feature_range=(0,1))
        df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


        ##splitting dataset into train and test split
        training_size=int(len(df1)*0.70)
        test_size=len(df1)-training_size
        train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


        # convert an array of values into a dataset matrix
        def create_dataset(dataset, time_step=1):
	        dataX, dataY = [], []
	        for i in range(len(dataset)-time_step-1):
    		    a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
    		    dataX.append(a)
    		    dataY.append(dataset[i + time_step, 0])
	        return np.array(dataX), np.array(dataY)


        #f=st.slider('No. of days in past', 1, 30)
        f=18
        
        # reshape into X=t,t+1,t+2,t+3 and Y=t+4
        time_step = f
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)
        
        
        
        # reshape input to be [samples, time steps, features] which is required for LSTM
        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
        
        
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM,Dropout
        from tensorflow.keras.layers import Dense
        import math
        from sklearn.metrics import mean_squared_error
        
        ### Create the Stacked LSTM model
        
        model=Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(f,1)))
        model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')
        
        
        
        
        model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=30,batch_size=5,verbose=1)
        
        
        
        ### Lets Do the prediction and check performance metrics
        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)
        
        
        ##Transformback to original form
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)
        
        
        
        ### Calculate RMSE performance metrics
        
        math.sqrt(mean_squared_error(y_train,train_predict))
        
        
        ### Test Data RMSE
        math.sqrt(mean_squared_error(ytest,test_predict))
        
        
        import matplotlib.pyplot as plt 
        
        ### Plotting 
        # shift train predictions for plotting
        look_back=f
        trainPredictPlot = np.empty_like(df1)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(df1)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
        # plot baseline and predictions
    
            
        x_input=test_data[(len(test_data)-f):].reshape(1,-1)
        #x_input.shape
    
        
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        
        
            
        # demonstrate prediction for next n days
    
        
        from numpy import array

        
        lst_output=[]
        n_steps=f
        i=0
        while(i<int(Z)):
            
            if(len(temp_input)>f):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                # print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                # print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                    x_input = x_input.reshape((1, n_steps,1))
                    yhat = model.predict(x_input, verbose=0)
                    # print(yhat[0])
                    temp_input.extend(yhat[0].tolist())
                    # print(len(temp_input))
                    lst_output.extend(yhat.tolist())
                    i=i+1
                    
                    day_new=np.arange(1,f+1)
                    day_pred=np.arange((f+1),(f+1)+int(Z))


            #plt.plot(day_new,scaler.inverse_transform(df1[(len(df1)-f):]))
            #plt.plot(day_pred,scaler.inverse_transform(lst_output))


            #day_pred=np.arange(1,int(Z)+1)
            #plt.plot(day_pred,scaler.inverse_transform(lst_output))
            


            #print(scaler.inverse_transform(lst_output))
            output = scaler.inverse_transform(lst_output)
            
            j=1
        for item in output:
            x=float(item)
            st.subheader(f"The Predicted Price for {j} Day is {x}")
            j=j+1
            
        st.header(' ')
        from PIL import Image
        image = Image.open("C:/Users/fahad/Desktop/Fahad/Fahad/Fahad/PBL/Final/SG23/disclaimer.jpeg")
        st.image(image, caption='DISCLAIMER', width = 200)
        
        st.header("DISCLAIMER !!")
        st.subheader("The information provided by STOCKPlus is for general informational purposes only. All information on our application provided in good faith, however we make no representation or warranty of any kind, express or implied, regarding the accuracy, adequacy, validity, realiability or completeness of any information on our application. Under no circumstance shall we have any liability to you for any loss or damage of any kind incurred as a result of the use of our mobile application or reliance on any information provided on our mobile application. Your use of our mobile application and your reliance on any information on these platforms is solely at your own risk.")
      
if nav == "Screener"  :

    st.header("WANT TO KNOW THE PERFORMANCE OF A COMAPNY OF YOUR INTEREST ?? ")
      
    st.subheader("Enter the Company name")
    selected_stock = st.text_input("please enter the ticker name only")
    
    if  st.button("SUBMIT"):
        
        from PIL import Image
        image = Image.open("C:/Users/fahad/Desktop/Fahad/Fahad/Fahad/PBL/Final/SG23/NSE.jpg")
        st.image(image, caption='National Stock Exchange', width = 600)
    
        from PIL import Image
        image = Image.open("C:/Users/fahad/Desktop/Fahad/Fahad/Fahad/PBL/Final/SG23/BSE.jpg")
        st.image(image, caption='Bomaby Stock Exchnage', width = 600)
        
       
    
        g = yf.Ticker(selected_stock)
        # get stock info
        
        st.subheader("CashFlow of " + selected_stock)
        x = g.cashflow
        st.table(x)
        
        st.subheader("BalanceSheet of " + selected_stock)
        y = g.balancesheet
        st.table(y)

        st.subheader("Earnings of " + selected_stock)
        z = g.earnings
        st.table(z)
        
        st.subheader("Financials of " + selected_stock)
        xx = g.financials
        st.table(xx)
        
      
    
if nav == "Process Flow":
    st.header('Process Flow')
    st.graphviz_chart(""" 
digraph{
    Give_The_Company_Name  -> Collect_Past_Data
    Collect_Past_Data -> Preprocessing
    Preprocessing -> Train_the_MachineLearning_Model
    Train_the_MachineLearning_Model -> Testing_The_Data
    Testing_The_Data -> Making_PREDICTION 
    Making_PREDICTION ->Display_The_Predicted_Value
    Display_The_Predicted_Value ->Give_The_Company_Name

    }
    """)
    
    st.header(''' BENEFITS
              
                 	Helps companies to raise capital 

                 	Helps create personal wealth

                 	Serves as an indicator of the state of the economy 
                 
                 	Helps to increase investment
                 
                ''')
    
    from PIL import Image
    image = Image.open("C:/Users/fahad/Desktop/Fahad/Fahad/Fahad/PBL/Final/SG23/block.jpg")
    st.image(image, caption='BLOCK DIAGRAM')
      

if nav == "About us":
    st.header('We are BMSITians !!')
    st.subheader("We are a team of curious fellows, who are indulged in building an Web Application, Which basically predicts future price of a particular stock selected by the user, for any user desired time/period.")
    st.write("Presently in 4th Sem B.E in Electronics and Communication Engineering in B.M.S Institute of Technology and Management, to be completed in the year 2023. We are a smartworking team who defines time and completes task pre hand, an opportunity explorer.")
    
    from PIL import Image
    image = Image.open("C:/Users/fahad/Desktop/Fahad/Fahad/Fahad/PBL/Final/SG23/Team.jpg")
    st.image(image, caption='TEAM WORK')
    
    st.subheader("Submitted By SG23")
    st.write( "1. Mohammed Fahad R – 1BY19EC102")
    st.write( "2. Gowtham C – 1BY19EC056")
    st.write( "3. Manjunath L – 1BY19EC097")
    st.write( "4. Monish Kumar N – 1BY19EC104")
    
             
             
            