Fire Weather Index (FWI) Prediction System Forest fires are one of the most dangerous natural disasters that destroy wildlife, vegetation, and property.Fire Weather Index (FWI) is a numerical rating used worldwide to estimate the probability of fire occurrence based on weather conditions.

Project Objective   
Predict the Fire Weather Index using weather parameters.  
Classify the fire risk level as:  
Low  
Moderate  
High  
Provide automatic FWI prediction using real-time weather.  
Create an easy-to-use web application.  

Machine Learning Model   
Dataset Used  
Algerian Forest Fire Dataset  

Contains fields like:  
a. Temperature  
b. Relative Humidity  
c. Wind Speed  
d. Rain  
e. FFMC, DMC, DC, ISI, BUI  
f. FWI (target)  

Steps Performed  
a. Data cleaning  
b. Handling missing values  
c. Removing duplicates  
d. Outlier analysis  
e. Feature scaling using MinMaxScaler  
f. ML Model training (Linear Regression)  
g. Saving model using joblib  

Technology Stack  
a. Scikit-Learn  
b. Pandas  
c. NumPy  
d. Joblib  
e. Streamlit  
f. Custom CSS  
g. FastAPI  

Features Provided a. Input weather manually  
b. FWI prediction using ML  
c. Fire danger level  
e. Auto FWI from live weather  
f. One interactive map  
g. Modern UI with gradient  

Conclusion  
The Fire Weather Index (FWI) Prediction System successfully demonstrates how machine learning, real-time weather data, and geolocation technologies can be integrated to assess fire risk accurately and efficiently. By using the Algerian Forest Fire dataset and a Linear Regression model with strong performance (R² ≈ 0.98), the project delivers reliable predictions based on critical weather variables such as temperature, humidity, wind speed, and drought codes.This project highlights the practical application of machine learning in environmental monitoring and disaster prevention. Deploying such systems can help authorities, forest departments, and local communities take proactive decisions to prevent forest fires and minimize damage.

Overall, the FWI Prediction System is a valuable step toward smart environmental management, providing a scalable foundation for future enhancements such as fire hotspot detection, satellite imagery integration, or mobile deployment.
