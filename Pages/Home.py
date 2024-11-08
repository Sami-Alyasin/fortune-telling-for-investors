import streamlit as st
# make wide mode the default
st.set_page_config(layout="wide",initial_sidebar_state="expanded")

st.markdown("<h1 style='text-align: center;'>The Crystal Stockball: Fortune Telling for Investors</h1>", unsafe_allow_html=True)
st.divider()

# add a button called next: Project wlakthorugh that will take the user to the next page
left, middle, right = st.columns(3,gap='large')  
with left:
    st.page_link("Pages/Home.py",label = "About", icon = "📝")
    
with middle:
    st.page_link("Pages/Project.py",label = "Project walkthorugh", icon = "📚")  
    
with right:
    st.page_link("Pages/LiveModel.py",label = "Give the model a try!", icon = "🔮")
    

with st.container(border=True):     
    st.header("About this project", divider=True)

    st.markdown('''
        **This is an end-to-end data science project that will deploy a predictive model and explore how the amount of data collected, engineered features used, models selected, and the selection of the parameters of those models contribute to the accuracy and efficiency of the stock price predictive model. The model will generate the next trading day's prediction for the selected stock's adjusted closing price.**

        **This project will cover the following steps:**
        1. **Data collection** - We will collect historical stock price data from Yahoo Finance using the yfinance API.
        2. **Exploratory data analysis (EDA)** - We will explore the historical stock price data to understand the patterns and trends.
        3. **Feature engineering** - We will create new features from the historical stock price data to help predict the stock price movement.
        4. **Modeling** - We will train, optimize, and evaluate a variety of machine learning models.
        5. **Model deployment** - Finally, we will deploy the best performing model as a web application using Streamlit.            
        ''')
    
    st.header("In the works:", divider=True)
    
    st.markdown('''
        **This project is continuously evolving, and the following is what's currently in progress:**

        1. Exploring the use of different time ranges for the historical stock price data. 
        2. Correctly implementing the ARIMA and SARIMA models. The way they are currently implemented is incorrect.
        3. Exploring the use of shorter time ranges for the feature engineering process. Using 5, 10, & 15 days instead of 30,60, 90 days.
        4. Implementing sentiment analysis from news and social media for each stock and study the correlation to the stock price movement.
        5. Implementing PCA (Principle Component Analysis) to reduce the dimensionality of the feature space.
        ''')

with st.container(border=True):     
    st.header("Skills & Tools Utilized",divider=True)
    st.markdown('''
        * Github 
        * Machine Learning Frameworks
        * Python
        * Data Visualization
        * Streamlit
        * Exploratory Data Analysis
        * Generative AI
        * Model Selection, Evaluation, & Deployment
        * Feature Engineering
        * PCA (Principle Component Analysis)
        * Sentiment Analysis
                ''')
    
    # Skills dictionary with icons and descriptions
    skills = {
        "Github": {
            "icon": "https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg"  # Github logo URL
        },
         "Machine Learning Frameworks": {
            "icon": "https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg"  # Scikit-learn logo URL
        },
        "Python": {
            
            "icon": "https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/files/python-logo-only.svg" # Python logo URL
        },
        "Data Visualization": {
            "icon": "https://upload.wikimedia.org/wikipedia/commons/0/01/Created_with_Matplotlib-logo.svg" # Seaborn logo URL
        },
        "Streamlit": {
            "icon": "https://streamlit.io/images/brand/streamlit-mark-color.png"  # Streamlit logo URL
        },
        "Exploratory Data Analysis": {
            "icon": "https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" # Seaborn logo URL
        },
        "Generative AI": {
            "icon": "https://upload.wikimedia.org/wikipedia/commons/1/13/ChatGPT-Logo.png" # Chatpgt logo URL
        },
        "Model Selection, Evaluation, & Deployment": {
            "icon": "https://upload.wikimedia.org/wikipedia/commons/d/d9/LightGBM_logo_black_text.svg" # LGBM logo URL
        },
        "Feature Engineering": {
            "icon": "https://upload.wikimedia.org/wikipedia/commons/3/37/Yahoo_Finance_Logo_2019.png" # yahoo finance logo URL
        }
    }
                        
# add all logos on the same row
# Create a container for logos
st.divider()
logos = [details["icon"] for details in skills.values()]  # Extract logos from the skills dictionary

# Create columns for each logo
logo_cols = st.columns(len(logos))  # Create a column for each logo

# Populate the columns with logos
for index, logo in enumerate(logos):
    with logo_cols[index]:
        st.markdown(f'<img src="{logo}" style="height: 30px;">', unsafe_allow_html=True)

