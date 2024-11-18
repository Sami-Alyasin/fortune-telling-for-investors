import streamlit as st
# make wide mode the default
st.set_page_config(layout="wide",initial_sidebar_state="expanded")

st.sidebar.markdown("On this page:", unsafe_allow_html=True)
st.sidebar.markdown("[About this project](#about-this-project)", unsafe_allow_html=True)
st.sidebar.markdown("[In the works](#in-the-works)", unsafe_allow_html=True)
st.sidebar.markdown("[Skills & Tools Utilized](#skills-tools-utilized)", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Fortune Telling for Investors</h1>", unsafe_allow_html=True)
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
        **This is an end-to-end data science project that will deploy a live stock price prediction model and explore how the amount of data used for training, engineered features used, models selected, and the selection of the parameters of those models contribute to the performance and efficiency of the model. The model will generate the next trading day's prediction for the selected stock's adjusted closing price.**

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

        * Demonstrating the impact of parameter selection on the model's performance. I'm thinking of adding tabs for each model with the different itterations. 
        * Looking into making it easier to navigate the Project walkthrough page, possibly adding more links for the different sections under the modeling section.
        * Adding details about the model deployment process.
        
        ''')

with st.container(border=True):     
    st.header("Skills & Tools Utilized",divider=True)
    st.markdown('''
        * Github 
        * Python
        * Streamlit
        * Machine Learning Frameworks
        * Modeling
            * Model Selection 
            * Model Evaluation 
            * Hyperparameter Tuning
            * Model Deployment
        * Feature Engineering
        * Data Visualization
        * Exploratory Data Analysis
            * Outlier Detection
            * Time Series Analysis
            * Normality Testing
        * Generative AI
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

