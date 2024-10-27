import streamlit as st

Home = st.Page(
    page = "Pages/Home.py",
    title = "About this project",
    icon = "📝",
    default = True,
)

Project = st.Page(
    page = "Pages/Project.py",
    title = "Project walkthorugh", 
    icon = "📚",
)

Model = st.Page(
    page = "Pages/LiveModel.py",
    title = "Give the model a try!",
    icon = "🔮",
)

pg = st.navigation({"Home":[Home],
     "Project": [Project], 
     "Model":[Model]})
    
pg.run()