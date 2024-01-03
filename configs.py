import streamlit as st

def config():
    
        original_title = '<h1 style="font-family: serif; color:white; font-size: 20px;"></h1>'
        st.markdown(original_title, unsafe_allow_html=True)


        # Set the background image
        background_image = """
        <style>
        [data-testid="stAppViewContainer"] > .main {
            background-image: url("https://www.qwesta.fr/image/partial/m/e/d/2face61b54d5c72fd4d077c7ea258ff0_medecine-travailfDn0I-.jpg");
            background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
            background-position: center;  
            background-repeat: no-repeat;
        }
        </style>
        """

        st.markdown(background_image, unsafe_allow_html=True)