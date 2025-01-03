import streamlit as st
import runpy

def main():
    # Main menu options
    menu = ["Neural Network", "Perceptron"]
    choice = st.sidebar.selectbox("Select Main Menu", menu)

    if choice == "Neural Network":
        nn_sub_menu()
    elif choice == "Perceptron":
        p_sub_menu()

def nn_sub_menu():
    st.subheader("Neural Network Menu")

    # Neural Network submenu options
    sub_menu = ["Fish", "Fruit", "Pumpkin"]
    choice = st.sidebar.radio("Select Sub Menu", sub_menu)

    if choice == "Fish":
        st.write("Kamu memilih Neural Network -> Fish.")
        runpy.run_path('supervised/nn_fish.py')
    elif choice == "Fruit":
        st.write("Kamu memilih Neural Network -> Fruit.")
        runpy.run_path('supervised/nn_fruit.py')
    elif choice == "Pumpkin":
        st.write("Kamu memilih Neural Network -> Pumpkin.")
        runpy.run_path('supervised/nn_pumpkin.py')

def p_sub_menu():
    st.subheader("Perceptron Menu")

    # Perceptron submenu options
    sub_menu = ["Fruit"]
    choice = st.sidebar.radio("Select Sub Menu", sub_menu)

    if choice == "Fruit":
        st.write("Perceptron -> Fruit.")
        runpy.run_path('supervised/percep_fruit.py')

if __name__ == "__main__":
    main()
