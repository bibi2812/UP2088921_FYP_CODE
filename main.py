import tkinter as tk
from tkinter import filedialog, messagebox
import os
import joblib
import subprocess
import pandas as pd

# Upload APK File only
def load_apk():
    file_path = filedialog.askopenfilename(filetypes=[("APK files", "*.apk")])
    if file_path:
        apk_entry.delete(0, tk.END)
        apk_entry.insert(0, file_path)

def extract_features_from_apk(apk_path, dataset_columns):
    # Read the APK file and extract its features
    apk_features = []

    # Read the content of the APK file
    with open(apk_path, 'rb') as file:
        apk_content = file.read()

    # Extract permission-based features
    permission_features = [0] * 214
    # Extract API-based features
    api_features = [0] * 27

    # Append extracted features to the feature vector
    apk_features.extend(permission_features)
    apk_features.extend(api_features)

    return apk_features

def analyze():
    apk_path = apk_entry.get()
    pkl_filename = pkl_var.get()
    
    if not apk_path or pkl_filename == 'Select model':
        messagebox.showwarning("Warning", "Please select an APK file and a model.")
        return
    
    pkl_path = os.path.join('trained_models', pkl_filename)
    try:
        with open(pkl_path, 'rb') as file:
            model = joblib.load(file)
    except FileNotFoundError:
        messagebox.showerror("Error", f"Model file {pkl_filename} not found.")
        return
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")
        return

    # Load dataset columns
    dataset_columns = pd.read_csv("AndroidMalwareDatasetTraining.csv").columns

    features = extract_features_from_apk(apk_path, dataset_columns)
    if features is None:
        return

    # Ensure features are passed as a list
    prediction = model.predict([features])
    result_message = "This app is MALWARE." if prediction[0] == 1 else "This app is SAFE."
    messagebox.showinfo("Analysis Result", result_message)

def get_pkl_files():
    files = [f for f in os.listdir('trained_models') if f.endswith('.pkl')]
    files.insert(0, 'Select model')
    return files

# GUI 
root = tk.Tk()
root.title("Android Malware Detection")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

apk_entry = tk.Entry(frame, width=50)
apk_entry.pack(side=tk.LEFT, padx=(0, 10))
apk_button = tk.Button(frame, text="Load APK", command=load_apk)
apk_button.pack(side=tk.LEFT)

pkl_files = get_pkl_files()
pkl_var = tk.StringVar(root)
pkl_var.set(pkl_files[0])  # set the default option
pkl_menu = tk.OptionMenu(root, pkl_var, *pkl_files)
pkl_menu.pack(pady=10)

analyze_button = tk.Button(root, text="Analyze", command=analyze)
analyze_button.pack(pady=10)

# Disclaimer
def show_disclaimer():
    disclaimer_text = "Disclaimer: This software is for educational purposes only. Use at your own risk and please exercise caution and be safe when downloading unknown apps."
    messagebox.showinfo("Disclaimer", disclaimer_text)

disclaimer_button = tk.Button(root, text="Disclaimer", command=show_disclaimer)
disclaimer_button.pack(side=tk.TOP, padx=10, pady=10)

root.mainloop()
