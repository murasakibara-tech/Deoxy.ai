How to run DEOXY.AI


-Install Python 3.10 or later
-Open a terminal in this folder
-Run: 'pip install -r requirements.txt'
-Run: 'python app.py'
-Open 'http://127.0.0.1:5000' in your browser
- Basic Mode - Enter temperature, salinity, depth, latitude, longitude, year and month
- Advanced Mode- tick the box to also enter nitrate, phosphate and silicate
-Each input box shows the valid range underneath it
-Click Oxygen Predictor to get a result
-Produce the dissolved oxygen prediction in µmol/kg in addition to a risk band
-Risk bands: Hypoxic (under 60), Low oxygen (60-120), Healthy (120 and above)
-To see the code and training, open 'individual_project.ipynb' in Google Colab 
-The GLODAP Atlantic dataset would not be included so download the Atlantic Ocean file specifically, not any other global or regional products.The link can be found here: https://glodap.info/index.php/merged-and-adjusted-data-product-v2-2023/.
-Don't run the 'joblib.dump' cells in the notebook as they can overwrite trained models supplied here
-scikit-learn is pinned to 1.6.1 in 'requirements.txt', as tree models do vary between versions