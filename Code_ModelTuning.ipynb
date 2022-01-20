{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "YTfbltWs4jaf"
   },
   "source": [
    "# ReneWind\n",
    "\n",
    "Renewable energy sources play an increasingly important role in the global energy mix, as the effort to reduce the environmental impact of energy production increases.\n",
    "\n",
    "Out of all the renewable energy alternatives, wind energy is one of the most developed technologies worldwide. The U.S Department of Energy has put together a guide to achieving operational efficiency using predictive maintenance practices.\n",
    "\n",
    "Predictive maintenance uses sensor information and analysis methods to measure and predict degradation and future component capability. The idea behind predictive maintenance is that failure patterns are predictable and if component failure can be predicted accurately and the component is replaced before it fails, the costs of operation and maintenance will be much lower.\n",
    "\n",
    "The sensors fitted across different machines involved in the process of energy generation collect data related to various environmental factors (temperature, humidity, wind speed, etc.) and additional features related to various parts of the wind turbine (gearbox, tower, blades, break, etc.). \n",
    "\n",
    "\n",
    "\n",
    "## Objective\n",
    "“ReneWind” is a company working on improving the machinery/processes involved in the production of wind energy using machine learning and has collected data of generator failure of wind turbines using sensors. They have shared a ciphered version of the data, as the data collected through sensors is confidential (the type of data collected varies with companies). Data has 40 predictors, 40000 observations in the training set and 10000 in the test set.\n",
    "\n",
    "The objective is to build various classification models, tune them and find the best one that will help identify failures so that the generator could be repaired before failing/breaking and the overall maintenance cost of the generators can be brought down. \n",
    "\n",
    "“1” in the target variables should be considered as “failure” and “0” will represent “No failure”.\n",
    "\n",
    "The nature of predictions made by the classification model will translate as follows:\n",
    "\n",
    "- True positives (TP) are failures correctly predicted by the model.\n",
    "- False negatives (FN) are real failures in a wind turbine where there is no detection by model. \n",
    "- False positives (FP) are detections in a wind turbine where there is no failure. \n",
    "\n",
    "So, the maintenance cost associated with the model would be:\n",
    "\n",
    "**Maintenance cost** = `TP*(Repair cost) + FN*(Replacement cost) + FP*(Inspection cost)`\n",
    "where,\n",
    "\n",
    "- `Replacement cost = $40,000`\n",
    "- `Repair cost = $15,000`\n",
    "- `Inspection cost = $5,000`\n",
    "\n",
    "Here the objective is to reduce the maintenance cost so, we want a metric that could reduce the maintenance cost.\n",
    "\n",
    "- The minimum possible maintenance cost  =  `Actual failures*(Repair cost) = (TP + FN)*(Repair cost)`\n",
    "- The maintenance cost associated with model = `TP*(Repair cost) + FN*(Replacement cost) + FP*(Inspection cost)`\n",
    "\n",
    "So, we will try to maximize the ratio of minimum possible maintenance cost and the maintenance cost associated with the model.\n",
    "\n",
    "The value of this ratio will lie between 0 and 1, the ratio will be 1 only when the maintenance cost associated with the model will be equal to the minimum possible maintenance cost.\n",
    "\n",
    "## Data Description\n",
    "- The data provided is a transformed version of original data which was collected using sensors.\n",
    "- Train.csv - To be used for training and tuning of models. \n",
    "- Test.csv - To be used only for testing the performance of the final best model.\n",
    "- Both the datasets consist of 40 predictor variables and 1 target variable"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "jjFpJBnb4jak"
   },
   "source": [
    "## Importing libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# !pip install nb-black"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# !pip install imblearn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "id": "83D17_Wl4jal"
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 3;\n",
       "                var nbb_unformatted_code = \"# To help with reading and manipulating data\\nimport pandas as pd\\nimport numpy as np\\n\\n# To help with data visualization\\n%matplotlib inline\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns\\n\\n# To be used for missing value imputation\\nfrom sklearn.impute import SimpleImputer\\n\\n# To help with model building\\nfrom sklearn.linear_model import LogisticRegression\\nfrom sklearn.tree import DecisionTreeClassifier\\nfrom sklearn.ensemble import (\\n    AdaBoostClassifier,\\n    GradientBoostingClassifier,\\n    RandomForestClassifier,\\n    BaggingClassifier,\\n)\\nfrom xgboost import XGBClassifier\\n\\n# To get different metric scores, and split data\\nfrom sklearn import metrics\\nfrom sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\\nfrom sklearn.metrics import (\\n    f1_score,\\n    accuracy_score,\\n    recall_score,\\n    precision_score,\\n    confusion_matrix,\\n    roc_auc_score,\\n    plot_confusion_matrix,\\n)\\n\\n# To oversample and undersample data\\nfrom imblearn.over_sampling import SMOTE\\nfrom imblearn.under_sampling import RandomUnderSampler\\n\\n# To be used for data scaling and one hot encoding\\nfrom sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder\\n\\n# To be used for tuning the model\\nfrom sklearn.model_selection import GridSearchCV, RandomizedSearchCV\\n\\n# To use statistical functions\\nimport scipy.stats as stats\\n\\n# To be used for creating pipelines and personalizing them\\nfrom sklearn.pipeline import Pipeline\\nfrom sklearn.compose import ColumnTransformer\\n\\n# To define maximum number of columns to be displayed in a dataframe\\npd.set_option(\\\"display.max_columns\\\", None)\\n\\n# To supress scientific notations for a dataframe\\npd.set_option(\\\"display.float_format\\\", lambda x: \\\"%.3f\\\" % x)\\n\\n# To supress warnings\\nimport warnings\\n\\nwarnings.filterwarnings(\\\"ignore\\\")\\n\\n# This will help in making the Python code more structured automatically (good coding practice)\\n%load_ext nb_black\";\n",
       "                var nbb_formatted_code = \"# To help with reading and manipulating data\\nimport pandas as pd\\nimport numpy as np\\n\\n# To help with data visualization\\n%matplotlib inline\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns\\n\\n# To be used for missing value imputation\\nfrom sklearn.impute import SimpleImputer\\n\\n# To help with model building\\nfrom sklearn.linear_model import LogisticRegression\\nfrom sklearn.tree import DecisionTreeClassifier\\nfrom sklearn.ensemble import (\\n    AdaBoostClassifier,\\n    GradientBoostingClassifier,\\n    RandomForestClassifier,\\n    BaggingClassifier,\\n)\\nfrom xgboost import XGBClassifier\\n\\n# To get different metric scores, and split data\\nfrom sklearn import metrics\\nfrom sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\\nfrom sklearn.metrics import (\\n    f1_score,\\n    accuracy_score,\\n    recall_score,\\n    precision_score,\\n    confusion_matrix,\\n    roc_auc_score,\\n    plot_confusion_matrix,\\n)\\n\\n# To oversample and undersample data\\nfrom imblearn.over_sampling import SMOTE\\nfrom imblearn.under_sampling import RandomUnderSampler\\n\\n# To be used for data scaling and one hot encoding\\nfrom sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder\\n\\n# To be used for tuning the model\\nfrom sklearn.model_selection import GridSearchCV, RandomizedSearchCV\\n\\n# To use statistical functions\\nimport scipy.stats as stats\\n\\n# To be used for creating pipelines and personalizing them\\nfrom sklearn.pipeline import Pipeline\\nfrom sklearn.compose import ColumnTransformer\\n\\n# To define maximum number of columns to be displayed in a dataframe\\npd.set_option(\\\"display.max_columns\\\", None)\\n\\n# To supress scientific notations for a dataframe\\npd.set_option(\\\"display.float_format\\\", lambda x: \\\"%.3f\\\" % x)\\n\\n# To supress warnings\\nimport warnings\\n\\nwarnings.filterwarnings(\\\"ignore\\\")\\n\\n# This will help in making the Python code more structured automatically (good coding practice)\\n%load_ext nb_black\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# To help with reading and manipulating data\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# To help with data visualization\n",
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# To be used for missing value imputation\n",
    "from sklearn.impute import SimpleImputer\n",
    "\n",
    "# To help with model building\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import (\n",
    "    AdaBoostClassifier,\n",
    "    GradientBoostingClassifier,\n",
    "    RandomForestClassifier,\n",
    "    BaggingClassifier,\n",
    ")\n",
    "from xgboost import XGBClassifier\n",
    "\n",
    "# To get different metric scores, and split data\n",
    "from sklearn import metrics\n",
    "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n",
    "from sklearn.metrics import (\n",
    "    f1_score,\n",
    "    accuracy_score,\n",
    "    recall_score,\n",
    "    precision_score,\n",
    "    confusion_matrix,\n",
    "    roc_auc_score,\n",
    "    plot_confusion_matrix,\n",
    ")\n",
    "\n",
    "# To oversample and undersample data\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from imblearn.under_sampling import RandomUnderSampler\n",
    "\n",
    "# To be used for data scaling and one hot encoding\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder\n",
    "\n",
    "# To be used for tuning the model\n",
    "from sklearn.model_selection import GridSearchCV, RandomizedSearchCV\n",
    "\n",
    "# To use statistical functions\n",
    "import scipy.stats as stats\n",
    "\n",
    "# To be used for creating pipelines and personalizing them\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "\n",
    "# To define maximum number of columns to be displayed in a dataframe\n",
    "pd.set_option(\"display.max_columns\", None)\n",
    "\n",
    "# To supress scientific notations for a dataframe\n",
    "pd.set_option(\"display.float_format\", lambda x: \"%.3f\" % x)\n",
    "\n",
    "# To supress warnings\n",
    "import warnings\n",
    "\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "# This will help in making the Python code more structured automatically (good coding practice)\n",
    "%load_ext nb_black"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vqF4q7G94jam"
   },
   "source": [
    "## Loading Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "id": "oJnKoHy14jam"
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 4;\n",
       "                var nbb_unformatted_code = \"# Loading the dataset\\ntrain = pd.read_csv(\\\"train.csv\\\")\";\n",
       "                var nbb_formatted_code = \"# Loading the dataset\\ntrain = pd.read_csv(\\\"train.csv\\\")\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Loading the dataset\n",
    "train = pd.read_csv(\"train.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(40000, 41)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 5;\n",
       "                var nbb_unformatted_code = \"# Checking the number of rows and columns in the data\\ntrain.shape\";\n",
       "                var nbb_formatted_code = \"# Checking the number of rows and columns in the data\\ntrain.shape\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Checking the number of rows and columns in the data\n",
    "train.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- There are 40,000 rows and 41 attributes (including the predictor) in the dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vqF4q7G94jam"
   },
   "source": [
    "## Data Overview"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 6;\n",
       "                var nbb_unformatted_code = \"data = train.copy()\";\n",
       "                var nbb_formatted_code = \"data = train.copy()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "data = train.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>V10</th>\n",
       "      <th>V11</th>\n",
       "      <th>V12</th>\n",
       "      <th>V13</th>\n",
       "      <th>V14</th>\n",
       "      <th>V15</th>\n",
       "      <th>V16</th>\n",
       "      <th>V17</th>\n",
       "      <th>V18</th>\n",
       "      <th>V19</th>\n",
       "      <th>V20</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>V29</th>\n",
       "      <th>V30</th>\n",
       "      <th>V31</th>\n",
       "      <th>V32</th>\n",
       "      <th>V33</th>\n",
       "      <th>V34</th>\n",
       "      <th>V35</th>\n",
       "      <th>V36</th>\n",
       "      <th>V37</th>\n",
       "      <th>V38</th>\n",
       "      <th>V39</th>\n",
       "      <th>V40</th>\n",
       "      <th>Target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-4.465</td>\n",
       "      <td>-4.679</td>\n",
       "      <td>3.102</td>\n",
       "      <td>0.506</td>\n",
       "      <td>-0.221</td>\n",
       "      <td>-2.033</td>\n",
       "      <td>-2.911</td>\n",
       "      <td>0.051</td>\n",
       "      <td>-1.522</td>\n",
       "      <td>3.762</td>\n",
       "      <td>-5.715</td>\n",
       "      <td>0.736</td>\n",
       "      <td>0.981</td>\n",
       "      <td>1.418</td>\n",
       "      <td>-3.376</td>\n",
       "      <td>-3.047</td>\n",
       "      <td>0.306</td>\n",
       "      <td>2.914</td>\n",
       "      <td>2.270</td>\n",
       "      <td>4.395</td>\n",
       "      <td>-2.388</td>\n",
       "      <td>0.646</td>\n",
       "      <td>-1.191</td>\n",
       "      <td>3.133</td>\n",
       "      <td>0.665</td>\n",
       "      <td>-2.511</td>\n",
       "      <td>-0.037</td>\n",
       "      <td>0.726</td>\n",
       "      <td>-3.982</td>\n",
       "      <td>-1.073</td>\n",
       "      <td>1.667</td>\n",
       "      <td>3.060</td>\n",
       "      <td>-1.690</td>\n",
       "      <td>2.846</td>\n",
       "      <td>2.235</td>\n",
       "      <td>6.667</td>\n",
       "      <td>0.444</td>\n",
       "      <td>-2.369</td>\n",
       "      <td>2.951</td>\n",
       "      <td>-3.480</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-2.910</td>\n",
       "      <td>-2.569</td>\n",
       "      <td>4.109</td>\n",
       "      <td>1.317</td>\n",
       "      <td>-1.621</td>\n",
       "      <td>-3.827</td>\n",
       "      <td>-1.617</td>\n",
       "      <td>0.669</td>\n",
       "      <td>0.387</td>\n",
       "      <td>0.854</td>\n",
       "      <td>-6.353</td>\n",
       "      <td>4.272</td>\n",
       "      <td>3.162</td>\n",
       "      <td>0.258</td>\n",
       "      <td>-3.547</td>\n",
       "      <td>-4.285</td>\n",
       "      <td>2.897</td>\n",
       "      <td>1.508</td>\n",
       "      <td>3.668</td>\n",
       "      <td>7.124</td>\n",
       "      <td>-4.096</td>\n",
       "      <td>1.015</td>\n",
       "      <td>-0.970</td>\n",
       "      <td>-0.968</td>\n",
       "      <td>2.064</td>\n",
       "      <td>-1.646</td>\n",
       "      <td>0.427</td>\n",
       "      <td>0.735</td>\n",
       "      <td>-4.470</td>\n",
       "      <td>-2.772</td>\n",
       "      <td>-2.505</td>\n",
       "      <td>-3.783</td>\n",
       "      <td>-6.823</td>\n",
       "      <td>4.909</td>\n",
       "      <td>0.482</td>\n",
       "      <td>5.338</td>\n",
       "      <td>2.381</td>\n",
       "      <td>-3.128</td>\n",
       "      <td>3.527</td>\n",
       "      <td>-3.020</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.284</td>\n",
       "      <td>5.105</td>\n",
       "      <td>6.092</td>\n",
       "      <td>2.640</td>\n",
       "      <td>-1.041</td>\n",
       "      <td>1.308</td>\n",
       "      <td>-1.876</td>\n",
       "      <td>-9.582</td>\n",
       "      <td>3.470</td>\n",
       "      <td>0.763</td>\n",
       "      <td>-2.573</td>\n",
       "      <td>-3.350</td>\n",
       "      <td>-0.595</td>\n",
       "      <td>-5.247</td>\n",
       "      <td>-4.310</td>\n",
       "      <td>-16.232</td>\n",
       "      <td>-1.000</td>\n",
       "      <td>2.318</td>\n",
       "      <td>5.942</td>\n",
       "      <td>-3.858</td>\n",
       "      <td>-11.599</td>\n",
       "      <td>4.021</td>\n",
       "      <td>-6.281</td>\n",
       "      <td>4.633</td>\n",
       "      <td>0.930</td>\n",
       "      <td>6.280</td>\n",
       "      <td>0.851</td>\n",
       "      <td>0.269</td>\n",
       "      <td>-2.206</td>\n",
       "      <td>-1.329</td>\n",
       "      <td>-2.399</td>\n",
       "      <td>-3.098</td>\n",
       "      <td>2.690</td>\n",
       "      <td>-1.643</td>\n",
       "      <td>7.566</td>\n",
       "      <td>-3.198</td>\n",
       "      <td>-3.496</td>\n",
       "      <td>8.105</td>\n",
       "      <td>0.562</td>\n",
       "      <td>-4.227</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3.366</td>\n",
       "      <td>3.653</td>\n",
       "      <td>0.910</td>\n",
       "      <td>-1.368</td>\n",
       "      <td>0.332</td>\n",
       "      <td>2.359</td>\n",
       "      <td>0.733</td>\n",
       "      <td>-4.332</td>\n",
       "      <td>0.566</td>\n",
       "      <td>-0.101</td>\n",
       "      <td>1.914</td>\n",
       "      <td>-0.951</td>\n",
       "      <td>-1.255</td>\n",
       "      <td>-2.707</td>\n",
       "      <td>0.193</td>\n",
       "      <td>-4.769</td>\n",
       "      <td>-2.205</td>\n",
       "      <td>0.908</td>\n",
       "      <td>0.757</td>\n",
       "      <td>-5.834</td>\n",
       "      <td>-3.065</td>\n",
       "      <td>1.597</td>\n",
       "      <td>-1.757</td>\n",
       "      <td>1.766</td>\n",
       "      <td>-0.267</td>\n",
       "      <td>3.625</td>\n",
       "      <td>1.500</td>\n",
       "      <td>-0.586</td>\n",
       "      <td>0.783</td>\n",
       "      <td>-0.201</td>\n",
       "      <td>0.025</td>\n",
       "      <td>-1.795</td>\n",
       "      <td>3.033</td>\n",
       "      <td>-2.468</td>\n",
       "      <td>1.895</td>\n",
       "      <td>-2.298</td>\n",
       "      <td>-1.731</td>\n",
       "      <td>5.909</td>\n",
       "      <td>-0.386</td>\n",
       "      <td>0.616</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-3.832</td>\n",
       "      <td>-5.824</td>\n",
       "      <td>0.634</td>\n",
       "      <td>-2.419</td>\n",
       "      <td>-1.774</td>\n",
       "      <td>1.017</td>\n",
       "      <td>-2.099</td>\n",
       "      <td>-3.173</td>\n",
       "      <td>-2.082</td>\n",
       "      <td>5.393</td>\n",
       "      <td>-0.771</td>\n",
       "      <td>1.107</td>\n",
       "      <td>1.144</td>\n",
       "      <td>0.943</td>\n",
       "      <td>-3.164</td>\n",
       "      <td>-4.248</td>\n",
       "      <td>-4.039</td>\n",
       "      <td>3.689</td>\n",
       "      <td>3.311</td>\n",
       "      <td>1.059</td>\n",
       "      <td>-2.143</td>\n",
       "      <td>1.650</td>\n",
       "      <td>-1.661</td>\n",
       "      <td>1.680</td>\n",
       "      <td>-0.451</td>\n",
       "      <td>-4.551</td>\n",
       "      <td>3.739</td>\n",
       "      <td>1.134</td>\n",
       "      <td>-2.034</td>\n",
       "      <td>0.841</td>\n",
       "      <td>-1.600</td>\n",
       "      <td>-0.257</td>\n",
       "      <td>0.804</td>\n",
       "      <td>4.086</td>\n",
       "      <td>2.292</td>\n",
       "      <td>5.361</td>\n",
       "      <td>0.352</td>\n",
       "      <td>2.940</td>\n",
       "      <td>3.839</td>\n",
       "      <td>-4.309</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      V1     V2    V3     V4     V5     V6     V7     V8     V9    V10    V11  \\\n",
       "0 -4.465 -4.679 3.102  0.506 -0.221 -2.033 -2.911  0.051 -1.522  3.762 -5.715   \n",
       "1 -2.910 -2.569 4.109  1.317 -1.621 -3.827 -1.617  0.669  0.387  0.854 -6.353   \n",
       "2  4.284  5.105 6.092  2.640 -1.041  1.308 -1.876 -9.582  3.470  0.763 -2.573   \n",
       "3  3.366  3.653 0.910 -1.368  0.332  2.359  0.733 -4.332  0.566 -0.101  1.914   \n",
       "4 -3.832 -5.824 0.634 -2.419 -1.774  1.017 -2.099 -3.173 -2.082  5.393 -0.771   \n",
       "\n",
       "     V12    V13    V14    V15     V16    V17   V18   V19    V20     V21   V22  \\\n",
       "0  0.736  0.981  1.418 -3.376  -3.047  0.306 2.914 2.270  4.395  -2.388 0.646   \n",
       "1  4.272  3.162  0.258 -3.547  -4.285  2.897 1.508 3.668  7.124  -4.096 1.015   \n",
       "2 -3.350 -0.595 -5.247 -4.310 -16.232 -1.000 2.318 5.942 -3.858 -11.599 4.021   \n",
       "3 -0.951 -1.255 -2.707  0.193  -4.769 -2.205 0.908 0.757 -5.834  -3.065 1.597   \n",
       "4  1.107  1.144  0.943 -3.164  -4.248 -4.039 3.689 3.311  1.059  -2.143 1.650   \n",
       "\n",
       "     V23    V24    V25    V26    V27    V28    V29    V30    V31    V32  \\\n",
       "0 -1.191  3.133  0.665 -2.511 -0.037  0.726 -3.982 -1.073  1.667  3.060   \n",
       "1 -0.970 -0.968  2.064 -1.646  0.427  0.735 -4.470 -2.772 -2.505 -3.783   \n",
       "2 -6.281  4.633  0.930  6.280  0.851  0.269 -2.206 -1.329 -2.399 -3.098   \n",
       "3 -1.757  1.766 -0.267  3.625  1.500 -0.586  0.783 -0.201  0.025 -1.795   \n",
       "4 -1.661  1.680 -0.451 -4.551  3.739  1.134 -2.034  0.841 -1.600 -0.257   \n",
       "\n",
       "     V33    V34   V35    V36    V37    V38    V39    V40  Target  \n",
       "0 -1.690  2.846 2.235  6.667  0.444 -2.369  2.951 -3.480       0  \n",
       "1 -6.823  4.909 0.482  5.338  2.381 -3.128  3.527 -3.020       0  \n",
       "2  2.690 -1.643 7.566 -3.198 -3.496  8.105  0.562 -4.227       0  \n",
       "3  3.033 -2.468 1.895 -2.298 -1.731  5.909 -0.386  0.616       0  \n",
       "4  0.804  4.086 2.292  5.361  0.352  2.940  3.839 -4.309       0  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 7;\n",
       "                var nbb_unformatted_code = \"# let's view the first 5 rows of the data\\ndata.head()\";\n",
       "                var nbb_formatted_code = \"# let's view the first 5 rows of the data\\ndata.head()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# let's view the first 5 rows of the data\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>V10</th>\n",
       "      <th>V11</th>\n",
       "      <th>V12</th>\n",
       "      <th>V13</th>\n",
       "      <th>V14</th>\n",
       "      <th>V15</th>\n",
       "      <th>V16</th>\n",
       "      <th>V17</th>\n",
       "      <th>V18</th>\n",
       "      <th>V19</th>\n",
       "      <th>V20</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>V29</th>\n",
       "      <th>V30</th>\n",
       "      <th>V31</th>\n",
       "      <th>V32</th>\n",
       "      <th>V33</th>\n",
       "      <th>V34</th>\n",
       "      <th>V35</th>\n",
       "      <th>V36</th>\n",
       "      <th>V37</th>\n",
       "      <th>V38</th>\n",
       "      <th>V39</th>\n",
       "      <th>V40</th>\n",
       "      <th>Target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>39995</th>\n",
       "      <td>-3.897</td>\n",
       "      <td>-3.942</td>\n",
       "      <td>-0.351</td>\n",
       "      <td>-2.417</td>\n",
       "      <td>1.108</td>\n",
       "      <td>-1.528</td>\n",
       "      <td>-3.520</td>\n",
       "      <td>2.055</td>\n",
       "      <td>-0.234</td>\n",
       "      <td>-0.358</td>\n",
       "      <td>-3.782</td>\n",
       "      <td>2.180</td>\n",
       "      <td>6.112</td>\n",
       "      <td>1.985</td>\n",
       "      <td>-8.330</td>\n",
       "      <td>-1.639</td>\n",
       "      <td>-0.915</td>\n",
       "      <td>5.672</td>\n",
       "      <td>-3.924</td>\n",
       "      <td>2.133</td>\n",
       "      <td>-4.502</td>\n",
       "      <td>2.777</td>\n",
       "      <td>5.728</td>\n",
       "      <td>1.620</td>\n",
       "      <td>-1.700</td>\n",
       "      <td>-0.042</td>\n",
       "      <td>-2.923</td>\n",
       "      <td>-2.760</td>\n",
       "      <td>-2.254</td>\n",
       "      <td>2.552</td>\n",
       "      <td>0.982</td>\n",
       "      <td>7.112</td>\n",
       "      <td>1.476</td>\n",
       "      <td>-3.954</td>\n",
       "      <td>1.856</td>\n",
       "      <td>5.029</td>\n",
       "      <td>2.083</td>\n",
       "      <td>-6.409</td>\n",
       "      <td>1.477</td>\n",
       "      <td>-0.874</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39996</th>\n",
       "      <td>-3.187</td>\n",
       "      <td>-10.052</td>\n",
       "      <td>5.696</td>\n",
       "      <td>-4.370</td>\n",
       "      <td>-5.355</td>\n",
       "      <td>-1.873</td>\n",
       "      <td>-3.947</td>\n",
       "      <td>0.679</td>\n",
       "      <td>-2.389</td>\n",
       "      <td>5.457</td>\n",
       "      <td>1.583</td>\n",
       "      <td>3.571</td>\n",
       "      <td>9.227</td>\n",
       "      <td>2.554</td>\n",
       "      <td>-7.039</td>\n",
       "      <td>-0.994</td>\n",
       "      <td>-9.665</td>\n",
       "      <td>1.155</td>\n",
       "      <td>3.877</td>\n",
       "      <td>3.524</td>\n",
       "      <td>-7.015</td>\n",
       "      <td>-0.132</td>\n",
       "      <td>-3.446</td>\n",
       "      <td>-4.801</td>\n",
       "      <td>-0.876</td>\n",
       "      <td>-3.812</td>\n",
       "      <td>5.422</td>\n",
       "      <td>-3.732</td>\n",
       "      <td>0.609</td>\n",
       "      <td>5.256</td>\n",
       "      <td>1.915</td>\n",
       "      <td>0.403</td>\n",
       "      <td>3.164</td>\n",
       "      <td>3.752</td>\n",
       "      <td>8.530</td>\n",
       "      <td>8.451</td>\n",
       "      <td>0.204</td>\n",
       "      <td>-7.130</td>\n",
       "      <td>4.249</td>\n",
       "      <td>-6.112</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39997</th>\n",
       "      <td>-2.687</td>\n",
       "      <td>1.961</td>\n",
       "      <td>6.137</td>\n",
       "      <td>2.600</td>\n",
       "      <td>2.657</td>\n",
       "      <td>-4.291</td>\n",
       "      <td>-2.344</td>\n",
       "      <td>0.974</td>\n",
       "      <td>-1.027</td>\n",
       "      <td>0.497</td>\n",
       "      <td>-9.589</td>\n",
       "      <td>3.177</td>\n",
       "      <td>1.055</td>\n",
       "      <td>-1.416</td>\n",
       "      <td>-4.669</td>\n",
       "      <td>-5.405</td>\n",
       "      <td>3.720</td>\n",
       "      <td>2.893</td>\n",
       "      <td>2.329</td>\n",
       "      <td>1.458</td>\n",
       "      <td>-6.429</td>\n",
       "      <td>1.818</td>\n",
       "      <td>0.806</td>\n",
       "      <td>7.786</td>\n",
       "      <td>0.331</td>\n",
       "      <td>5.257</td>\n",
       "      <td>-4.867</td>\n",
       "      <td>-0.819</td>\n",
       "      <td>-5.667</td>\n",
       "      <td>-2.861</td>\n",
       "      <td>4.674</td>\n",
       "      <td>6.621</td>\n",
       "      <td>-1.989</td>\n",
       "      <td>-1.349</td>\n",
       "      <td>3.952</td>\n",
       "      <td>5.450</td>\n",
       "      <td>-0.455</td>\n",
       "      <td>-2.202</td>\n",
       "      <td>1.678</td>\n",
       "      <td>-1.974</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39998</th>\n",
       "      <td>0.521</td>\n",
       "      <td>0.096</td>\n",
       "      <td>8.457</td>\n",
       "      <td>2.138</td>\n",
       "      <td>-1.636</td>\n",
       "      <td>-2.713</td>\n",
       "      <td>-2.693</td>\n",
       "      <td>-3.410</td>\n",
       "      <td>1.936</td>\n",
       "      <td>2.012</td>\n",
       "      <td>-4.989</td>\n",
       "      <td>-0.819</td>\n",
       "      <td>4.166</td>\n",
       "      <td>-1.192</td>\n",
       "      <td>-5.033</td>\n",
       "      <td>-8.523</td>\n",
       "      <td>-1.950</td>\n",
       "      <td>0.017</td>\n",
       "      <td>4.505</td>\n",
       "      <td>2.031</td>\n",
       "      <td>-8.849</td>\n",
       "      <td>0.566</td>\n",
       "      <td>-6.040</td>\n",
       "      <td>-0.043</td>\n",
       "      <td>1.656</td>\n",
       "      <td>4.250</td>\n",
       "      <td>1.727</td>\n",
       "      <td>-1.686</td>\n",
       "      <td>-3.963</td>\n",
       "      <td>-2.642</td>\n",
       "      <td>1.939</td>\n",
       "      <td>-1.257</td>\n",
       "      <td>-1.136</td>\n",
       "      <td>1.434</td>\n",
       "      <td>5.905</td>\n",
       "      <td>3.752</td>\n",
       "      <td>-1.867</td>\n",
       "      <td>-1.918</td>\n",
       "      <td>2.573</td>\n",
       "      <td>-5.019</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39999</th>\n",
       "      <td>2.403</td>\n",
       "      <td>-1.336</td>\n",
       "      <td>6.451</td>\n",
       "      <td>-5.356</td>\n",
       "      <td>-0.434</td>\n",
       "      <td>0.255</td>\n",
       "      <td>-1.120</td>\n",
       "      <td>-2.523</td>\n",
       "      <td>-0.654</td>\n",
       "      <td>2.316</td>\n",
       "      <td>-2.862</td>\n",
       "      <td>0.199</td>\n",
       "      <td>1.593</td>\n",
       "      <td>-0.337</td>\n",
       "      <td>-0.709</td>\n",
       "      <td>-4.408</td>\n",
       "      <td>-3.683</td>\n",
       "      <td>2.973</td>\n",
       "      <td>-1.223</td>\n",
       "      <td>-1.958</td>\n",
       "      <td>-4.454</td>\n",
       "      <td>0.464</td>\n",
       "      <td>-4.952</td>\n",
       "      <td>-1.624</td>\n",
       "      <td>2.965</td>\n",
       "      <td>2.009</td>\n",
       "      <td>5.712</td>\n",
       "      <td>-2.910</td>\n",
       "      <td>-2.287</td>\n",
       "      <td>-3.676</td>\n",
       "      <td>5.678</td>\n",
       "      <td>-4.310</td>\n",
       "      <td>-0.709</td>\n",
       "      <td>-1.359</td>\n",
       "      <td>1.639</td>\n",
       "      <td>7.766</td>\n",
       "      <td>-0.245</td>\n",
       "      <td>-1.124</td>\n",
       "      <td>2.872</td>\n",
       "      <td>1.902</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          V1      V2     V3     V4     V5     V6     V7     V8     V9    V10  \\\n",
       "39995 -3.897  -3.942 -0.351 -2.417  1.108 -1.528 -3.520  2.055 -0.234 -0.358   \n",
       "39996 -3.187 -10.052  5.696 -4.370 -5.355 -1.873 -3.947  0.679 -2.389  5.457   \n",
       "39997 -2.687   1.961  6.137  2.600  2.657 -4.291 -2.344  0.974 -1.027  0.497   \n",
       "39998  0.521   0.096  8.457  2.138 -1.636 -2.713 -2.693 -3.410  1.936  2.012   \n",
       "39999  2.403  -1.336  6.451 -5.356 -0.434  0.255 -1.120 -2.523 -0.654  2.316   \n",
       "\n",
       "         V11    V12   V13    V14    V15    V16    V17   V18    V19    V20  \\\n",
       "39995 -3.782  2.180 6.112  1.985 -8.330 -1.639 -0.915 5.672 -3.924  2.133   \n",
       "39996  1.583  3.571 9.227  2.554 -7.039 -0.994 -9.665 1.155  3.877  3.524   \n",
       "39997 -9.589  3.177 1.055 -1.416 -4.669 -5.405  3.720 2.893  2.329  1.458   \n",
       "39998 -4.989 -0.819 4.166 -1.192 -5.033 -8.523 -1.950 0.017  4.505  2.031   \n",
       "39999 -2.862  0.199 1.593 -0.337 -0.709 -4.408 -3.683 2.973 -1.223 -1.958   \n",
       "\n",
       "         V21    V22    V23    V24    V25    V26    V27    V28    V29    V30  \\\n",
       "39995 -4.502  2.777  5.728  1.620 -1.700 -0.042 -2.923 -2.760 -2.254  2.552   \n",
       "39996 -7.015 -0.132 -3.446 -4.801 -0.876 -3.812  5.422 -3.732  0.609  5.256   \n",
       "39997 -6.429  1.818  0.806  7.786  0.331  5.257 -4.867 -0.819 -5.667 -2.861   \n",
       "39998 -8.849  0.566 -6.040 -0.043  1.656  4.250  1.727 -1.686 -3.963 -2.642   \n",
       "39999 -4.454  0.464 -4.952 -1.624  2.965  2.009  5.712 -2.910 -2.287 -3.676   \n",
       "\n",
       "        V31    V32    V33    V34   V35   V36    V37    V38   V39    V40  \\\n",
       "39995 0.982  7.112  1.476 -3.954 1.856 5.029  2.083 -6.409 1.477 -0.874   \n",
       "39996 1.915  0.403  3.164  3.752 8.530 8.451  0.204 -7.130 4.249 -6.112   \n",
       "39997 4.674  6.621 -1.989 -1.349 3.952 5.450 -0.455 -2.202 1.678 -1.974   \n",
       "39998 1.939 -1.257 -1.136  1.434 5.905 3.752 -1.867 -1.918 2.573 -5.019   \n",
       "39999 5.678 -4.310 -0.709 -1.359 1.639 7.766 -0.245 -1.124 2.872  1.902   \n",
       "\n",
       "       Target  \n",
       "39995       0  \n",
       "39996       0  \n",
       "39997       0  \n",
       "39998       0  \n",
       "39999       0  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 8;\n",
       "                var nbb_unformatted_code = \"# let's view the last 5 rows of the data\\ndata.tail()\";\n",
       "                var nbb_formatted_code = \"# let's view the last 5 rows of the data\\ndata.tail()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# let's view the last 5 rows of the data\n",
    "data.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The attributes are ciphered"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 40000 entries, 0 to 39999\n",
      "Data columns (total 41 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   V1      39954 non-null  float64\n",
      " 1   V2      39961 non-null  float64\n",
      " 2   V3      40000 non-null  float64\n",
      " 3   V4      40000 non-null  float64\n",
      " 4   V5      40000 non-null  float64\n",
      " 5   V6      40000 non-null  float64\n",
      " 6   V7      40000 non-null  float64\n",
      " 7   V8      40000 non-null  float64\n",
      " 8   V9      40000 non-null  float64\n",
      " 9   V10     40000 non-null  float64\n",
      " 10  V11     40000 non-null  float64\n",
      " 11  V12     40000 non-null  float64\n",
      " 12  V13     40000 non-null  float64\n",
      " 13  V14     40000 non-null  float64\n",
      " 14  V15     40000 non-null  float64\n",
      " 15  V16     40000 non-null  float64\n",
      " 16  V17     40000 non-null  float64\n",
      " 17  V18     40000 non-null  float64\n",
      " 18  V19     40000 non-null  float64\n",
      " 19  V20     40000 non-null  float64\n",
      " 20  V21     40000 non-null  float64\n",
      " 21  V22     40000 non-null  float64\n",
      " 22  V23     40000 non-null  float64\n",
      " 23  V24     40000 non-null  float64\n",
      " 24  V25     40000 non-null  float64\n",
      " 25  V26     40000 non-null  float64\n",
      " 26  V27     40000 non-null  float64\n",
      " 27  V28     40000 non-null  float64\n",
      " 28  V29     40000 non-null  float64\n",
      " 29  V30     40000 non-null  float64\n",
      " 30  V31     40000 non-null  float64\n",
      " 31  V32     40000 non-null  float64\n",
      " 32  V33     40000 non-null  float64\n",
      " 33  V34     40000 non-null  float64\n",
      " 34  V35     40000 non-null  float64\n",
      " 35  V36     40000 non-null  float64\n",
      " 36  V37     40000 non-null  float64\n",
      " 37  V38     40000 non-null  float64\n",
      " 38  V39     40000 non-null  float64\n",
      " 39  V40     40000 non-null  float64\n",
      " 40  Target  40000 non-null  int64  \n",
      "dtypes: float64(40), int64(1)\n",
      "memory usage: 12.5 MB\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 9;\n",
       "                var nbb_unformatted_code = \"# let's check the data types of the columns in the dataset\\ndata.info()\";\n",
       "                var nbb_formatted_code = \"# let's check the data types of the columns in the dataset\\ndata.info()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# let's check the data types of the columns in the dataset\n",
    "data.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- All attributes except for the predictor \"Target\" are of float type\n",
    "- There are 46 missing values for attribute \"V1\" and 39 missing values for attribute \"V2\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 10;\n",
       "                var nbb_unformatted_code = \"# let's check for duplicate values in the data\\ndata.duplicated().sum()\";\n",
       "                var nbb_formatted_code = \"# let's check for duplicate values in the data\\ndata.duplicated().sum()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# let's check for duplicate values in the data\n",
    "data.duplicated().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- There are no duplicate values in the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "V1        39954\n",
       "V2        39961\n",
       "V3        40000\n",
       "V4        40000\n",
       "V5        40000\n",
       "V6        40000\n",
       "V7        40000\n",
       "V8        40000\n",
       "V9        40000\n",
       "V10       40000\n",
       "V11       40000\n",
       "V12       40000\n",
       "V13       40000\n",
       "V14       40000\n",
       "V15       39999\n",
       "V16       40000\n",
       "V17       40000\n",
       "V18       40000\n",
       "V19       40000\n",
       "V20       40000\n",
       "V21       40000\n",
       "V22       40000\n",
       "V23       40000\n",
       "V24       40000\n",
       "V25       40000\n",
       "V26       40000\n",
       "V27       40000\n",
       "V28       40000\n",
       "V29       40000\n",
       "V30       40000\n",
       "V31       40000\n",
       "V32       40000\n",
       "V33       40000\n",
       "V34       40000\n",
       "V35       40000\n",
       "V36       40000\n",
       "V37       40000\n",
       "V38       40000\n",
       "V39       40000\n",
       "V40       40000\n",
       "Target        2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 11;\n",
       "                var nbb_unformatted_code = \"# let's check for number of unique values in each column\\ndata.nunique()\";\n",
       "                var nbb_formatted_code = \"# let's check for number of unique values in each column\\ndata.nunique()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# let's check for number of unique values in each column\n",
    "data.nunique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- All attributes except \"Target\" have all unique values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>V10</th>\n",
       "      <th>V11</th>\n",
       "      <th>V12</th>\n",
       "      <th>V13</th>\n",
       "      <th>V14</th>\n",
       "      <th>V15</th>\n",
       "      <th>V16</th>\n",
       "      <th>V17</th>\n",
       "      <th>V18</th>\n",
       "      <th>V19</th>\n",
       "      <th>V20</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>V29</th>\n",
       "      <th>V30</th>\n",
       "      <th>V31</th>\n",
       "      <th>V32</th>\n",
       "      <th>V33</th>\n",
       "      <th>V34</th>\n",
       "      <th>V35</th>\n",
       "      <th>V36</th>\n",
       "      <th>V37</th>\n",
       "      <th>V38</th>\n",
       "      <th>V39</th>\n",
       "      <th>V40</th>\n",
       "      <th>Target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>39954.000</td>\n",
       "      <td>39961.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "      <td>40000.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>-0.288</td>\n",
       "      <td>0.443</td>\n",
       "      <td>2.506</td>\n",
       "      <td>-0.066</td>\n",
       "      <td>-0.045</td>\n",
       "      <td>-1.001</td>\n",
       "      <td>-0.893</td>\n",
       "      <td>-0.563</td>\n",
       "      <td>-0.008</td>\n",
       "      <td>-0.002</td>\n",
       "      <td>-1.918</td>\n",
       "      <td>1.578</td>\n",
       "      <td>1.591</td>\n",
       "      <td>-0.947</td>\n",
       "      <td>-2.436</td>\n",
       "      <td>-2.943</td>\n",
       "      <td>-0.143</td>\n",
       "      <td>1.189</td>\n",
       "      <td>1.181</td>\n",
       "      <td>0.027</td>\n",
       "      <td>-3.621</td>\n",
       "      <td>0.943</td>\n",
       "      <td>-0.388</td>\n",
       "      <td>1.142</td>\n",
       "      <td>-0.003</td>\n",
       "      <td>1.896</td>\n",
       "      <td>-0.617</td>\n",
       "      <td>-0.888</td>\n",
       "      <td>-1.005</td>\n",
       "      <td>-0.033</td>\n",
       "      <td>0.506</td>\n",
       "      <td>0.327</td>\n",
       "      <td>0.057</td>\n",
       "      <td>-0.464</td>\n",
       "      <td>2.235</td>\n",
       "      <td>1.530</td>\n",
       "      <td>-0.000</td>\n",
       "      <td>-0.351</td>\n",
       "      <td>0.900</td>\n",
       "      <td>-0.897</td>\n",
       "      <td>0.055</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>3.449</td>\n",
       "      <td>3.139</td>\n",
       "      <td>3.406</td>\n",
       "      <td>3.437</td>\n",
       "      <td>2.107</td>\n",
       "      <td>2.037</td>\n",
       "      <td>1.757</td>\n",
       "      <td>3.299</td>\n",
       "      <td>2.162</td>\n",
       "      <td>2.183</td>\n",
       "      <td>3.116</td>\n",
       "      <td>2.915</td>\n",
       "      <td>2.865</td>\n",
       "      <td>1.788</td>\n",
       "      <td>3.341</td>\n",
       "      <td>4.212</td>\n",
       "      <td>3.344</td>\n",
       "      <td>2.586</td>\n",
       "      <td>3.395</td>\n",
       "      <td>3.675</td>\n",
       "      <td>3.557</td>\n",
       "      <td>1.646</td>\n",
       "      <td>4.052</td>\n",
       "      <td>3.913</td>\n",
       "      <td>2.025</td>\n",
       "      <td>3.421</td>\n",
       "      <td>4.392</td>\n",
       "      <td>1.925</td>\n",
       "      <td>2.676</td>\n",
       "      <td>3.031</td>\n",
       "      <td>3.483</td>\n",
       "      <td>5.499</td>\n",
       "      <td>3.574</td>\n",
       "      <td>3.186</td>\n",
       "      <td>2.924</td>\n",
       "      <td>3.820</td>\n",
       "      <td>1.778</td>\n",
       "      <td>3.964</td>\n",
       "      <td>1.751</td>\n",
       "      <td>2.998</td>\n",
       "      <td>0.227</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>-13.502</td>\n",
       "      <td>-13.212</td>\n",
       "      <td>-11.469</td>\n",
       "      <td>-16.015</td>\n",
       "      <td>-8.613</td>\n",
       "      <td>-10.227</td>\n",
       "      <td>-8.206</td>\n",
       "      <td>-15.658</td>\n",
       "      <td>-8.596</td>\n",
       "      <td>-11.001</td>\n",
       "      <td>-14.832</td>\n",
       "      <td>-13.619</td>\n",
       "      <td>-13.830</td>\n",
       "      <td>-8.309</td>\n",
       "      <td>-17.202</td>\n",
       "      <td>-21.919</td>\n",
       "      <td>-17.634</td>\n",
       "      <td>-11.644</td>\n",
       "      <td>-13.492</td>\n",
       "      <td>-13.923</td>\n",
       "      <td>-19.436</td>\n",
       "      <td>-10.122</td>\n",
       "      <td>-16.188</td>\n",
       "      <td>-18.488</td>\n",
       "      <td>-8.228</td>\n",
       "      <td>-12.588</td>\n",
       "      <td>-14.905</td>\n",
       "      <td>-9.685</td>\n",
       "      <td>-12.579</td>\n",
       "      <td>-14.796</td>\n",
       "      <td>-19.377</td>\n",
       "      <td>-23.201</td>\n",
       "      <td>-17.454</td>\n",
       "      <td>-17.985</td>\n",
       "      <td>-15.350</td>\n",
       "      <td>-17.479</td>\n",
       "      <td>-7.640</td>\n",
       "      <td>-17.375</td>\n",
       "      <td>-7.136</td>\n",
       "      <td>-11.930</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>-2.751</td>\n",
       "      <td>-1.638</td>\n",
       "      <td>0.203</td>\n",
       "      <td>-2.350</td>\n",
       "      <td>-1.507</td>\n",
       "      <td>-2.363</td>\n",
       "      <td>-2.037</td>\n",
       "      <td>-2.660</td>\n",
       "      <td>-1.494</td>\n",
       "      <td>-1.391</td>\n",
       "      <td>-3.941</td>\n",
       "      <td>-0.431</td>\n",
       "      <td>-0.209</td>\n",
       "      <td>-2.165</td>\n",
       "      <td>-4.451</td>\n",
       "      <td>-5.632</td>\n",
       "      <td>-2.227</td>\n",
       "      <td>-0.403</td>\n",
       "      <td>-1.051</td>\n",
       "      <td>-2.434</td>\n",
       "      <td>-5.921</td>\n",
       "      <td>-0.112</td>\n",
       "      <td>-3.119</td>\n",
       "      <td>-1.483</td>\n",
       "      <td>-1.373</td>\n",
       "      <td>-0.319</td>\n",
       "      <td>-3.692</td>\n",
       "      <td>-2.193</td>\n",
       "      <td>-2.799</td>\n",
       "      <td>-1.908</td>\n",
       "      <td>-1.799</td>\n",
       "      <td>-3.392</td>\n",
       "      <td>-2.238</td>\n",
       "      <td>-2.128</td>\n",
       "      <td>0.332</td>\n",
       "      <td>-0.937</td>\n",
       "      <td>-1.266</td>\n",
       "      <td>-3.017</td>\n",
       "      <td>-0.262</td>\n",
       "      <td>-2.950</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>-0.774</td>\n",
       "      <td>0.464</td>\n",
       "      <td>2.265</td>\n",
       "      <td>-0.124</td>\n",
       "      <td>-0.097</td>\n",
       "      <td>-1.007</td>\n",
       "      <td>-0.935</td>\n",
       "      <td>-0.384</td>\n",
       "      <td>-0.052</td>\n",
       "      <td>0.106</td>\n",
       "      <td>-1.942</td>\n",
       "      <td>1.485</td>\n",
       "      <td>1.654</td>\n",
       "      <td>-0.957</td>\n",
       "      <td>-2.399</td>\n",
       "      <td>-2.719</td>\n",
       "      <td>-0.028</td>\n",
       "      <td>0.867</td>\n",
       "      <td>1.278</td>\n",
       "      <td>0.030</td>\n",
       "      <td>-3.559</td>\n",
       "      <td>0.963</td>\n",
       "      <td>-0.275</td>\n",
       "      <td>0.964</td>\n",
       "      <td>0.021</td>\n",
       "      <td>1.964</td>\n",
       "      <td>-0.910</td>\n",
       "      <td>-0.905</td>\n",
       "      <td>-1.206</td>\n",
       "      <td>0.185</td>\n",
       "      <td>0.491</td>\n",
       "      <td>0.056</td>\n",
       "      <td>-0.050</td>\n",
       "      <td>-0.251</td>\n",
       "      <td>2.110</td>\n",
       "      <td>1.572</td>\n",
       "      <td>-0.133</td>\n",
       "      <td>-0.319</td>\n",
       "      <td>0.921</td>\n",
       "      <td>-0.949</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.837</td>\n",
       "      <td>2.538</td>\n",
       "      <td>4.585</td>\n",
       "      <td>2.149</td>\n",
       "      <td>1.346</td>\n",
       "      <td>0.374</td>\n",
       "      <td>0.207</td>\n",
       "      <td>1.714</td>\n",
       "      <td>1.426</td>\n",
       "      <td>1.486</td>\n",
       "      <td>0.089</td>\n",
       "      <td>3.541</td>\n",
       "      <td>3.476</td>\n",
       "      <td>0.266</td>\n",
       "      <td>-0.382</td>\n",
       "      <td>-0.113</td>\n",
       "      <td>2.072</td>\n",
       "      <td>2.564</td>\n",
       "      <td>3.497</td>\n",
       "      <td>2.513</td>\n",
       "      <td>-1.284</td>\n",
       "      <td>2.018</td>\n",
       "      <td>2.438</td>\n",
       "      <td>3.563</td>\n",
       "      <td>1.400</td>\n",
       "      <td>4.163</td>\n",
       "      <td>2.201</td>\n",
       "      <td>0.377</td>\n",
       "      <td>0.604</td>\n",
       "      <td>2.040</td>\n",
       "      <td>2.778</td>\n",
       "      <td>3.789</td>\n",
       "      <td>2.256</td>\n",
       "      <td>1.433</td>\n",
       "      <td>4.045</td>\n",
       "      <td>3.997</td>\n",
       "      <td>1.161</td>\n",
       "      <td>2.291</td>\n",
       "      <td>2.069</td>\n",
       "      <td>1.092</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>17.437</td>\n",
       "      <td>13.089</td>\n",
       "      <td>18.366</td>\n",
       "      <td>13.280</td>\n",
       "      <td>9.403</td>\n",
       "      <td>7.065</td>\n",
       "      <td>8.006</td>\n",
       "      <td>11.679</td>\n",
       "      <td>8.507</td>\n",
       "      <td>8.108</td>\n",
       "      <td>13.852</td>\n",
       "      <td>15.754</td>\n",
       "      <td>15.420</td>\n",
       "      <td>6.213</td>\n",
       "      <td>12.875</td>\n",
       "      <td>13.583</td>\n",
       "      <td>17.405</td>\n",
       "      <td>13.180</td>\n",
       "      <td>16.059</td>\n",
       "      <td>16.052</td>\n",
       "      <td>13.840</td>\n",
       "      <td>7.410</td>\n",
       "      <td>15.080</td>\n",
       "      <td>19.769</td>\n",
       "      <td>8.223</td>\n",
       "      <td>16.836</td>\n",
       "      <td>21.595</td>\n",
       "      <td>6.907</td>\n",
       "      <td>11.852</td>\n",
       "      <td>13.191</td>\n",
       "      <td>17.255</td>\n",
       "      <td>24.848</td>\n",
       "      <td>16.692</td>\n",
       "      <td>14.358</td>\n",
       "      <td>16.805</td>\n",
       "      <td>19.330</td>\n",
       "      <td>7.803</td>\n",
       "      <td>15.964</td>\n",
       "      <td>7.998</td>\n",
       "      <td>10.654</td>\n",
       "      <td>1.000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             V1        V2        V3        V4        V5        V6        V7  \\\n",
       "count 39954.000 39961.000 40000.000 40000.000 40000.000 40000.000 40000.000   \n",
       "mean     -0.288     0.443     2.506    -0.066    -0.045    -1.001    -0.893   \n",
       "std       3.449     3.139     3.406     3.437     2.107     2.037     1.757   \n",
       "min     -13.502   -13.212   -11.469   -16.015    -8.613   -10.227    -8.206   \n",
       "25%      -2.751    -1.638     0.203    -2.350    -1.507    -2.363    -2.037   \n",
       "50%      -0.774     0.464     2.265    -0.124    -0.097    -1.007    -0.935   \n",
       "75%       1.837     2.538     4.585     2.149     1.346     0.374     0.207   \n",
       "max      17.437    13.089    18.366    13.280     9.403     7.065     8.006   \n",
       "\n",
       "             V8        V9       V10       V11       V12       V13       V14  \\\n",
       "count 40000.000 40000.000 40000.000 40000.000 40000.000 40000.000 40000.000   \n",
       "mean     -0.563    -0.008    -0.002    -1.918     1.578     1.591    -0.947   \n",
       "std       3.299     2.162     2.183     3.116     2.915     2.865     1.788   \n",
       "min     -15.658    -8.596   -11.001   -14.832   -13.619   -13.830    -8.309   \n",
       "25%      -2.660    -1.494    -1.391    -3.941    -0.431    -0.209    -2.165   \n",
       "50%      -0.384    -0.052     0.106    -1.942     1.485     1.654    -0.957   \n",
       "75%       1.714     1.426     1.486     0.089     3.541     3.476     0.266   \n",
       "max      11.679     8.507     8.108    13.852    15.754    15.420     6.213   \n",
       "\n",
       "            V15       V16       V17       V18       V19       V20       V21  \\\n",
       "count 40000.000 40000.000 40000.000 40000.000 40000.000 40000.000 40000.000   \n",
       "mean     -2.436    -2.943    -0.143     1.189     1.181     0.027    -3.621   \n",
       "std       3.341     4.212     3.344     2.586     3.395     3.675     3.557   \n",
       "min     -17.202   -21.919   -17.634   -11.644   -13.492   -13.923   -19.436   \n",
       "25%      -4.451    -5.632    -2.227    -0.403    -1.051    -2.434    -5.921   \n",
       "50%      -2.399    -2.719    -0.028     0.867     1.278     0.030    -3.559   \n",
       "75%      -0.382    -0.113     2.072     2.564     3.497     2.513    -1.284   \n",
       "max      12.875    13.583    17.405    13.180    16.059    16.052    13.840   \n",
       "\n",
       "            V22       V23       V24       V25       V26       V27       V28  \\\n",
       "count 40000.000 40000.000 40000.000 40000.000 40000.000 40000.000 40000.000   \n",
       "mean      0.943    -0.388     1.142    -0.003     1.896    -0.617    -0.888   \n",
       "std       1.646     4.052     3.913     2.025     3.421     4.392     1.925   \n",
       "min     -10.122   -16.188   -18.488    -8.228   -12.588   -14.905    -9.685   \n",
       "25%      -0.112    -3.119    -1.483    -1.373    -0.319    -3.692    -2.193   \n",
       "50%       0.963    -0.275     0.964     0.021     1.964    -0.910    -0.905   \n",
       "75%       2.018     2.438     3.563     1.400     4.163     2.201     0.377   \n",
       "max       7.410    15.080    19.769     8.223    16.836    21.595     6.907   \n",
       "\n",
       "            V29       V30       V31       V32       V33       V34       V35  \\\n",
       "count 40000.000 40000.000 40000.000 40000.000 40000.000 40000.000 40000.000   \n",
       "mean     -1.005    -0.033     0.506     0.327     0.057    -0.464     2.235   \n",
       "std       2.676     3.031     3.483     5.499     3.574     3.186     2.924   \n",
       "min     -12.579   -14.796   -19.377   -23.201   -17.454   -17.985   -15.350   \n",
       "25%      -2.799    -1.908    -1.799    -3.392    -2.238    -2.128     0.332   \n",
       "50%      -1.206     0.185     0.491     0.056    -0.050    -0.251     2.110   \n",
       "75%       0.604     2.040     2.778     3.789     2.256     1.433     4.045   \n",
       "max      11.852    13.191    17.255    24.848    16.692    14.358    16.805   \n",
       "\n",
       "            V36       V37       V38       V39       V40    Target  \n",
       "count 40000.000 40000.000 40000.000 40000.000 40000.000 40000.000  \n",
       "mean      1.530    -0.000    -0.351     0.900    -0.897     0.055  \n",
       "std       3.820     1.778     3.964     1.751     2.998     0.227  \n",
       "min     -17.479    -7.640   -17.375    -7.136   -11.930     0.000  \n",
       "25%      -0.937    -1.266    -3.017    -0.262    -2.950     0.000  \n",
       "50%       1.572    -0.133    -0.319     0.921    -0.949     0.000  \n",
       "75%       3.997     1.161     2.291     2.069     1.092     0.000  \n",
       "max      19.330     7.803    15.964     7.998    10.654     1.000  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 12;\n",
       "                var nbb_unformatted_code = \"# let's view the statistical summary of the numerical columns in the data\\ndata.describe()\";\n",
       "                var nbb_formatted_code = \"# let's view the statistical summary of the numerical columns in the data\\ndata.describe()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# let's view the statistical summary of the numerical columns in the data\n",
    "data.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The spread of attributes will be explored further (univariate analysis)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "DVCj6_DD4jan"
   },
   "source": [
    "## Univariate Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABZgAAA4ICAYAAAAkzOF9AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAEAAElEQVR4nOz9e5hdZ3kffn8faSRLtkMA29hGsjxSZRU7hVJwaPrLmxSIKZYhcdI3Sd1fgt3DG6AB2+EYsMdYAkE5U2yScggUu0kgkBYCseUADQlpm0PtFGKwHXsiywf5LIeDsWwd/Lx/zIGtmb1Ho6WZWbP3fD7XNZe01157rXsd9nM/z73XXrvUWgMAAAAAAIdrWdsBAAAAAADQnxSYAQAAAABoRIEZAAAAAIBGFJgBAAAAAGhEgRkAAAAAgEYUmAEAAAAAaESBGQAAAACARhSYoc+UUv6olPLWLtPPLaXcV0p5USnlq6WU75RSdrYQIgD0vVnk2zeUUr5ZSvleKeX2Usob2ogTAPrdLHLu60spO0op3y2l3FNK+UApZaiNWIHuFJih/3wyyctKKWXK9Jcl+Z0k30nyiSQGugDQ3Cczc74tSc5P8pQkZyd5dSnlvAWNEAAGwyczc879XJLn1FqflOQfJfnHSS5a0AiBGZVaa9sxAIehlLI6yX1JfrrW+rXxaU9Jcm+Sf1pr/cb4tLOS/FatdbitWAGgX80233bMf0XG+tYXLniwANDHDifnllKOS/J7SW6ttf5qG/EC07mCGfpMrXVPks9k7KqpCb+Y5Japg10AoJnDybfjV1z9RJJvLVyEADAYZpNzSyn/bynlu0keytgVzB9Z8ECBnhSYoT9dleQXxj/pTcYS8VUtxgMAg2i2+XZLxvrV/2WB4gKAQTNjzq21/u74LTI2JflwkvsXPkSgFwVm6EO11v+Z5MEk55ZSNiT50SS/225UADBYZpNvSymvztgg+CW11scXPkoA6H+zHePWWm/L2DeGfnNhIwRm4lc3oX9dnbEB7T9M8qVaq09wAWDu9cy3pZR/l+RNSX6y1np3S/EBwKCY7Rh3KMk/WLCogENyBTP0r6uTnJXkV9Lx1aFSyrJSyqokK8YellWllJUtxQgA/a5Xvv2lJO9I8qJa646WYgOAQdIr5/7/SilPG///GUnenOR/tBIh0FWptbYdA9BQKeVPMvYDBydNfC23lPL8JF+dMuuf1lqfv5CxAcCg6JFvb0+yNknnbTF+u9b6yoWPEAAGQ4+c+1+SnJPk2IzdRuOzSS6rtT7WVpzAwRSYAQAAAABoxC0yAAAAAABoRIEZAAAAAIBGFJgBAAAAAGhEgRkAAAAAgEaG5mIhxx9/fB0eHp6LRQHAQLjhhhseqrWeMNfLlXMB4GByLgDMv5ny7ZwUmIeHh3P99dfPxaIAYCCUUu6Yj+XKuQBwMDkXAObfTPnWLTIAAAAAAGhEgRkAAAAAgEYUmAEAAAAAaESBGQAAAACARhSYAQAAAABoRIEZAAAAAIBGFJgBAAAAAGhEgRkAAAAAgEYUmAEAAAAAaESBGQAAAACARhSYAQAAAABoRIEZAAAAAIBGFJgBAAAAAGhEgRkAAAAAgEYUmAEAAAAAaESBGQAAAACARhSYAQAAAABoZKjtAGAuXXnllRkdHV2Qde3atStJsmbNmgVZ36Fs3LgxF154YdthAEDrFrI/0Mti6CfoGwDQqY38KB/C0qDAzEAZHR3N1795cw4c/dR5X9fyR7+TJLnv8fbfRssffbjtEABg0VjI/kAvbfcT9A0AmKqN/CgfwtLQfmUM5tiBo5+aPc84Z97Xs/qWa5NkQdZ1KBOxAABjFqo/0Evb/QR9AwC6Wej8KB/C0uAezAAAAAAANKLADAAAAABAIwrMAAAAAAA0osAMAAAAAEAjCswAAAAAADSiwAwAAAAAQCMKzAAAAAAANKLADAAAAABAIwrMAAAAAAA0osAMAAAAAEAjCswAAAAAADSiwAwAAAAAQCMKzAAAAAAANKLADAAAAABAIwrMAAAAAAA0osAMAAAAAEAjCswAAAAAADSiwAwAAAAAQCMKzAAAAAAANKLADAAAAABAIwrMAAAAAAA0osAMAAAAAEAjCswAAAAAADSiwAwAAAAAQCMKzAAAAAAANKLADAAAAABAIwrMAAAAAAA0osAMAAAAAEAjCswAAAAAADSiwAwAAAAAQCMKzAAAAAAANKLADAAAAABAIwrMAAAAAAA0osAMAAAAAEAjCswAAAAAADSiwAwAAAAAQCMKzAAAAAAANKLAPM+uvPLKXHnllW2HAcwh72ugG20D0I22AWbmPQKDwXt5aRtqO4BBNzo62nYIwBzzvga60TYA3WgbYGbeIzAYvJeXNlcwAwAAAADQiAIzAAAAAACNKDADAAAAANCIAjMAAAAAAI0oMAMAAAAA0IgCMwAAAAAAjSgwAwAAAADQiAIzAAAAAACNKDADAAAAANCIAjMAAAAAAI0oMAMAAAAA0IgCMwAAAAAAjSgwAwAAAADQiAIzAAAAAACNKDADAAAAANCIAjMAAAAAAI0oMAMAAAAA0IgCMwAAAAAAjSgwAwAAAADQiAIzAAAAAACNKDADAAAAANCIAjMAAAAAAI0oMAMAAAAA0IgCMwAAAAAAjSgwAwAAAADQiAIzAAAAAACNKDADAAAAANCIAjMAAAAAAI0oMAMAAAAA0IgCMwAAAAAAjSgwAwAAAADQiAIzAAAAAACNKDADAAAAANCIAjMAAAAAAI0oMAMAAAAA0MhQ2wF0s3v37mzdujWXX355jjvuuDlfZpKMjIyklJK3ve1tOe644zI6OpqLL744H/zgB7Nx48aur3/zm9+cu+66K1deeeXkPLt3787IyEi+//3v584778zTnva0PPDAA3nd616XL37xi7n11luzevXq7N69e862BWjXN77xjSTJ85///AVb59DQUA4cOJBaa5LkZS97Wf7X//pf2bFjx0HzPe1pT8t3v/vdPO1pT8t9992Xffv2pdaatWvXZsWKFbnnnntSSsnTn/70rFq1Kj//8z+fbdu25bLLLsvv/M7vZNeuXVm7dm1+/dd/Pe94xzuyY8eOrFq1Ktu2bcvVV1+dn/3Zn52c/zOf+UwOHDiQ5cuX53Wve12uuOKKyXZ7ajs+tQ2eTRt/qHZ5rsxHzuknTfdz0/02Ojqaiy66aPK8uOKKKw7KqVu3bs1FF100eT4lYzn78ccfzz333JOTTz45Q0NDeeyxx7Jr1678yq/8Sj7ykY9k1apV+dVf/dV84AMfyCmnnJJjjz02r33ta3PJJZfkgQceyPnnn5+vfe1rueOOOybfR52WLVuWJ554omvMxxxzTL7//e/Pehs//elP57zzzpv1/MBg27VrVx566KEj6jeUUibbrnXr1uUtb3lL3vWud+Wuu+7K0572tNx///3Zu3dvjj/++Dz44IM58cQT893vfjcnn3zyZPu2a9euyeWdeOKJeeCBB3LyySenlJJ77703a9euzTvf+c4kyWte85rceeedWbduXT7wgQ9Ma+e75Y7Z5PqlnnNnu/3d5jvU/p2aQy+66KK8//3vz969e7Ns2bLs378/99xzT9atW5f/+B//4+QY+NWvfvXkubVmzZokY+fsRH/xiSeeyL333ptk7Dx8+9vfnt/4jd/Ijh07ctJJJ+W+++47KPbly5fnwIEDk+Piidd1y71TjY6OzmufD5hft912Wx599NEFHSf38rznPS9/9Vd/Nfl4aGgo+/fv7zrvRLt11FFH5elPf3pWrFiRffv2ZdeuXZPj6TVr1uTv//7vc8UVV+Tb3/523vCGN2RoaChDQ0PZtm1bPv7xj2fv3r1ZuXLlZJ1x9+7dueyyy7J///4sX74827Zt6zpWTmaXQ7uZKa8sdM5dlFcwX3XVVbnxxhtz9dVXz8syr7rqqtx888256aabJtexbdu2fP/738+2bdt6vv7WW2/Nnj17DppnYll33nlnkkwm0fe///259dZbkyR79uyZ020Blp79+/cf1DH/r//1v04rLidjbdBjjz2WO++8M3v37p18zd13353bb789jz/+eB577LHs2LEjN910U97xjnfkiSeeyNvf/vaMjo5mz549ue2227Jt27bJ5T/22GPZsmVLbrzxxoPmv/nmm3Prrbfm5ptvzrZt2w5qt6e241Pb4Nm08Ydql+fKfOScftJ0Pzfdb9u2bcujjz6au+66K48++ui0nHrjjTcedD5N5NkdO3bksccey+23357bbrstd911V5544ol85CMfSTJ2nn7gAx9IrTV33nlnbrrppmzbtm0yL1999dXZuXNnzwFur+JyksMqLifJhz/84cOaHxhsDz300BEvo7PtuvPOO7Nt27bcdtttkzn/8ccfT601Dz74YJLk/vvvz549e7Jjx46Mjo5mx44defzxxyf/7rzzzsk2dceOHZP5f6LdnRjb3HnnnV3b+W65Yza5fqnn3Nluf7f5DrV/p+bQbdu25aabbsro6GhuvfXWyTx66623HjQGfuyxxybPix07dkyeKxP9xZ07d04+/9hjj+Xyyy+f7CNOLS4nyYEDB5L8YFycZFbF5Yl4gP716KOPth3CpM7icpKexeXkB+3W448/nttvvz233nprbr/99oPG07t27Zocu2zZsiW11uzbty979uzJ5ZdfPtnedtYZr7rqqtx0002TY+ZeY+Wp0w4nV84070Ln3EVXYN69e3euu+661Fpz3XXXZffu3XO6zO3bt+faa6+dfG779u25/vrrs3PnziTJzp07Mzo6Ou3127dvn3w8Mc/U6Z2mJtFrr712TrYFaNe5557bdghzaiLRTk24E23ihEceeSS11hnnn2i3R0dHD2rHOx9v374927dvP2QbPzo6OmO7PFfmI+f0k6b7uel+61zfhM6cOrHMifNpas4+lKm5d+q6FtKnP/3p1tYNLB4f+9jH5mW589W+XXPNNbnmmmsOmvaHf/iHB7Xz3XLH1PFWtxyx1HPubLe/23xT9+/UvlS3HDrTOXLttdceNAY+HI888shhv2a25rPPB8yv17/+9W2HsCB27tw5rR2c+nj79u2TY+Be0w+nje9lprzSRs5ddLfIuOqqqyavIjpw4ECuvvrqvOY1r5mzZU5c3j5h37592bJly0Hzb9u2LZ/85CcPev2+ffumzfOsZz1rxk9BOu3bty8vf/nLs3bt2mYbwayMjo5m2d7ZfUI+SJY99t2Mjn4vF198cduhDLzvfOc7bYewqB04cCDbtm07qB3vfNzZls7Uxk+9gmVquzxX5iPn9JOm+7npfut1ZdJETp16FfHUnN1PPvzhD+fP//zP2w5jyVqq/YFO+gaLw8RttfpFt7HN1Ha+W+7obMN75fqlnnNnu/3d5qu1zrh/O5+fjW5j4MXiVa96VZ7xjGe0HcbAWor5UT5cGP2W7+bTvn37sm3btml1xInph9vG98qVM+WVNnJu4yuYSykvL6VcX0q5fuKrWHPhK1/5ykFXyH35y1+e02VOHajWWqd92jD1k9yvfOUr05a5c+fOfOUrXzmsge/f//3fz3pegH60f//+7Ny586B2vPNxrXWy3Zypje92let8mI+cMx/mK+c23c9N91uv5U/k1KmFjX4tLgMMms52vlvumDre6pbrl3rOne32d5vvUPu3Ww49lPm8EvlIPP74422HAHBEOr9N0m364bbxvcyUV9rIuY2vYK61fjTJR5PkzDPPnLMR4FlnnZVrr702+/fvz9DQUF70ohfN6TKn/sBAKSXHHHPMQQl2eHh42uu/8IUvHDRteHg4z3rWs/LFL35x1gPgn/7pn15Sn9K34eKLL84NO+5vO4wF98SqJ2XjhhPzwQ9+sO1QBt5i+MGCxWxoaChr167N3XffPdmOdz4upSQZS5wztfHDw8MHDWCntstzZT5yznyYr5zbdD833W9T19c5/VnPetbkMifM9keBFittcnuWan+gk77B4jAo/YbOdr5b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vfcmSbZt25YPfOAD2bVrV1auXJlf/MVfzG//9m+nlJJa6+R6ly9fnqc+9al58MEHc+yxx+aRRx6Z/Q6fwVe/+tW84AUvmJNlAYvbww8/nLvuuivPf/7z53zZy5YtyxNPPNHz+aGhoSxbtix79+5NkqxYsSL79u2bfP6EE07I9773vRw4cCD79u3LihUrJpe5f//+rF+/Pr/6q7+ayy67LLXWfOhDH8rGjRszOjqaCy+8MCeccEIeeOCBnHzyyZPLvv/++/O2t70tv/Vbv5VSSt72trfNOucMktHR0Vx88cX54Ac/mI0bNx7x8nbv3p3LLrssjz76aB544IFcccUVBy134vlaa7Zt2za5z0dHR3PRRRdN5sWhoaEMDQ3ljW98Y9797nfn5JNPzrJly7Jy5cq87W1vSzKWw88///xcfvnlef3rX5/3vOc9OfHEE7NixYo8+uij2bVrV57+9KfnoYceyhNPPHHQObVq1ao89thjk4/XrFlzxNsODIZHH300t91225zlwxNOOCEPPfRQfumXfim//du/3fW5tWvXZsWKFbnnnnty4okn5oEHHsiVV16ZJAflsXXr1uWNb3xj3ve+96WUkte+9rW54oorJsc2IyMjk9Pf//73T7a1SXLZZZdl//79Wb58eV73utcd9PxC5b8m47xB0vb2H+5YeKKP8LrXvS7ve9/78vrXvz7vete7kiQf+tCHkiQXX3xx3vrWt+YjH/lI7rjjjuzduzennHJKjjnmmLzuda/Lu971rtx111158pOfnPvuuy8rVqxIrTX79+/PU5/61Dz88MNJpvf9zjrrrPzZn/1ZHn/88SSZHMvP1qc//emcd955h7+TgIHz3e9+N7fffnujvL5y5cosW7YsP/RDP5QHH3wwK1asyCmnnJIkuffee3PyySen1pr7778/r3/96/Pe9743a9asyTvf+c4kyZvf/ObcddddufLKKyfHp6961auyd+/evPe9783w8HAuu+yy7N27N8uWLcvy5csn8/ZsapzdTMxz0UUXTfYRFiLnlM4iZVNnnnlmvf766+cgnDHvf//788UvfjE/8zM/k9e85jVH9Jqp0zsf11pntZ5uy55pfV/4whemLWNq0XdoaCj79+8/5HbNZbG4iaGhoXzlK19pbf1L1cUXX5wbdtyfPc84Z8HXvfqWa5OklXVPrP+5G07MBz/4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AGhEgRkAAAAAgEYUmAEAAAAAaESBGQAAAACARhSYAQAAAABoRIEZAAAAAIBGFJgBAAAAAGhEgRkAAAAAgEYUmAEAAAAAaGSo7QC62b17d7Zu3ZrLL788xx133LTpF110Ua644oqDnu/1ms7XXnbZZam1Ztu2bTnuuOO6ThsdHc2rX/3qJMkpp5ySd77znUmSkZGRHDhwIPv378+9996bN7zhDXnPe96Tk08+ObXW7Nq1K3v37s2KFSuyfPnynHjiibnnnnuyb9++edlHb3nLW/LWt751XpYNzN4dd9yRb3/723n+85/fahxPecpT8vd///dJkmXLluWJJ56YfG7q45NOOim7d++ebJ9OPfXUXHbZZXn729+e22+/PUmyZs2abN26Ne94xzuyY8eOJEkpJbXWrFq1Kh/60Ify7W9/O294wxuydu3a/NAP/VBe+9rX5n3ve1/PtnJoaCzl7Nu3L/fff39e+cpX5j/9p/+U97znPXnuc5/bc9umtu/d2u5e8yeZMTd0W/5SM9/bP5v8PPX5Xq8ZHR3NxRdfnLe+9a256qqrctFFF+Ud73hHbr/99rz3ve+dPI8mcvkTTzyRffv2pdaaZOzcHxoayq5duyandTruuOOybNmyPPjgg5PTli9fngMHDszpPunlN37jN/KqV71qQdYF9I+77rqraz9j3bp1ueiii3L55Zfngx/8YO6888689a1vTSklQ0ND2bdvX1auXJlkLPeuX78+73nPe5KMjW3279+fxx9/PHfddddkLn/b297WdfzVmYNn22Z3W0Zy6Lw8yA7Vh+n1ms4x8NSx8Ez7f/fu3Xnzm9+cu+66K2984xvz7ne/OyeccEIeeuihXHHFFXnKU56SN73pTbn77ruzdu3avOIVr8jll1+et771rfn4xz+e733ve7n77rtz/PHHH5QbV6xYMdmPPOqoo/KUpzwl991337TnJ/qOSXLsscfmkUcemZxndHQ0GzduPKL9CdDL7bff3niM/sM//MN55JFHcuDAgTzpSU/Kd7/73a7zlVLy8pe/PB/96Efz9Kc/PQ8//HB++Zd/OR/72MdSSsnw8HAuvfTSvP/97z+o3b/++uvzxje+MaeeemouvfTSXHHFFTn//PMn8/nGjRuze/fuyTrk3r1788ADD0y225dddln27t2blStXTsvbnRZ6nLsor2C+6qqrcuONN+bqq6/uOn3btm3Tnu/1ms7nb7rpptx8882T83Sbtm3btjz22GN57LHHctttt+Xqq6/OVVddlZtvvjm33nprduzYkT179uQd73hH9uzZkx07duT222/P3r17k4x13h577LHccccd81ZcTpKvfe1r87ZsYPa+/e1vtx1CkkwWl5McVEzu9vi+++47qH264447sm3btsnicpLs2rUr27ZtmywuJ5kcIDz22GPZtm1btmzZklpr7rrrrtx0003Ztm3bjG3lrbfemltvvTW33357Hn300XzgAx/IE088MTng7GVq+96t7e41/6FyQ7flLzXzvf2zyc+zzenbtm3L97///Vx++eWT/YEdO3ak1nrQeTSRy/fu3XtQIfm+++7L3Xff3bW4nIx1wjoH0EkWrLicJJ/97GcXbF1A/3j44Ye7Tr/zzjuzZcuWfP/738+2bdvyjne8I8lYvp7I83v37p1sC3fs2HHQ2Oa2227LnXfeeVAu7zX+6szBTcZhh5OXB9mh+jC9XtM5Bp46Fp5pn1511VW59dZbs2fPnrz97W/Pnj17cuedd+bRRx/Ntm3bctVVV+W2227Lnj17ctttt02eT5dffnluuumm3HXXXam1TsuNnf3Ixx9//KDicufznfm2s7icjOVqgPnSqyg8G9/5zncmxwAzLafWmo985COTF53u2bMnH/vYxyafu/3227Nt27Zp7f6WLVvyxBNPTD5/4403HpTPkxxUh9y5c+dB7fZNN92U0dHRrnm700Ln3EVXYN69e3euu+661Fpz3XXXZffu3dOm79y586Dne71m6jInbN++PaOjo9Om3XDDDdm5c+dBr73mmmuyffv2aXHu379/Dre6mbe85S1thwBL2rve9a62Q5gzU9u+XtM6n5s6UOg2/0xt5cSg45FHHskNN9zQdZ6p7Xu3truzze+cf/v27TPmhm7L7zbPIJvv7Z9tfp5NTh8dHZ08xx555JHJ/sCEifOoc75+9Bu/8RtthwAsInfdddeMz0/k4p07d85qfHLNNdfk2muv7fl8Z17tlYObjMMm8vL27duXfM6dMLUPM9NrOsfAnWPhbsek87Wd49ip58fOnTtzzTXXHDRt4nya2sebDzt37szo6Oi8rwdYejovnGpb57hk+/bt+eM//uOD2tiJNr0zn99www1d65A7d+6clsN75ZI2xrmL7hYZV111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AAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhEwAwAAAADQiIAZAAAAAIBGBMwAAAAAADQiYAYAAAAAoBEBMwAAAAAAjQiYAQAAAABoRMAMAAAAAEAjAmYAAAAAABoRMAMAAAAA0IiAGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmAAAAAAAaETADAAAAANCIgBkAAAAAgEYEzAAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhEwAwAAAADQiIAZAAAAAIBGBMwAAAAAADQiYAYAAAAAoJF1g25AJ/v27cu1116bq6++Ops3b+7bOtqnJVlQG1rPueiii3L11VfnpptuytjY2CHTd+7cmST5jd/4jSTJpZdemuc85zkppeThhx/OzTffnLGxsRnL/bmf+7nce++9efrpp/P2t78973nPe3LfffflmWeeSZKsWbMmxxxzTB577LEkyc/8zM/kxhtvTCkl69aty/79+6eWt27duhw4cCCllCRJrfWQ9W3atClPPfVU4/05l0suuSS7du3qybKBwfnGN76RL3/5y3nBC17Q8fGRkZEce+yxeeCBB5Ikz372s/O1r33tkHm2bNmSr3/963nzm9+cd7zjHfme7/mePPTQQ3nta1+b2267LUcffXTuv//+bNiwIdu2bcsv/dIvJZnony+99NLcfPPN+dEf/dHccMMN2bp1a4488shcf/31SZK3ve1tqbXmhhtuyObNm2ft49t/7lXNWc5a+6W1Pxe6H2arozt37kwpJa95zWum6t9xxx2XRx55JLt27cpRRx2Vt771rbn33ntzwgkn5JlnnsnDDz+co446Kg888EBGRkayf//+rFmzZqreJUkpJRs2bMhFF12Ud73rXRkZGcmaNWtSa83TTz/dk32zGl8PsBJ84QtfyFNPPTVr/emFF73oRfnTP/3TQ6Zt3rw5jz/+eE466aR85zvfycMPPzw1rXUuvm3btmzcuDHf/va3c++99+akk07K4YcfniuuuCK/9mu/llJK3vSmN+Xmm2+eMZ5YiKWMm/ox5loO61xOFrL9i9lHi92fC51/3759h5xHJZmq79dff32++tWv5rLLLsvP/uzP5ld+5Vdy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HMj69euzdevWlFLywAMP5IQTTkiSqf7rqKOOytve9rYcOHAg+/fvz8MPP5ybb745Y2NjSSb6z507d6aUkte85jW5+uqrc9NNN+Woo45acB/eWs5i5l+sXi9/uVvq9u/bty8///M/n3vvvXeqJs62/CQzanLrNXL99ddn8+bN2bNnT97ylrfkp3/6p/Oud70r69evz+tf//q8853vzNatW/Poo49mzZo1efWrX513vetdOfbYY/Poo48mSUZGRrJ///5s27YtTz/9dB566KFD2tJeM6ebqy6Oj4/P2C6Axbr33nsXXdcPP/zwfOtb3zpkWikll19+ef7Tf/pPee1rX5tf//Vfz0/91E/ld37nd3LcccflkUceydve9ra8733vmxojXHHFFbn55ps7joGmj32uvfbaXHTRRbn66qvzsz/7s/nVX/3V3HTTTVNjnk7P2blz59S6brjhhlnriZq7sO3vNGZtH8vu2LEjb3/726eOy3zL+OVf/uXce++92bZtWy6++OK87W1vy9FHH537778/SfLsZz87X/va17Jly5Z87WtfS601xx13XB5++OEcPHhwxnljkqxZsybPPPNMd3ZMDxmLQv997nOfy4EDB3oyll2KheZgxx57bL71rW/luuuuy2/91m/lS1/6UkZGRnLgwIEcd9xxeeihh7Jx48b8zM/8TN75znem1pqtW7fmWc961ox62ylnbJmrJrT34TfeeGNqrVPLnq2GT9fvmlu6ETKeffbZdc+ePV1ozoQbb7wxH/zgB/Pyl788l19++aIfn2ve1u8nn3xy7rnnnrz85S9PrXXGtNZyb7zxxnzgAx84ZJmjo6PZu3fv1M+33HLLjPkuvPDCqeW+/OUvz8c//vE88cQTM5bz2GOPzZg+jD75yU8OugkrxmWXXZY7v/Rwnnru+T1f16a7dydJT9e16e7def7243LTTTf1bB3D6KUvfWm+853vDLoZsxodHc1ZZ53VsX+85ZZbkhzafx5xxBF54oknpp630D68tZzFzL9YvVp+KeXOWuvZXVvgpH7X3IU8v3Wc249/p+W318VWTW4998ILL8zll1+eCy64YEZd7OeHop102i4Gr5/1ci79qKULpeYuX5dddlnuuuuuri2v1S/O1j9O/zB5dHQ099xzT8cx0PSxzwc/+MEcfvjheeKJJ6aW0z7m6fSc6eOg2eqJmruw7e80Zm0fy7aOz1z1qX0ZrbFr8t1zstXEWHTp+l1z+11b1c/uWm7BclPz9ZedavD0etspZ2yZqyZ06sNby56thk/Xi5o7V71ddrfI2LdvX+64447UWnPHHXdk3759i3p8rnnHx8enft+7d29qrbn99ttz++23HzKttdzW86drL9B79+7NnXfeOWO+3bt3T63rwx/+cMcX5d69e1dNcX/7298+6CbAijE+Pr6sw+Vkov/avXt3x+nj4+PZt29fbr/99qnprb6u9byF9OHJ4vr8Jnq9/OVuqds//Ti3jn+n5bfX21ZNbn/u7bffnj/90z/tWBcHGS4nM7cLYLHuvfferi6v1S/O1j9Ov+K0Nc5pHwNN7//b++xWX9xaTvuYZ/pz2vvyZKI/71RP1NyFbX/7fO3j0/axbPt5Vaf6NH0Z7VbL+LOdsSj0z4//+I8PugldM19/2akGT6+303PGlrlqwmx9eKccc7Z6Moiau+xukXHrrbdOfdXm4MGDue222w5J2ud7fK5l3XDDDTO+xrN///4Zz2stt9ba8fHprr766hnz7d+/P6WUJDNP8FajT33qU7nssssG3YwVYXx8PGueHmyY0k1rvv2NjI9/0/FfhLvvvnvQTViQ2fq2G264IWedddasj7emz9eHJ4vr85vo9fKXu6Vu/6233jqj/t1www1TV1O1L799vlZNbn+N7N+/P//hP/yHppvSc69//evz3Oc+d9DNoM2w1ctuUHOXr8cff3zQTTjE9D65NfaZ65YHV1999YyaUWudUe/379/fsZ6ouQvb/vb5WmYbyyaH1t25lrGaGYsu3bDXXPWzex588MFBN2FZmt7vz1UT5uvDO9Xw5VBzG1/BXEq5uJSyp5Syp3Xvw2742Mc+NnWScuDAgXz0ox9d1ONzzbt3794ZJ0C11hmfOrSW+7GPfWxBV0098cQTHecb9BVXwMq03K9ens/evXsX1H/O14cni+vzm+j18rtlUDV3Ic+frv1T9vblt9fbVk1uf410CimWk5X+vgRoN71Pbo195uqHn3jiiRk1o1O9r7V2rCdq7sK2v9NxmG0sm2TGFcqzLQOAwZne789VE+brwzvV8OkGUXMbX8Fca31XknclE/em6laDzjnnnOzevTsHDhzIunXr8pKXvGRRj88174knnpj77rvvkAPVusq4/cSotdzWPU3mC0mOOOKIfOtb35ox36DvGbncuJ/RwrTubzUsntn4rIy5n9Wi/MRP/ETHwcJK0X6f5bn6wPn68GRxfX4TvV5+twyq5i7k+Z3uw91p+e31tlWT77nnnqnXSPsfmFyORkdH9WPLzLDVy25Qc5ev5XY/yul9cmvs0+qzOzniiCPy7W9/+5Ca0Wm8VErpWE/U3IVtf/t8LbONZZND6+5cy1jt9ItLM+w1V/3snuVW75aL6f3+XDVhvj68Uw2fbhA1d9ndg3nHjh1Zs2aiWWvXrs1FF120qMfnmnfnzp1Tv7eMjIxk3bpDc/bWcnfs2JGRkZF523zttdfOmG9kZGRq2vTlr0Y//MM/POgmwIqxc+fOQTdhQWbr23bu3JkdO3bM+nhr+nx9eLK4Pr+JXi9/uVvq9neqk+2v3/blt9fbVk1uf42MjIzkyiuvbLQd/bBS3pfA8nT00UcPugmHaB+rtI99po+V2l177bUzakanej8yMtKxnqi5C9v+TsdhtrFs0rk+zXcsVxtjUeif448/ftBNWJam9/tz1YT5+vDp46rlUnOXXdXZvHlzzj333JRScu6552bz5s2LenyuecfGxqZ+Hx0dTSkl5513Xs4777xDprWW23r+dO2fEo+Ojub5z3/+jPnOP//8qXW97GUvyxFHHNFxOZ2mD6Prrrtu0E2AFWNsbCwbNmwYdDPmNDo6mvPPn/lXnUdHRzM2NpbNmzfnvPPOm5re6utaz1tIH54srs9votfLX+6Wuv3Tj3Pr+Hdafnu9bdXk9ueed955edGLXtSxLrY+pR+U6dsFsFgnnXRSV5fX6hdn6x+nh76tcU77GGh6/9/eZ7f64tZy2sc805/T3pcnE/15p3qi5i5s+9vnax+fto9l28+rOtWn6ctot1rGn+2MRaF/3vve9w66CV0zX3/ZqQZPr7fTc8aWuWrCbH14pxxztnoyiJq77ALmZCJpP/PMM+f8VHeux+eat/X7zp07p6Z3mtb+/DPOOCNjY2M57bTTcvrpp2fnzp059dRTs2nTpqlPjFvznXrqqTnjjDMOWe5FF12Ua665JqWUbNu2Ldu3b596bmt6MnHiecopp2TTpk3Ztm1bNm7cmFNOOSXbtm1LKSVbt27Nhg0bsnbt2qn2tb5SnOSQ6fPp52DdJ8aweNu2bevZsjtdWdyaVkqZ+nn9+vU55ZRTsn379mzcuDHbt28/pP9q9XunnXZaTjnllBx22GEzrl49/fTTc8YZZ+Saa67J4YcfPvW8hfbhreUsZv7F6vXyl7ulbv+OHTty2mmnHVITZ1t+p5rceo20pl1zzTVZs2ZNXvva16aUkg0bNuTyyy9PKSUnnnhiNmzYkE2bNuXiiy9Okhx77LFT62pdjbdt27Zs2bJlRlvaa2anx2bj6mWgG5pcxXz44YfPmFZKyeWXX57DDz88l19+edasWZOLL744pZRs2bIla9asyVVXXZXTTz/9kPHLbGOg6WOfM888c6puX3XVVVP1u/3x6c9pX9d83zBVc+ff/tnGp63p11577SHHZb5lnHrqqdm4cWNOO+20XHPNNdm0aVO2bt06Ne+zn/3sJMmWLVuycePGbNiwIdu2bcuGDRtm/UbaSrlC2lgU+m+5fot/oTnYsccem8MOOyzXXHNNtm/fnmRinNGqs0mycePGqTFKkmzdurVjve2UM7bMVRPa+/AzzjjjkGXPVsMXs/xeKN24R/DZZ59d9+zZ04XmMExaf4HVfYwWp3V/q6eeO/Pq0G7bdPfuJOnpujbdvTvPdz+rRfP+WflKKXfWWs/u9nLV3P7xPlze+lkv59KPWrpQau7ypT/pLTWX6bznuqvfNbfftVX97C7vv+E1V71dGR87AgAAAACw7AiYAQAAAABoRMAMAAAAAEAjAmYAAAAAABoRMAMAAAAA0IiAGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmAAAAAAAaETADAAAAANCIgBkAAAAAgEYEzAAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhEwAwAAAADQiIAZAAAAAIBGBMwAAAAAADQiYAYAAAAAoBEBMwAAAAAAjQiYAQAAAABoRMAMAAAAAEAjAmYAAAAAABoRMAMAAAAA0IiAGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI0ImAEAAAAAaGTdoBvA8BobGxt0E2DF8v6BwfM+BLpFfwL95T0Hg+P9tzoJmOmZSy65ZNBNgBXL+wcGz/sQ6Bb9CfSX9xwMjvff6uQWGQAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhEwAwAAAADQiIAZAAAAAIBGBMwAAAAAADQiYAYAAAAAoBEBMwAAAAAAjQiYAQAAAABoRMAMAAAAAEAjAmYAAAAAABoRMAMAAAAA0IiAGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmAAAAAAAaETADAAAAANCIgBkAAAAAgEYEzAAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhEwAwAAAADQiIAZAAAAAIBGBMwAAAAAADQiYAYAAAAAoBEBMwAAAAAAjQiYAQAAAABoRMAMAAAAAEAj6wbdAGCmtU8+nk137+7DevYlSU/XtfbJx5Mc17PlA7B69atezt2G3tfShVJzAeiVftbcftdW9ROWTsAMy8zY2Fjf1nX//QeSJFu39rKYHtfXbQJgdVgutaU/tXSh1FwAuq/ftaX/tVX9hKUSMMMyc8kllwy6CQCw7KmXANAfai4wH/dgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhEwAwAAAADQiIAZAAAAAIBGBMwAAAAAADQiYAYAAAAAoBEBMwAAAAAAjQiYAQAAAABoRMAMAAAAAEAjAmYAAAAAABoRMAMAAAAA0IiAGQAAAACARgTMAAAAAAA0ImAGAAAAAKCRUmtd+kJKeTTJPW2Tjkny2JIXPJzsm9nZN53ZL7OzbzqzX2bXz31zcq312G4vtEPN7YfV8JqyjcPBNg4H2zgc1NzlZTW85hbLPjmU/TGTfTKTfXKo1bY/Zq23XQmYZyy0lD211rO7vuAhYN/Mzr7pzH6ZnX3Tmf0yO/ummdWw32zjcLCNw8E2DofVsI0rieMxk31yKPtjJvtkJvvkUPbHd7lFBgAAAAAAjQiYAQAAAABopFcB87t6tNxhYN/Mzr7pzH6ZnX3Tmf0yO/ummdWw32zjcLCNw8E2DofVsI0rieMxk31yKPtjJvtkJvvkUPbHpJ7cgxkAAAAAgOHnFhkAAAAAADQiYAYAAAAAoJGuBsyllFeUUj5XSnmmlHJ22/TRUspTpZRPT/77j91c73I3236ZfOznSynjpZS/K6W8dFBtXA5KKdeUUu5ve52cP+g2DVop5dzJ18Z4KeWtg27PclFK2VtK+czk62TPoNszSKWUd5dSHimlfLZt2tGllI+WUr44+f9Rg2zjoMyyb/QzDZVSnldK+YvW+66U8v2DblMvlFIumex3P1dK+ZVBt6dXSik/W0qppZRjBt2WbiulvKOUcncp5W9LKX9cSnn2oNvULcN+XlBKOamU8olSyucn34OXDbpNvVBKWVtK+ZtSyocG3ZZeKKU8u5TyB5Pvw8+XUv75oNu0Whmfz2RsPjfnyhOGvd42YQxu7D2fbl/B/Nkk/yrJpzo89ve11udN/ntdl9e73HXcL6WUM5K8Msn3Jjk3yW+VUtb2v3nLyjvbXie7B92YQZp8LfxmkvOSnJHkxydfM0x44eTr5Oz5Zx1qt2Si/2j31iQfr7WemuTjk7+vRrdk5r5J9DNN/UqSa2utz0vy9snfh0op5YVJLkxyVq31e5P86oCb1BOllJOSvCTJVwbdlh75aJJ/WGs9K8kXkvz8gNvTFavkvOBAkitqracn+WdJXj+E25gklyX5/KAb0UM3Jbmj1vrcJP8ow72ty53x+UzG5vNb1efKq6TeNrXax+C3xNh7Vl0NmGutn6+1/l03lzkM5tgvFyb5vVrrd2qtX04ynmQorwijke9PMl5r/VKt9ekkv5eJ1wxMqbV+Ksnj0yZfmOTWyZ9vTfKj/WzTcjHLvqG5muRZkz9/T5IHBtiWXvn3SX6p1vqdJKm1PjLg9vTKO5O8JRPHdOjUWj9Saz0w+etfJDlxkO3poqE/L6i1Plhr/evJn7+ZiWBy62Bb1V2llBOTvCzJ7w66Lb1QSnlWkh9O8p+TpNb6dK31awNt1CpmfD6TsTkLMPT1lmaMvefWz3swnzL5VbD/VUr5oT6udznbmuTett/vy5CdRDfwhsmvtL57NX+1YJLXx+xqko+UUu4spVw86MYsQ8fVWh9MJgbrSZ4z4PYsN/qZZt6Y5B2llHszcWXvUFwVOs1pSX6olPKXk+cr3zfoBnVbKeXlSe6vtd416Lb0yWuS3D7oRnTJqjovKKWMJvnHSf5ywE3ptl/PxAc8zwy4Hb2yPcmjSf7L5Njvd0sphw+6UXRkfH6oVdXHzmO1nyt7LXRmDN6ZsfekdYt9QinlY0m2dHjoqlrr/5zlaQ8m2VZr3VdKeX6S95dSvrfW+o3Frn+5arhfSodpQ3k1Uctc+ynJbye5PhP74Pokv5aJgeFqtepeH4vwA7XWB0opz0ny0VLK3ZOfJsJ89DNzmKePfnGSy2utf1hK+TeZuDrtnH62rxvm2cZ1SY7KxFfzvy/J+0op22utK6rvnWcbr0zyL/vbou5byHlXKeWqTNxy4T39bFsPrZrzglLKEUn+MMkbh2y8cEGSR2qtd5ZSXjDg5vTKuiT/JMkltda/LKXclImvC79tsM0aXsbnMxmbz82YfF6r5rWwSMbgzGnRAXOtddGDycmvmra+bnpnKeXvM3GV0NDcGLzJfsnEJ2Entf1+YobzK8dTFrqfSim/k2Qo//DJIqy618dC1VofmPz/kVLKH2fia0yK23c9XEo5vtb6YCnl+CTD+jX/Rau1Ptz6WT8z01x9dCnltkzcNzRJ/kdW6Ne759nGf5/kjyYD5b8qpTyT5JhMXI23Ysy2jaWUM5OckuSuUkoyUVf+upTy/bXWh/rYxCWb73yilLIjyQVJXrzSPiCYw6o4LyiljGQiXH5PrfWPBt2eLvuBJC+f/KNZG5M8q5Ty32qtrxpwu7rpviT31VpbV57/QVbx/Sj7wfh8JmPzuRmTz2vVvBYWwxh8Vsbek/pyi4xSyrGtG+SXUrYnOTXJl/qx7mXuA0leWUrZUEo5JRP75a8G3KaBmXwztvxYJv4Aw2r2f5KcWko5pZSyPhN/dOIDA27TwJVSDi+lHNn6ORNX4q3218p0H0iyY/LnHUlmu1Jj1dHPLMkDSf7F5M8vSvLFAbalV96fiW1LKeW0JOuTPDbIBnVTrfUztdbn1FpHa62jmRhA/ZOVFi7Pp5RybpKfS/LyWuuTg25PFw39eUGZ+OTjPyf5fK31xkG3p9tqrT9faz1x8v33yiR/OmThcib7k3tLKf9gctKLk/zfATaJDozPOzI2j3PlSUNfbxfLGHxOxt6TFn0F81xKKT+WZFeSY5N8uJTy6VrrSzPxhx6uK6UcSHIwyetqravmDy/Ntl9qrZ8rpbwvEyddB5K8vtZ6cJBtHbBfKaU8LxNfP9mb5LUDbc2A1VoPlFLekORPkqxN8u5a6+cG3Kzl4Lgkfzx59d26JP+91nrHYJs0OKWU9yZ5QZJjSin3Jbk6yS9l4qv9P5nkK0leMbgWDs4s++YF+pnGfjrJTaWUdUm+nWQY77327iTvLqV8NsnTSXYM0dWvq8lvJNmQia9vJslf1FpfN9gmLd0qOS/4gSSvTvKZUsqnJ6ddWWvdPbgm0cAlSd4zGcx8Kcm/G3B7Vi3j85mMzee16sfkq6TeLpYxeIy951OMmwAAAAAAaKIvt8gAAAAAAGD4CJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhEwAwAAAADQiIAZVohSyidLKS+dNu2NpZTfKqXcUUr5WinlQ9Me/8+llLtKKX9bSvmDUsoR/W01AKw8TWpu23y7SilP9KelALByNRzj3lJK+XIp5dOT/57X10YDHQmYYeV4b5JXTpv2ysnp70jy6g7PubzW+o9qrWcl+UqSN/S2iQAwFJrU3JRSzk7y7J62DACGR6N6m+TNtdbnTf77dA/bByyQgBlWjj9IckEpZUOSlFJGk5yQ5M9rrR9P8s3pT6i1fmNy3pJkU5Lat9YCwMq16JpbSlmbicHwW/rYTgBYyRZdb4HlScAMK0StdV+Sv0py7uSkVyb5/VrrnKFxKeW/JHkoyXOT7OppIwFgCDSsuW9I8oFa64O9bh8ADIOmY9wkvzB5G8h3tsJpYLAEzLCytH+FqPXVoTnVWv9dJj4F/nySf9u7pgHAUFlwzS2lnJDkFfFBLgAs1mLHuD+fiYunvi/J0Ul+rndNAxZKwAwry/uTvLiU8k+SbKq1/vVCnlRrPZjk95P8v3vYNgAYJu/PwmvuP04ylmS8lLI3yWGllPHeNxEAVrz3ZxFj3Frrg3XCd5L8lyTf34c2AvMQMMMKUmt9Isknk7w783yyWyaMtX5O8iNJ7u51GwFgGCym5tZaP1xr3VJrHa21jiZ5stY61vtWAsDKtph6mySllOMn/y9JfjTJZ3vYPGCByvy3tgGWk1LKjyX5oySn11rvnpz2Z5n4mtARSfYl+ckkH03yZ0melaQkuSvJv2/94T8AYG4Lrbm11j+Z9rwnaq1H9Lu9ALASLabellL+NMmxmRjjfjrJ6yZDamCABMwAAAAAADTiFhkAAAAAADQiYAYAAAAAoBEBMwAAAAAAjQiYAQAAAABoRMAMAAAAAEAjAmYAAAAAABoRMAMAAAAA0IiAGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmWOZKKX9SSrmuw/QLSykPlVJeUkr5RCnl66WUvbMs47JSypdLKd8qpXy+lHJazxsOACvMUmpuKWVbKeWJaf9qKeWKvm0AAKwQSx3nllKeV0r5s8nH7yulvL0vDQc6EjDD8ndLkleXUsq06a9O8p4kX0/y7iRv7vTkUspPJfnJJC9LckSSC5I81qvGAsAKdksa1txa61dqrUe0/iU5M8kzSf6wt00GgBXplixhnJvkvyf5VJKjk/yLJP++lPLy3jQVmI+AGZa/92eiaP5Qa0Ip5ahMBMW31Vr/qtb6X5N8afoTSylrklyd5PJa6/+tE/6+1vp4f5oOACvK+9Ow5nZwUZJP1Vr39qCdALDSvT9Lq7mjSd5Taz1Ya/37JH+e5Ht72mJgVgJmWOZqrU8leV8mBqot/ybJ3bXWu+Z5+omT//5hKeXeydtkXDsZPAMAbZZYc6e7KMmt3WobAAyTLtTcX09yUSllpJTyD5L88yQf63pDgQURMsHKcGuSV5RSNk3+vtBB64mT///LTHxV94VJfjwTt8wAAGZqWnOnlFJ+KMlxSf6gy20DgGGylJr7oST/OslTSe5O8p9rrf+n+00EFkLADCtArfXPkzya5MJSyvYk35eJe07N56nJ/3+l1vq1ya/p/qck5/ekoQCwwi2h5rbbkeQPa61PdLt9ADAsmtbcUsrRSe5Icl2SjUlOSvLSUsrP9LC5wBzWDboBwILdlolPdP9Bko/UWh9ewHP+LsnTSWovGwYAQ6ZJzU2STF6F9YokP9ajtgHAMGlSc7cnOVhrvW3y9/tKKb+XiQupfqs3zQTm4gpmWDluS3JOkp9O29eGSilrSikbk4xM/Fo2llLWJ0mt9ckkv5/kLaWUI0spJ04+/0N9bz0ArByLrrltfizJ15J8ok9tBYCVrEnN/cLktP/P5HxbkvzbJIv9ewlAlwiYYYWYvL3F/5Pk8CQfaHvohzNxK4zdSbZN/vyRtsffkOSJJA8k+f9l4itH7+59iwFgZVpCzU0mbo9xW63Vt4cAYB5Nam6t9RtJ/lWSy5N8Ncmnk3w2yS/0qdnANMW5LwAAAAAATbiCGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI2s68ZCjjnmmDo6OtqNRQHAULjzzjsfq7Ue2+3lqrkAcCg1FwB6b65625WAeXR0NHv27OnGogBgKJRS7unFctVcADiUmgsAvTdXvXWLDAAAAAAAGhEwAwAAAADQiIAZAAAAAIBGBMwAAAAAADQiYAYAAAAAoBEBMwAAAAAAjQiYAQAAAABoRMAMAAAAAEAjAmYAAAAAABoRMAMAAAAA0IiAGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmAAAAAAAaWTfoBgDJrl27Mj4+Puhm5P7770+SbN26dSDrHxsbyyWXXDKQdQMwHPpVU/tVM9VGAFaaXtTiQYxV1WBYOAEzLAPj4+P59Gc/n4OHHT3Qdqx98utJkoe+0/+uYe2Tj/d9nQAMn37V1H7UTLURgJWoF7W432NVNRgWR8AMy8TBw47OU889f6Bt2HT37iQZSDta6waApepHTe1HzVQbAVipul2L+z1WVYNhcdyDGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmAAAAAAAaETADAAAAANCIgBkAAAAAgEYEzAAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhEwAwAAAADQiIAZAAAAAIBGBMwAAAAAADQiYAYAAAAAoBEBMwAAAAAAjQiYAQAAAABoRMAMAAAAAEAjAmYAAAAAABoRMAMAAAAA0IiAGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmAAAAAAAaETADAAAAANCIgBkAAAAAgEYEzAAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwEySZNeuXdm1a9egmwEMiD4Ali/vT1gY7xWg1/QzcCjvCVrWDboBLA/j4+ODbgIwQPoAWL68P2FhvFeAXtPPwKG8J2hxBTMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmAAAAAAAaETADAAAAANCIgBkAAAAAgEYEzAAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhEwAwAAAADQiIAZAAAAAIBGBMwAAAAAADQiYAYAAAAAoBEBMwAAAAAAjQiYAQAAAABoRMAMAAAAAEAjAmYAAAAAABoRMAMAAAAA0IiAGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmAAAAAAAaETADAAAAANCIgBkAAAAAgEYEzAAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCPrBt2ATvbt25drr702V199dTZv3tx4/oUuZ/p8rd8vvfTS3Hjjjdm/f39GRkbypje9KTfffHMuuuiivO1tb0utNb/wC7+Q3/7t384DDzyQm2++OWNjYxkfH89ll12W6667Lr/7u7+bgwcPJkn279+fBx98MCeccEKeeuqpPPjggzPactRRR+WrX/3q1O8jIyM58sgj8/jjj3ds+5o1a/LMM8/Mu48W6u1vf3uuu+66ri0PWBmefPLJfPGLX8wLXvCCGY9t2rQpTz311CHTjjzyyHzzm9+c6oOOOeaYPPbYYxkZGcn+/ftz4okn5sgjj8wVV1wx1W/u3LkzpZRcf/31efe7351vfvObuffee6eekySve93rcsstt+Tb3/52tm/fnne84x1T/Xd7X/3Vr341l156abZu3Zqf+7mfy8033zxrX9/qk2+66aaMjY1NTd+3b1927tyZgwcPZu3atbnhhhsWVHOGzWJr7lKX32R97XW5/Vi3LyvJ1M9f/vKX85a3vCU7d+7MLbfckq985Ss5+eSTc+ONN069dr7ne74nDz30UNatW5dSSp5++uls3bo1jz76aJ5++ulD1n/00Ud3rMPPe97z8ulPf3rJ+2ihLrnkkuzatatv64OV5vHHH8+9997bsZbNpVWH1q1blwMHDsw57xVXXJHjjz8+b3nLW/JTP/VT+d3f/d2ceOKJWbt2bR544IEkyTPPPJMDBw6k1pok2bJlSx5++OGsX78+xx13XB555JE85znPycMPP5wkqbVmzZo1ueGGG3LbbbdN1bk3vOENSZJt27blF3/xF2ftM2cbu7zmNa/J2972tpx00kmHPL99/l/7tV/LwYMH88wzz2T9+vW5/vrrZ4yHptfYufrx+fr4Xtec5W4hNXG2fTQ+Pj517vNLv/RL2bx5c/bs2ZM3v/nNOeWUU3LllVdOHaskeetb35p77703J5xwQjZu3Dg1lv3RH/3RXH/99Vm/fn22bt2aDRs25IorrsgNN9yQe+65J6Ojo7nqqqvyy7/8y/nKV74ytf6TTz45r33ta3PVVVel1pqTTz45v/iLv5gvf/nLefOb35xa69T54Lp163Lw4MGsX78+W7ZsySOPPJItW7bkqaeeykMPPdTfnd4D69YtyxgFBuLLX/5yvvGNbyy69g7a2rVrk2Qqr0u+m68dccQReeKJJ7JmzZqpfmzNmjV54IEHcuSRR+bRRx/N8ccfnyOPPHIq59u8eXPuv//+JMnWrVtz7bXX5uabb84555yTG2+8MSeeeGLWr1+fBx98MLt27Zoal7bGpKWUvOlNb8qv/Mqv5N57782WLVuyadOmXH/99UnSaPzUPv7uR80trROfpTj77LPrnj17utCcCTfeeGM++MEP5uUvf3kuv/zyxvMvdDnT52v9fvLJJ2fv3r1T842Ojuaee+7J4YcfnieeeCJJpl54rcdvueWW/MRP/ET27t17yGMrySc/+clBN2HVueyyy3Lnlx7OU889f6Dt2HT37iQZSDs23b07z99+XG666aa+r5vkpS99ab7zne90fbnz9ZvzufDCC6f67/a++q677prqn1vrmK2vb/XJrT665cYbb8wHPvCBjuvqhlLKnbXWs7u2wEmDrrlLXX6T9bXX5fZj3b6sWuvUzx//+MfzxBNPzAiLLrzwwkNeOyuRGj2/ftXUftRMtXFx+jG4LaUcUtO66Ygjjsi3vvWtGXUumbtGzTZ2aa+3nerp9Pnb55ut321/fqd+fL4+vlc1Z6XW3E77Y7Z91DqfSb57nC644IJDxqKtY1VrPeQcp/3xtWvXzvggZXR0dMa4t1OtnH4Od+GFF07V3NVGPZ5bL2pxv8eqavDCrLRguV9afW6STM9d28el7WPSTn3vhRdeeMg4ZzHjp9ZyuznOnaveLrtbZOzbty933HFHaq254447sm/fvkbzL3Q50+cbHx+f+n36gd27d29qrYcU0Paf9+7dm0984hNTz1uphfbtb3/7oJsA9NH4+HhPwuVk/n5zPrt3786+ffsO6at37959SP/cWkenvn58fHxq3r1792Z8fDzJRN9/++23HzLv7bffPm/NGTaLrblLXX57jV3o+tqX0X6s25d1++23T/384Q9/eOo1Nn0A/aEPfWhFh8vJxFXMwEx/+qd/2pf1TK9p3fTEE090rHPJd+vhdNP7yOnLm/78ueZPJmrh9PFQe589V92Yr6b0uuYsdwupibPto/bzmWTieH7iE5+YMRZt1cTdu3fPWH/r8U5X6Xca93Yy/bX/oQ99aMWOeZfqVa961aCbAAN37bXXDroJy1arz+10UW9rXDp9TNqp7929e3ej8VP7cvs1zl123+249dZbp275cPDgwdx2221zJu2zzb/Q5Uyf74YbbljSLSd+4Rd+ofFzl4tPfepTueyyywbdjFVlfHw8a55e+rcJVrI13/5Gxse/6bU3AHffffegmzCr/fv357bbbkutdapvbt1OY7pOff0NN9xwyDw33HBDbrnlltx6660zBlitdfXiKt7larE1d6nLb6+xC11f+zJapi+r/TUx19fb278Ct1J95jOf0U/OY5hqqtq4cHfdddegm9A1nercbDWqUx852zKn19PZ5us0Hmr12e3Pn96Pz1dTel1zlruF1MTZ9u/085n9+/fPOu7cv39/x0CjF4ahrjZ133336ZvnMAy1WA2e3zDV3n674YYbctZZZ817a679+/enlJJkceOn9uX2a5zb+ArmUsrFpZQ9pZQ9jz76aNca9LGPfWxqRxw4cCAf/ehHG82/0OVMn2/v3r3zHuC5LOW5AIPQq6uXu+WjH/3oIX31bDr19bNdkfOxj31sxuCr1jpvzRmU5VJzl7r89hq70PV1OvbTlzXb1QEAw6RTn7mQ+tj+/Pnmb1213Knfnf786f34fDWl1zWnW/pVczvVxNn2Uaer2mY7juohwPK3d+/ejmPSTlrzLGb81L7cfo1zG1/BXGt9V5J3JRP3pupWg84555zs3r07Bw4cyLp16/KSl7yk0fwLXc70+U488cTcd999jYPihfxxkJXAfYb6q3WPqtXsmY3Pyph7XA1E+z39lqOXvOQlU18Znqt/7dTXd7qnYDLR93/wgx88pPCWUuatOYOyXGruUpffXmMXur72ZbRMX1brU/3VMqjWT85tmGqq2rhw55xzzlCcg8+lU5/ZqY+c6/nz1dNSSk4++eQZ46FWn93+/On9+Hw1pdc1p1v6VXM71cTZ9m+n+3LONu4spayaejho+ubZDUMtVoPn5/7LzY2Ojuass86aMSbtpNWvL2b81L7cfo1zl909mHfs2JE1ayaatXbt2lx00UWN5l/ocqbPt3Pnzqnfm7jqqqsaP3e5+OEf/uFBNwHoo507dw66CbMaGRnJRRdddEhfPTIy0nHeTn399G1r/b5jx44ZfwG8ta7VZLE1d6nLb6+xC11f+zJapi9rZGRk6nUx1192b/216JXszDPPHHQTYFm68sorB92ErulU52arUZ36yNmWOb2ezjZfp/FQq8+eq27MV1N6XXOWu4XUxNn20fTzmZGRkVnHnSMjI3PWwm4ahrra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mmuuydjYWH791389SfLv//2/zxNPPJEkueqqq/LUpz41P/3TP51169allJJaa0opOXToUA4ePJgkOeaYY/L1r389SWanaeKEE07owNIDTHnooYdyzz335HnPe97sbWvWrMmTTz6ZkZGRHDx4cM7YdujQoRw6dChr1qzJunXrctppp+UXf/EXs2nTpkxOTuayyy7LO97xjmzdunXO8+zZsyc//dM/nbGxsWzevDljY2N505velHe+85255JJLctVVV80+bmY+11xzTW666aZcddVVSbJgHnQqK7qZOYNgqcvfOl3Sfrvs378/b33rW3Pw4ME89thj2bdvXzZv3pynPe1pufzyy3P99dfn8ccfz3333ZdnPetZSZL77rsvmzZtyr333pvR0dEcOHAgp512WtavX59vfOMb+eIXv5gtW7bk/PPPz7ve9a6cdNJJWbt2be69996sXbs2hw4dSpJs3749u3fvXjRnjyaH29m8eXPH5gWsTI8++mg+//nPz8nLpmbGoBNOOCFf+9rXkiRPPvlkDhw4kNHR0dkcPnDgwOz0p556amqt2bdvX0499dQcc8wx+fEf//FceeWVefzxx7Nly5b86q/+6mFj8c/+7M/mi1/8YmqtWbNmTY4//vh85Stfmc3vhx9+eDZz3/Oe9+SJJ57Ik08+mfvvvz833HDDYXm+kOVmqMz95vLPbIOdO3fm5ptvzhve8IZcf/31OXToUNauXZtXvvKVue666/ITP/ET+Z3f+Z2sWbMmhw4dyujoaI477rg88MADWbduXX7zN38zSfKTP/mTs6+dGVu2bOnHYgID4sCBA/nsZz/bkYxr4rjjjsvXvva12f5ZKSXHH398HnzwwYyOjmZkZCTbt2/PH/3RH+XFL35xdu/enbe+9a35wAc+kEOHDuXAgQO577778vjjj+f666/P05/+9DnHpTNj7hve8Ia87W1vyz/+4z/mjDPOyBVXXJHrr78+pZRce+21SZKJiYnZ37t13Nlrva67dOJAa9u2bXXPnj0dKGfK29/+9txyyy152ctelje+8Y0dm+/MvD/84Q/P/j4+Pp5nP/vZs7dt3LgxX//617N27drZF/n4+HiSZO/evbOPGxkZyfr16/PII490tL7FfOITn+jZc7F0l112We74wgN57Fu397uUbLhrV5KsmFrO23Ji3vGOd/S7FNroxE7ERRddlDe+8Y151atelb1792Z8fDw33njjnGle8pKXHDZOjo+P5+67784xxxyTRx55ZPZxM/OZGYdf9rKXpda6YB50Kiu6lTmllDtqrds6NsNp/crc1ukW2i7zM7bV+Pj4nBwdFrK5t/qduf3MWbnaHy960Yvy+OOP97uMOTZu3DgnW2fyeMZiY/HM9Hfeeeds5rbL6fl5vpDlZqjM/ebyt26Dr3/96zn99NMPO96cOR5dTLtj1Rlr1qzJxz72sQ5Vz9HoZ345RmQhr3jFK7J///5+l7EsC42NGzduzPHHHz/nuHRmzJ0/vrYel1x00UWptc7m5vxMTbrbo+ymbtS9WN6uuEtk7N+/P7fddltqrbnttts6+mLfv39/br311jm37d27d85tjzzySGqtc16we/fuPSywDx482NPmcpK8973v7enzAcOpUwcau3btyp49e2bHx71792ZycnL2/j179rQdJ/fu3Zta6+x9e/fuzcc+9rHZ+cyMw7t3714wDzqVFd3MnEGw1OVvnW6h7TIzzUKGsbmcJLfccku/SwC6ZHJycsU1l5Mclq27du2aMxbv3r170cd/5CMfmZO5883P84UsN0Nl7jeXf9euXYft97Q73lyKdseqM5588snccccdR1E1MKz2798/kOPwQmPjI488Mue49I477pgdc+ePka2/79q1K7t27Zr9fffu3V057uy1ftS94i6RcdNNN+XJJ59Mkhw6dCg333xzxzrtN9100+xXaVu1u20luvnmm3PnnXf2uwzmmZyczJonOveV62Gx5htfy+TkP+Wyyy7rdynM06lx5MCBA7n66qvn3HbdddfNnvU0/77FvO1tb2s7/xnz86BTWdHNzBkES13+1ukW2i433XTTYV/NXQ2uv/763H777f0uY9VYzZkrV3vvrrvu6ncJS3LgwIFljcVLOfZpzfOFLDdDZW77LO22N7/5zTn33HN79ny0t5rzq5UsWzn27dvX7xK66qqrrpodcxczfzxuzdRkcLOrH3U3PoO5lPLqUsqeUsqeBx98sGMF3X777bOfSBw8eDAf/ehHOzpvADpn/plPrZ8GL+dbHu0+ia61zl4vd34edCorupk5ndTvzG2dbqHtcvvtt3f0+sYA/bYSz15eSOtY3AlL+dbJcjNU5t6+5LOSO2kpDRZg9Xn44Yf7XUJXPfLII43G3FprV447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9rsMoA88/+HweR7ByuN5CTRh7ICVy/NzMIz1uwD6Z2pqqt8lAH3i+Q+Hz/MIVh7PS6AJYwesXJ6fg8EnmAEAAAAAaETADAAAAABAIwJmAAAAAAAaETADAAAAANCIgBkAAAAAgEYEzAAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhEwAwAAAADQiIAZAAAAAIBGBMwAAAAAADQiYAYAAAAAoBEBMwAAAAAAjQiYAQAAAABoRMAMAAAAAEAjAmYAAAAAABoRMAMAAAAA0IiAGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmAAAAAAAaETADAAAAANCIgBkAAAAAgEYEzAAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhnrdwHd7NmzJ9u3b88VV1yRjRs3Lnq5xd5uMduZ77r2y5NkcnIypZS8/e1vT5K86U1vyp133pkdO3bkN37jN1JrzY4dO2br+6Vf+qXs3r07v/Irv5InPvGJueSSS3LMMcfknnvuyaZNm3LUUUflsssuyzXXXJNLLrkk7373u3P77bfPbv+kk07KPffck7179+a4447LAw88kAsvvDAf+tCHsmnTpnznO9/Jgw8+eEj39wc/+MH8wi/8wiHdBhhse/fuzd/93d/l6U9/etfrSymptS56fevXr88jjzwye9uxsbHs3bt3zrpe8pKX5MMf/nD27duX1atXZ9WqVdm7d282b96co446Ki960YuyY8eOvOc978n4+Hi2b9+eF7zgBdmxY0de85rX5P3vf382btyYb37zm7n22mvzrW99K294wxtml3/zm9+cO+64I9dee20mJiaSTI/Zb33rW+eMxS2t8fySSy7JNddcs+jeMWy69btD7aeHY2pqKpdcckk2bNiQ++67L6973evyyU9+cs62p6amcvHFF+eUU07J2rVrs2PHjjz44IN59atfnVJKrr322hx77LF585vfnK9//evZvHlzXvziF+dtb3tbkuToo4/OP/zDP2T//v1JklNPPTWrV6/O17/+9YyNjWXfvn1da1u9evXsbeYzNjaWPXv2jORjB1aiu+66K9/85jfn7W/98IQnPCHf+c53kiQ/8RM/kb/4i7+Yd9lSSlavXj1nXFq9enUef/zx2THp5S9/ef7zf/7PKaXkjDPOyOWXX55rrrkm27Zty7//9/8+27dvzwc/+MHs3bs3+/fvz913351aa04//fS84Q1vyLve9a7ceeedOe200/LOd77zoGN/L+ZYo6LJ/nfOP+fr2ZdcckmuvvrqPPbYYymlzD6WWnPLiy66KFdccUXe9773ZWJiYra3btq0Ke94xztm1/2CF7wgb3vb27J27dps3rw5r3jFK2bnvK9//evzrne9K4899lhOPPHE3HfffTnppJNy7733zv7/cz/3c/mt3/qtnH766dm2bVve/va3Z9OmTRkbG8s999yTCy+8ML/+67++PHcwMLS+8Y1vzNvbTzvttDzhCU/Iy172srz1rW/NySefnP379895rb958+a8973vzde+9rW8/vWvT601J598cu699978yq/8yuz885JLLslVV101m/N19sM3vvGNufvuu3PNNdfMjqWXXnpp3ve+9+XYY4896Bg/35je/vPB+sNCueTh9O/DVQ4lQJjPueeeW3ft2rUE5Uy7+uqrc+ONN+b5z39+Xvva1y56ucXebjHbme+69strrfnYxz6WJLngggvm/L5hw4Y8/PDDs9e16mu//vjjj8/u3bsPqGt8fDy33357Tj/99K7XL4fPfOYzPdkO33fppZfmlq/em0eevLUv219/284k6ev2n7blpLzvfe/ry/ZH3Qtf+MLs2bOn32XM0Wr+GzZsyE/91E/lxhtvnJ1kdwbe4+Pj+eY3v5mHH354dvnW+Do+Pp7rrrsuSeaMu62xuKU1np9++um5/fbbF907FquUckut9dwlW+GMXvTcQ+2nh+OlL33pnF5XSkmSOdvuXOaCCy7IrbfeOnvZ+Ph4nvrUp84e6yQLBsdLrfOxxeHpZX/sZy/UB5fHSgqWe6U1dzjqqKNm+2JrHtJt2c7x9GBjfy/mWIdrkHvuodym1jpvz55v7tj5+Gi9Tmrvra25bPtrr5b2x9Oh9tZe9uKlYl689Po57+1ln9fXl8+ll16aW2+99aDLLdT/kumx7lOf+tQBy7TPP9vH0m7zx845Z2ssbc1HDjbGzzemdxvf57NQLnk4/XsxFuq3K+4UGXv27MnNN9+cWmtuvvnmeQOQzuWmpqYWdbvFbGe+69ovv+mmm7Jz587Z2+zcuXPO7+0P2Jtu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MAMAAAAA0IiAGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmAAAAAAAaETADAAAAANCIgBkAAAAAgEYEzAAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhEwAwAAAADQiIAZAAAAAIBGBMwAAAAAADQy1u8CutmzZ0+2b9+eK664Ihs3bux3OXMsdW2d69uzZ0/e+ta3ptaaHTt2JEne+ta35rHHHsuqVauyevXqXHbZZbn66quzd+/e7Nu3L3fddVcee+yxrFmzJqtWrcqpp56adevW5WUve1kmJydzyimnZPXq1dm/f3/uvvvuPOlJT8ratWvzspe9LG95y1vy6KOP5oorrshv//Zv54477sj69evz4IMPzqnz5JNPzre+9a2ceOKJueeee7J3797D3vckmZqaysTExJKsC4bd7bffnm9961t5+tOfvqTrXbNmTUop2bt3b2qtKaWk1po1a9Zk06ZNGRsby759+3LHHXdk7969OeKII/KOd7wjv/Ebv5HHHnssjz/+eO6888489thjWbt2bd74xjfmqquuyvbt2/PBD34w+/bty+rVq/Oyl70sV1xxRbZv354PfOADueOOO7Jjx45cf/31ueSSS3LNNdccMLZ2G3MPdRw+2PIruef0Qi/3f8+ePZmcnEwpJW9/+9vnbG9qaiqXXHJJTj311LzxjW/M1VdfnUceeST33HNPNm/enHe84x2zfXL79u255JJL8q53vSt33nlnjj322Nxzzz054ogjcvTRR+f+++/Pxo0b88ADD6TWmrVr1+akk07Kvffem0cffTRr167Nxo0bc88996TWOqfGE044IUcccUTuuuuug+7PjTfemOc973lLfj8B/bF379783d/93ZL32UN1zDHH5Fvf+tacy1q9+ZWvfGWuu+66PP7443nsscdy3HHH5YEHHkiSnHTSSbnvvvvypCc9KQ888EA2bdqUd7zjHUmS7du3Z9u2bXnTm96UvXv35rLLLsuP/diPzY7/SWbH59e97nWzc41SSlavXp0dO3bM2yOmpqZy6aWX5n3ve98Br+vbe8yDDz6Yiy++eLYuPXfh1ztN1tt6PdX5uqo1x2y9Jrvsssty1VVXzR7v1m3e/e5354477sjrX//6vOc978kpp5ySsbGx7N27N3fffXceffTRHHHEETn55JNnH2tr167NZZddll/+5V/O7t27c8YZZ+THfuzH8lu/9Vu56KKL8tnPfjYveMEL8ra3vS1r1qzJvn37cvzxx+f+++/PE5/4xHz7299OkqxatSqPP/747P9Jcuyxxx4wL11OV111VS677LKebQ9Gyde+9rU89NBDfemxrR66kFZm1m5sbCynnHJK7r///hxzzDH5xje+kVJKXv7yl+fXf/3X8zM/8zP5wz/8w9l586pVq3LEEUfk7W9/e5LkNa95Te64445cccUV2bRp0+xc5xWveEWuuOKKA/rmQnOl1vXtfbs9O1wJ89xysDt5Mc4999y6a9euJShn2tVXX50bb7wxz3/+8/Pa1752yda7FJa6ts71XX311fnYxz6WJLngggtSa539vWV8fDy7d+8+6Lo3bNiQhx9+eFHXtwKkXhsfH891113X8+0OoksvvTS3fPXePPLkrcu6nfW37UySnmznaVtOyvve975l3c4w6feEt91C40trPOlcpvV7++UbNmzIP/zDP+T000/P7bfffsDY2m3MPdRx+GDLL1fPKaXcUms9d8lWOGOQe25nj2vf3ktf+tLZ3tatz7WWb9V7+umnL6oXLqdSSj796U/3tYZR0ase2KlXPXGh7euVvfPCF74we/bs6XcZS6o1n7jxxhtz1FFHzfbfUkqe97znzY7/7XOOhcbgblrjd7fX9e095tZbb51d70Lra2LQe+7h9uL23nj77bcf8Lqqvf8mc4/x+Pj47G1alx3q3HCh+WnrTYp+zDWb+MxnPtPvEgbWsM1X27enFx++lTSXXW6dWd7Y2FhOO+202XGyNR/u7JsLzZVa13fr2/P11OWY5y3Ub1fcKTL27NmTm2++ObXW3HzzzSvqRd5S19a5vqmpqdx8882z1+/cuTM33XTTAbdb7IR6oXC58/p+Nfzdu3dnamqqL9uGQfKud72r3yXMsdD40hpPOpdp/d5++cMPP5xaa3bv3n3A2NptzD3Ucfhgy6/kntMLvdz/PXv2zOlpN9100+z2pqam5vS2bn1u586ds32y9Zjpt1ZoAwy+Vo8ZNh//+Mdnx832/ltrzcc//vHUWnPTTTdl586ds9d1G1/bx+x27eN35+v69h6zc+fOOevduXPnUN7fC5mv5x5uL26/fev1VPvrqs45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8274jPW7AFisqampfpcAS87jGlgJjEWMIo97IDEWQD943g0fn2AGAAAAAKARATMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmAAAAAAAaETADAAAAANCIgBkAAAAAgEYEzAAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhEwAwAAAAAQCMCZgAAAAAAGhEwAwAAAADQiIAZAAAAAIBGBMwAAAAAADQiYAYAAAAAoBEBMwAAAAAAjQiYAQAAAABoRMAMAAAAAEAjAmYAAAAAABoRMAMAAAAA0IiAGQAAAACARgTMAAAAAAA0ImAGAAAAAKARATMAAAAAAI0ImAEAAAAAaETADAAAAABAIwJmAAAAAAAaETADAAAAANCIgBkAAAAAgEYEzAAAAAAANCJgBgAAAACgEQEzAAAAAACNCJgBAAAAAGhkrN8FdLNnz55s3749V1xxRTZu3Njz7UxNTeXSSy/N+973vkxMTBxwu127duUNb3hDJicn89GPfnTBOnft2pVf+qVfytq1a3P66afnHe94R5Jk+/btueSSS3LVVVdl//79eeSRR3LnnXem1ppSSsbGxrJ3794kyStf+cr82Z/9Wfbv359aa7773e/mrrvuyvHHH59vfvObWbNmzeyySVJKSZLUWpfsvlpJnv3sZ+cTn/hEv8uAJfGlL30p3/ve9/L0pz+936Us6LjjjssDDzyQJLnwwgvzoQ99KKtWrcr+/ftz/PHH5+GHH87+/fuzd+/eHHPMMfnWt76V448/Pt/97ndzzTXXZGJiIlNTU7nkkkuycePG3H///UmS//gf/2PXcbY1Pl9yySW55pprcsUVVyTJnDF7z549mZyczL59+7JmzZq8/e1vnx2L28f39tt1rqNz2eXsOSvVcu7/oRyHg9XVrc6pqalcfPHFOeWUU7Jq1ao8/vjjueeee7Jjx4782q/9Wu66665s2rQp73jHO/Lggw/m4osvzgknnJD7778/b3jDG/Lud787p5xySpLk7rvvzgknnJA9e/bkmmuuybHHHjv7GLz66qvz2GOPpZSS/fv355577snrX//6vPOd78xjjz2WJDnttNNSa81dd901W+/mzZtzySWX5K1vfWtqrVm/fn0efPDBrFu3Lt/73vcWfT9u3LgxDzzwwKL7+tFHH52HHnpo0etvWb16dfbs2TOSzwNG1+c+97me9eBTTjklDz74YI455ph84xvfSJKMjY3lpJNOyl133ZVSypy5wL59+7Jp06asXr0699xzT0488cR84xvfyGOPPZa1a9dm8+bNeeMb35hrrrlmTr/s7G+tcazWmssuu+yAvjrf9Z1jb2v5w+kXeu7h73+310gLravbNg922YMPPphLL70027dvzw033HDQft3qc63HT6u2bdu2ZXJyMieccELuvfferFq1Ki984QvzW7/1W7nwwgvze7/3e3nssceyZs2arF69OkcffXTuu+++RvfLUvhP/+k/5VWvelXftg+j5Gtf+1oeeuihvs6Djz322Dz44IM59dRTs3bt2tx111159NFHc8QRR2TVqlU58cQTc99992Xz5s15wxvekHe+8525884786QnPSnr1q3L6173ulxzzTV55jOfmfe+97254oor8oQnPCGvf/3rc8YZZ+Q973nP7Hjayhdbc9hSSt7+9rcnWXxvXUxPbu8RV1111ex2etFzy1KEkOeee27dtWvXEpQz7eqrr86NN96Y5z//+Xnta1+7ZOtd7HZe+tKXZvfu3RkfH8911113wO2e+9zn5uGHH87Y2Fj279+/YJ2tZVsuuOCC1Fpz44035vTTT8/u3buXerdGwmc+85l+l9B3l156aW756r155Mlbl2X962/bmSRLvv71t+3M07aclPe9731Lut5BtdKD5aXQGktbY2u36zq1xufTTz89t99+e57//OfPjp2tMffqq6/Oxz72sdnbXHDBBbNjcfv43n67znV0LruUPaeUckut9dwlW+GMQeq5h3IcDlZXtzq7PaaSZMOGDQf03ltvvXXOsq3wppvx8fE89alPXbBXL3T7hWpZ6dqfR0xbzn67XL22cxv6bneXXnppbr311n6XcVjGx8dz++23z+mXnf2tfRxrLd8+Fs93fefYu5ix+2D03MPf/26vkRZaV7dtHuyyVs/csGFD/uEf/uGg/br1eqzz8XjUUUcNVA9MzDMP13LPUTv1oo92bk9PXRqDNg8eHx/vOpe9/fbbk0x/wHNsbCzr1q2bHffa5yCteW/7mNmeDy6mLyymJ3fr/Uv5+n6hfrviTpGxZ8+e3Hzzzam15uabb86ePXt6up2pqanZg7B79+5MTU3Nud2uXbtmHyz79u1bsM72ZVs+/vGPz25XuNzcs5/97H6XAIft3/27f9fvEnpi9+7d+a//9b92HfO6jbPt4/Pu3btTa81NN92Um266aXbMnZqayk033TTndjfddFP27Nkz5/btt7vpppsOGPd71XNWquXc/0M5Dgera2pq6oDbtPfrTt16b+eyC4XDu3fvnq13vm0sJlzuVstKd+ONN47c84DR9bnPfa7fJRy2Vp9s/d+tv7WPY+19daHrO8fexYzdB6PnHv7+d3uNtNC6um3zYJft3Llz9jHx8MMPL6pft3Q+HgetBybTn2IGltf27dv7XcIhm28uW2ud/Zbhvn375ox7N95445x88ZZbbpkzh925c+ecOe5CfaFzbtWtn8zX+1vz5OW24k6Rcf311+fxxx9Pkuzfvz833HDDsnySZr7t7NixY85yO3bsmPPpuiuvvPKAdc1XZ7dl9+3bN3sKC5rbu3dvLr300n6X0VdTU1NZ9djgnQZl1fceytTUd0b++CXJF7/4xX6X0DP/4T/8h3mv6xxn28fnlvbTAO3fvz87duw4IODbu3dvbrjhhtRaZ2/ffrvOdXQuu5w9Z6Vazp7bvu6DHYfObXbWtWPHjgPqPJRPHS42DG7XXucoefzxx/Pyl788p512Wr9LWTEGtd+26LvzG8bTyXXrb90cbIzrHHsXM3YfTK/meSvVUux/t9dIC62r2za7vfaZ77XTwbZx/fXXD12//P3f//38/d//fb/LGFiD3jMPRk9dGoP+7aHF6hyvr7jiijnzkr17987mgwfrC/PNrdpv161HtJbvRc9t/AnmUsrLSym7Sim7WufSXAqf/OQnZ+/wffv2Ldu5dufbTue7Ep2/d3sXdr4653vHdhhfzAIs5GCfFm3XPj63dL4z3Hq3uHOZT3ziE3Nu3367znV0LrucPedwDWLPPZTjcLC6du/efUCdvgW0fB588MF+lwA01K2/ddM+Fs+3nvaxdzFj98HouYe//92O60Lr6rbNg13WzUL92twWYHFa3wppt9jeOt/cqv12843lrXnycmv8CeZa6weSfCCZPjfVUhX0rGc9Kzt37sy+ffsyNja2bKdCmG87nedVGR8fn3O7budSnK/O+c672PoDHhyeUT/vUev8VoPm8XVHZ8J5q5IM3nmnDsfBznfbrn18bmn/46VjY2M57bTTcvvtt88ZS0spefaznz379c72b4y0/mhS+zo6l13OnnO4BrHntq/7YMfhYHWddtpp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QCcCZgAAAAAAOhEwAwAAAADQiYAZAAAAAIBOBMwAAAAAAHQiYAYAAAAAoBMBMwAAAAAAnQiYAQAAAADoRMAMAAAAAEAnAmYAAAAAADoRMAMAAAAA0ImAGQAAAACATgTMAAAAAAB0ImAGAAAAAKATATMAAAAAAJ0ImAEAAAAA6ETADAAAAABAJwJmAAAAAAA6ETADAAAAANCJgBkAAAAAgE4EzAAAAAAAdCJgHlOXX355Lr/88lGXASwDxgMYDe89YC7GBlgevBdh5fG+H53Voy6AbjZv3jzqEoBlwngAo+G9B8zF2ADLg/cirDze96PjCGYAAAAAADoRMAMAAAAA0ImAGQAAAACATgTMAAAAAAB0ImAGAAAAAKATATMAAAAAAJ0ImAEAAAAA6ETADAAAAABAJwJmAAAAAAA6ETADAAAAANCJgBkAAAAAgE4EzAAAAAAAdCJgBgAAAACgEwEzAAAAAACdCJgBAAAAAOhEwAwAAAAAQCcCZgAAAAAAOhEwAwAAAADQiYAZAAAAAIBOBMwAAAAAAHQiYAYAAAAAoBMBMwAAAAAAnQiYAQAAAADoRMAMAAAAAEAnAmYAAAAAADoRMAMAAAAA0ImAGQAAAACATgTMAAAAAAB0ImAGAAAAAKATATMAAAAAAJ0ImAEAAAAA6ETADAAAAABAJwJmAAAAAAA6ETADAAAAANCJgBkAAAAAgE5Wj7qAudx333254IILcv755+fAAw8ci8eeebskueCCC/KKV7wil1122S6nnX766Tn//PNz4YUX5oorrthu3gte8IJcdNFFWbt2bY488sj81m/9Vq666qp88pOfTJJs3rw5GzduXMxNAIyZL37xi7n//vvzrGc9a4/va9WqVdm2bVuqKoceemi++c1v5oADDshdd92V1lrWrl2bvfbaK4985CNzzz33JEmqKocffniS5Pbbb5++jzVr1uQxj3lM7rvvvrzsZS/LpZdemtZafuM3fiNXX331DmPjJZdckoceeihr1qzJRRddlAMPPHB6bB2MnYMx8aijjsqb3vSmXY7TCxnTR9lzloPF6HuD2803bdOmTamqvOpVr8pll12WF77whbnwwgtz+OGHZ/369bnooouSZLvb3nfffXnta1+b7373u7nzzjtTVdOvu3POOSfXXXddtm3blocffjh77bVXHnjggdx111157Wtfm//+3/97vvzlL2fdunV59atfnde//vX53ve+N+d6VFVaa1m9enW2bt3aaRvefPPNedrTntbptsDkeeCBB/KFL3xht/vyoH8OrF27No95zGNyzz335FGPelS++tWvZv/995/uzd/4xjeyZs2a7LXXXjnyyCNz2mmn5eKLL85rX/va/Pmf/3m++93v5u67784v/dIv5dJLL83hhx+e/fbbLxdffPH0OPua17wmd911Vy677LLpfYrZ+zXnnHNObr/99rzpTW/a6Vg3GLdba9OPMdcyem739Z9rn3N37mtnjz/fvMHz+tBDDyVJ1qxZk5/5mZ/J+eefn3POOSdvfOMbkyS/+Zu/mbe+9a158MEHc+eddyZJNmzYkHXr1uWkk07KpZdemiR5yUtekquuuipvectbcvvtt+fCCy/cac2z3xdd6NOwstx5551z9uDB3/0z7bXXXnn44Yd3eZ8bNmzI2WefnfPOOy/f+973pjO6X/iFX8h5552XBx98MGvXrs2hhx6ar33tazn44INz77335pBDDslDDz2UO++8M+ecc04+9KEPTe/bvuIVr8gb3vCG3Hbbbfne976XI444InvttVfuueeeefvyQsfuXc1bCjV743ZxwgkntJtuumkRyum55JJLcu211+b5z39+zj777EW736V87Jm3a63l2muvzZFHHpnbbrttl9P222+/3H///Vm/fn3++Z//ebt5q1at2m6H99RTT8373ve+6etTU1N5+9vfvpibgAU666yzcvOXvpbvPP6UoT/2PrdelyQjeezZdTzt6EPylre8ZaR1rHSLESwvtZnNfPXq1dm2bdsOY+OWLVumlz/11FNz9tlnT4+tg7Fz5pg4WGZnFjKmL1XPqaqbW2snLNod9i2XnjvX7eabds011yTp9azZz2PSey4Hr4XBbWfebra5/jgcmB0U70lwvFDr16/P+9///iV9DHZulD15YNS9WU9ePp773OfmwQcfHPrjDsa72ePe7DFzZo+dOT4P9ilm79cMltnVWDfz/ubr0Xrunq3/XPucu3NfO3v8+ebN1Y/Xr1+f+++/f7vX2mDaXObq21NTU7njjjuWvEcPatOnh2ul92U9eXTOOuus6QMyF9tc49zOxr7ZqipJpvdtZ+8DzzRfX17o2L2reV3trN8uu1Nk3HfffbnhhhvSWssNN9yQ++67b9k/9szbXX/99dOXt2zZMj3t+uuvn3fa4MV4//337zBvdsOdGS4nyZYtW7J58+bF2QDA2Dn33HNHXcKCzNyp2Lp165xj40zXX399Nm/evMN4OnNMvO6663Y6Ti9kTB9lz1kOFqPvDW4337Trr79++nZzPY9J77kcvBZuuOGG6ed+Pjv7cHz2fQ9jx/X+++/PzTffvOSPAyx/mzdvHkm4nPzLeDd73Js9Zg567HXXXTc9bbBPMXu/ZmYot7OxbnC7mY8xu6fouXu2/vPtcy70vnb2+PPNm/28Dgz2X2e+1nYWsMzVt7ds2TKUHp3o07CSDL5BsRTmGucWGi4nvbFw5r7tfOFyMndfXujYvat5S2XZnSLjiiuumD48fdu2bbnyyiuHdhRz18eeebvBV4dmWui0hcyby8tf/vI8/vGP363bsOc2b96cvb63598AGGd7ffdb2bz52znrrLNGXcqKtVSfzg7DrsbBiy++eKdfV3rooYd2Ok4vZEwfZc9ZDhaj7w1u11qbc9pCdh4feuih6U/0t23blosvvni3e+Go/eqv/mqe+MQnjrqMFUtP1pOXi1tvvXXUJezSoMfOHp8vvvjiPOlJT9puv2Z2MHj++efPeSToFVdcsd24PVeP1nP3bP3n2+dc6H3t7PHnmzf7eR1n+vRwrfS+rCePzte//vVRl7BoZvflhY7du5q3VDofwVxVv1BVN1XVTffee++iFXTjjTdu9+n7hz70oUW776V67Jm3G3wiMdNCpy1k3lxGdZQEwJ7Y1Ti4kCNbdjZOL2RMH2XP2R3LrefOdbv5pi20nw2W27p16/Sn+uNkIeduAybfOPxdPt9RU1u2bNlhv2a2+Y7Umj3et9Z26Cl67p6t/3z7nAu9r509/nzzdqePL3f6NDBuZvflhY7du5q3VDofwdxa+4Mkf5D0zk21WAWdeOKJue6666bPH/ajP/qji3XXS/bYM283OAJrZiNe6LSFzJvL1NSUc/uMwOC8UivZw3s/MhudW2qkxuH8y/PZ1Th45JFH7vLcfDsbpxcypo+y5+yO5dZz57pda23Oaddee+2C+tnMH9s77LDDctttt43VTu369euNhSOkJ+vJy8VLX/rSnX7ldTkY9NjZdU5NTeVJT3rSdvs1s8fh9evXz3mfJ5544nbjfVXt0FP03D1b//n2ORd6Xzt7/PnmzX5ex5k+PVwrvS/ryaMzzvvHs83uywsdu3c1b6ksu3Mwn3HGGdlrr15Zq1atyumnn77sH3vm7dasWZM1a9ZsN3/NmjVZvXr1LqctZN5cNm3atOBlgcnyQz/0Q6MuobNdjYObNm2aHlvnW2Zn4/RCxvRR9pzlYDH63uB2801bSD+b+VpYtWpVNm3atEMvXe4uuOCCUZcALAPj8Hf5oMfOHp83bdq0w37N7D4831h3xhlnbDduz9Wj9dw9W//59jkXel87e/z55s1+XseZPg0rw0EHHTTqEhbN7L680LF7V/OWyrILmA888MCcdNJJqaqcdNJJOfDAA5f9Y8+83cknnzx9eWpqanraySefPO+0wZEA69ev32He7D/8Tj311O2uT01NZePGjYuzAYCx81/+y38ZdQkLMjjSJun9yv1cY+NMJ598cjZu3LjDeDpzTDzllFN2Ok4vZEwfZc9ZDhaj7w1uN9+0k08+efp2cz2PSe+5HLwWTjrppOnnfj4zX0+zzb7v3fnAtqv169fnaU972pI/DrD8bdy4MevWrRvJYw/Gu9nj3uwxc9BjTznllOlpg32K2fs1z3ve86aX2dlYN7jdzMeY3VP03D1b//n2ORd6Xzt7/PnmzX5eBwb7rzNfa/Md3Z7M3benpqaG0qMTfRpWkg0bNizZfc81zu1s7Jutqrbbt529DzzTXH15oWP3ruYtlWUXMCe9pP2JT3ziSD7V7vrYM283uLxp06YFTXvd616X/fbbLxdccMEO884999xUVdatW5djjz02p59+en7qp35q+nHH4SgJYGntTlPblVWrViXpNb/HPvax2WeffbJhw4bpHYO1a9dm7733zsEHHzx9m6rKEUcckSOOOGK7+1izZk0OP/zw7Lvvvjn77LOn7+Pcc8+dc2w8/vjjc8wxx+T444/f7siZmWPnYEw8+uijF3y0zq7G9FH2nOVgMfrerqYdd9xxOf7446efx/POO2/6dTN4vmff9owzzsjxxx+fo48+OuvWrdvudfeqV70qxx13XI499ths3Lgxxx57bA477LDstddeOe+883L00UenqrL33nvn3HPPzdq1a+ddj8Hrck92ch0VBcw06Ie7a9A/B9auXZsjjjgie++9dx7zmMckSfbff/8kyQEHHJCk12sH+wnnnnvu9Dg4GD/32Wef6R58xBFH5LjjjttunN24cWP23Xff7fYpZu/XTE1N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fLM/0F5uH3ADAN0tu1NkXHHFFXn44YeTJNu2bcvFF1+83fUrr7wyrbU89NBDO9x2sPzMEGHw9em5zLX8MFxwwQW5+uqrh/qYdLd58+bs9b09P9J/0u313W9l8+Zv56yzzhp1KeyGT37yk6MuYdFs27Ztu+tbt27N6173ujmX3bp16w69YdBjFvMoZgAAAJh0nY9grqpfqKqbquqme++9d9EKuvHGG6cD361bt2bLli3bXf/Qhz6UG2+8MXOd2mOw/Ox5850GZL7lAZgMM49onm322D/oMcvRUvVcAAAA2FOdj2Burf1Bkj9IeueDXKyCTjzxxFx33XXZunVrVq9encMOOyx33HHH9PUf/dEfnT5v5uxwYLD8bbfdtt28qpozRJ5v+aW2evVq56kdI4NzMLNzD+/9yGx0DuaxM+lfT515DufZZveGQY9Zjpaq5wIAAMCeWnbnYD7jjDOy1169slatWpVNmzZtd/3000/PGWeckTVr1uxw28Hyq1f/S26+Zs2aOZedb/lhOO+884b6eADzedGLXjTqEhbNqlWrtru+evXqeU+RsXr16h16w6DHAAAAAAu37ALmAw88MCeddFKqKieddFI2bty43fUDDzxwepmBqamp7ZY/+eSTp+edcsop07cfLLez5Zfa6tWr8+xnP3tojwewM694xStGXcIOpqamMjU1laR3BPLM6XNdHnje85633fI//uM/nhNOOGF62ZkfJv74j//4Dr1h0GMAgMX30Y9+dKfXAYDxtewC5qR3FPMTn/jE6SPJZl8fTDv++ONz3HHHZdOmTTssf9xxx+X444+fPuL5iU984vRy8y1/9NFHZ926ddl7772zYcOGJMm6dety9NFH59BDD02SHHzwwXPWvP/++y9o3Ry9DCw3BxxwwKLd19q1a+f8VsjMI6XXrl2bY489dvrDvarKunXrpsffTZs2ZdOmTdlvv/1ywQUXTI/nmzZt2m7cP/7443PMMcfk2GOPzXHHHZfTTz89r3vd67LXXnvlqKOOmh7jB/d17rnnZt99980xxxwzZ29w9DIAAADsvlqMcw+fcMIJ7aabblqEcibXWWedlSTOTzuGBudg/s7jTxl1KdP2ufW6JFl2NT3NOZjHkvFpaVTVza21Exb7fhe75w7Ow+1IMgDG1VL1XABgYZblEcwAAAAAACx/AmYAAAAAADoRMAMA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\n",
      "text/plain": [
       "<Figure size 1440x3600 with 41 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 13;\n",
       "                var nbb_unformatted_code = \"cols = data.columns.tolist()\\n\\nplt.figure(figsize=(20, 50))\\nfor i, variable in enumerate(cols):\\n    plt.subplot(14, 3, i + 1)\\n    sns.boxplot(data[variable])\\n    plt.tight_layout()\\n    plt.title(variable)\\nplt.show()\";\n",
       "                var nbb_formatted_code = \"cols = data.columns.tolist()\\n\\nplt.figure(figsize=(20, 50))\\nfor i, variable in enumerate(cols):\\n    plt.subplot(14, 3, i + 1)\\n    sns.boxplot(data[variable])\\n    plt.tight_layout()\\n    plt.title(variable)\\nplt.show()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cols = data.columns.tolist()\n",
    "\n",
    "plt.figure(figsize=(20, 50))\n",
    "for i, variable in enumerate(cols):\n",
    "    plt.subplot(14, 3, i + 1)\n",
    "    sns.boxplot(data[variable])\n",
    "    plt.tight_layout()\n",
    "    plt.title(variable)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- There are positive and negative outliers for all attributes \"V1\" to \"V40\". The scale of attributes are more or less the same (somewhere between -20 to +20). Since not much is known about the attributes, the outliers will not be treated and are assumed to be real data trends"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "0    37813\n",
       "1     2187\n",
       "Name: Target, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 14;\n",
       "                var nbb_unformatted_code = \"plt.figure(figsize=(6, 4))\\nsns.countplot(data[\\\"Target\\\"])\\nplt.show()\\ndata[\\\"Target\\\"].value_counts()\";\n",
       "                var nbb_formatted_code = \"plt.figure(figsize=(6, 4))\\nsns.countplot(data[\\\"Target\\\"])\\nplt.show()\\ndata[\\\"Target\\\"].value_counts()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(6, 4))\n",
    "sns.countplot(data[\"Target\"])\n",
    "plt.show()\n",
    "data[\"Target\"].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- \"Target\" class is imbalanced with 37813 or 94.53% \"No failures (i.e., 0)\" and 2187 or 5.47% \"Failures (i.e., 1)\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "knk0w9XH4jao"
   },
   "source": [
    "## Data Pre-processing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "2JbJc1bX4jao"
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 15;\n",
       "                var nbb_unformatted_code = \"# Split data\\ndf = data.copy()\\n\\nX = df.drop([\\\"Target\\\"], axis=1)\\ny = df[\\\"Target\\\"]\";\n",
       "                var nbb_formatted_code = \"# Split data\\ndf = data.copy()\\n\\nX = df.drop([\\\"Target\\\"], axis=1)\\ny = df[\\\"Target\\\"]\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Split data\n",
    "df = data.copy()\n",
    "\n",
    "X = df.drop([\"Target\"], axis=1)\n",
    "y = df[\"Target\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(28000, 40) (12000, 40)\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 16;\n",
       "                var nbb_unformatted_code = \"# Splitting data into training and validation sets:\\n\\nX_train, X_val, y_train, y_val = train_test_split(\\n    X, y, test_size=0.30, random_state=1, stratify=y\\n)\\nprint(X_train.shape, X_val.shape)\";\n",
       "                var nbb_formatted_code = \"# Splitting data into training and validation sets:\\n\\nX_train, X_val, y_train, y_val = train_test_split(\\n    X, y, test_size=0.30, random_state=1, stratify=y\\n)\\nprint(X_train.shape, X_val.shape)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Splitting data into training and validation sets:\n",
    "\n",
    "X_train, X_val, y_train, y_val = train_test_split(\n",
    "    X, y, test_size=0.30, random_state=1, stratify=y\n",
    ")\n",
    "print(X_train.shape, X_val.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- There are 28000 rows in the training and 12000 rows in the validation sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    26469\n",
       "1     1531\n",
       "Name: Target, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 17;\n",
       "                var nbb_unformatted_code = \"y_train.value_counts()\";\n",
       "                var nbb_formatted_code = \"y_train.value_counts()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "y_train.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    11344\n",
       "1      656\n",
       "Name: Target, dtype: int64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 18;\n",
       "                var nbb_unformatted_code = \"y_val.value_counts()\";\n",
       "                var nbb_formatted_code = \"y_val.value_counts()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "y_val.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Stratify has maintained a distribution of 94.53% \"No failures\" or \"0\" and 5.47% \"Failures\" or \"1\" in both the test and validation splits"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Missing-Value Treatment\n",
    "\n",
    "* We will use median to impute missing values in \"V1\" and \"V2\" columns. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 19;\n",
       "                var nbb_unformatted_code = \"imputer = SimpleImputer(strategy=\\\"median\\\")\\nimpute = imputer.fit(X_train)\\n\\nX_train = impute.transform(X_train)\\nX_val = imputer.transform(X_val)\";\n",
       "                var nbb_formatted_code = \"imputer = SimpleImputer(strategy=\\\"median\\\")\\nimpute = imputer.fit(X_train)\\n\\nX_train = impute.transform(X_train)\\nX_val = imputer.transform(X_val)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "imputer = SimpleImputer(strategy=\"median\")\n",
    "impute = imputer.fit(X_train)\n",
    "\n",
    "X_train = impute.transform(X_train)\n",
    "X_val = imputer.transform(X_val)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ONL1sM1n4jap"
   },
   "source": [
    "## Model evaluation criterion\n",
    "\n",
    "### 3 types of cost are associated with the provided problem\n",
    "1. Replacement cost - False Negatives - Predicting no failure, while there will be a failure\n",
    "2. Inspection cost - False Positives - Predicting failure, while there is no failure \n",
    "3. Repair cost - True Positives - Predicting failure correctly\n",
    "\n",
    "### How to reduce the overall cost?\n",
    "* We need to create a customized metric, that can help to bring down the overall cost\n",
    "* The cost associated with any model = (TPX15000) + (FPX5000) + (FNX40000)\n",
    "* And the minimum possible cost will be when, the model will be able to identify all failures, in that case, the cost will be (TP + FN)X15000\n",
    "* So, we will try to maximize `Minimum cost/Cost associated with model`"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "djQTqGKU4jap"
   },
   "source": [
    "### Let's create two functions to calculate different metrics and confusion matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "id": "bIekBxwp4jaq"
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 20;\n",
       "                var nbb_unformatted_code = \"# defining a function to compute different metrics to check performance of a classification model built using sklearn\\ndef model_performance_classification_sklearn(model, predictors, target):\\n    \\\"\\\"\\\"\\n    Function to compute different metrics to check classification model performance\\n\\n    model: classifier\\n    predictors: independent variables\\n    target: dependent variable\\n    \\\"\\\"\\\"\\n\\n    TP = confusion_matrix(target, model.predict(predictors))[1, 1]\\n    FP = confusion_matrix(target, model.predict(predictors))[0, 1]\\n    FN = confusion_matrix(target, model.predict(predictors))[1, 0]\\n    Cost = TP * 15 + FP * 5 + FN * 40  # maintenance cost by using model\\n    Min_Cost = (\\n        TP + FN\\n    ) * 15  # minimum possible maintenance cost = number of actual positives\\n    Percent = (\\n        Min_Cost / Cost\\n    )  # ratio of minimum possible maintenance cost and maintenance cost by model\\n\\n    # predicting using the independent variables\\n    pred = model.predict(predictors)\\n\\n    acc = accuracy_score(target, pred)  # to compute Accuracy\\n    recall = recall_score(target, pred)  # to compute Recall\\n    precision = precision_score(target, pred)  # to compute Precision\\n    f1 = f1_score(target, pred)  # to compute F1-score\\n\\n    # creating a dataframe of metrics\\n    df_perf = pd.DataFrame(\\n        {\\n            \\\"Accuracy\\\": acc,\\n            \\\"Recall\\\": recall,\\n            \\\"Precision\\\": precision,\\n            \\\"F1\\\": f1,\\n            \\\"Minimum_Vs_Model_cost\\\": Percent,\\n        },\\n        index=[0],\\n    )\\n\\n    return df_perf\";\n",
       "                var nbb_formatted_code = \"# defining a function to compute different metrics to check performance of a classification model built using sklearn\\ndef model_performance_classification_sklearn(model, predictors, target):\\n    \\\"\\\"\\\"\\n    Function to compute different metrics to check classification model performance\\n\\n    model: classifier\\n    predictors: independent variables\\n    target: dependent variable\\n    \\\"\\\"\\\"\\n\\n    TP = confusion_matrix(target, model.predict(predictors))[1, 1]\\n    FP = confusion_matrix(target, model.predict(predictors))[0, 1]\\n    FN = confusion_matrix(target, model.predict(predictors))[1, 0]\\n    Cost = TP * 15 + FP * 5 + FN * 40  # maintenance cost by using model\\n    Min_Cost = (\\n        TP + FN\\n    ) * 15  # minimum possible maintenance cost = number of actual positives\\n    Percent = (\\n        Min_Cost / Cost\\n    )  # ratio of minimum possible maintenance cost and maintenance cost by model\\n\\n    # predicting using the independent variables\\n    pred = model.predict(predictors)\\n\\n    acc = accuracy_score(target, pred)  # to compute Accuracy\\n    recall = recall_score(target, pred)  # to compute Recall\\n    precision = precision_score(target, pred)  # to compute Precision\\n    f1 = f1_score(target, pred)  # to compute F1-score\\n\\n    # creating a dataframe of metrics\\n    df_perf = pd.DataFrame(\\n        {\\n            \\\"Accuracy\\\": acc,\\n            \\\"Recall\\\": recall,\\n            \\\"Precision\\\": precision,\\n            \\\"F1\\\": f1,\\n            \\\"Minimum_Vs_Model_cost\\\": Percent,\\n        },\\n        index=[0],\\n    )\\n\\n    return df_perf\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# defining a function to compute different metrics to check performance of a classification model built using sklearn\n",
    "def model_performance_classification_sklearn(model, predictors, target):\n",
    "    \"\"\"\n",
    "    Function to compute different metrics to check classification model performance\n",
    "\n",
    "    model: classifier\n",
    "    predictors: independent variables\n",
    "    target: dependent variable\n",
    "    \"\"\"\n",
    "\n",
    "    TP = confusion_matrix(target, model.predict(predictors))[1, 1]\n",
    "    FP = confusion_matrix(target, model.predict(predictors))[0, 1]\n",
    "    FN = confusion_matrix(target, model.predict(predictors))[1, 0]\n",
    "    Cost = TP * 15 + FP * 5 + FN * 40  # maintenance cost by using model\n",
    "    Min_Cost = (\n",
    "        TP + FN\n",
    "    ) * 15  # minimum possible maintenance cost = number of actual positives\n",
    "    Percent = (\n",
    "        Min_Cost / Cost\n",
    "    )  # ratio of minimum possible maintenance cost and maintenance cost by model\n",
    "\n",
    "    # predicting using the independent variables\n",
    "    pred = model.predict(predictors)\n",
    "\n",
    "    acc = accuracy_score(target, pred)  # to compute Accuracy\n",
    "    recall = recall_score(target, pred)  # to compute Recall\n",
    "    precision = precision_score(target, pred)  # to compute Precision\n",
    "    f1 = f1_score(target, pred)  # to compute F1-score\n",
    "\n",
    "    # creating a dataframe of metrics\n",
    "    df_perf = pd.DataFrame(\n",
    "        {\n",
    "            \"Accuracy\": acc,\n",
    "            \"Recall\": recall,\n",
    "            \"Precision\": precision,\n",
    "            \"F1\": f1,\n",
    "            \"Minimum_Vs_Model_cost\": Percent,\n",
    "        },\n",
    "        index=[0],\n",
    "    )\n",
    "\n",
    "    return df_perf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "8LXyI50s4jar"
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 21;\n",
       "                var nbb_unformatted_code = \"def confusion_matrix_sklearn(model, predictors, target):\\n    \\\"\\\"\\\"\\n    To plot the confusion_matrix with percentages\\n\\n    model: classifier\\n    predictors: independent variables\\n    target: dependent variable\\n    \\\"\\\"\\\"\\n    y_pred = model.predict(predictors)\\n    cm = confusion_matrix(target, y_pred)\\n    labels = np.asarray(\\n        [\\n            [\\\"{0:0.0f}\\\".format(item) + \\\"\\\\n{0:.2%}\\\".format(item / cm.flatten().sum())]\\n            for item in cm.flatten()\\n        ]\\n    ).reshape(2, 2)\\n\\n    plt.figure(figsize=(6, 4))\\n    sns.heatmap(cm, annot=labels, fmt=\\\"\\\")\\n    plt.ylabel(\\\"True label\\\")\\n    plt.xlabel(\\\"Predicted label\\\")\";\n",
       "                var nbb_formatted_code = \"def confusion_matrix_sklearn(model, predictors, target):\\n    \\\"\\\"\\\"\\n    To plot the confusion_matrix with percentages\\n\\n    model: classifier\\n    predictors: independent variables\\n    target: dependent variable\\n    \\\"\\\"\\\"\\n    y_pred = model.predict(predictors)\\n    cm = confusion_matrix(target, y_pred)\\n    labels = np.asarray(\\n        [\\n            [\\\"{0:0.0f}\\\".format(item) + \\\"\\\\n{0:.2%}\\\".format(item / cm.flatten().sum())]\\n            for item in cm.flatten()\\n        ]\\n    ).reshape(2, 2)\\n\\n    plt.figure(figsize=(6, 4))\\n    sns.heatmap(cm, annot=labels, fmt=\\\"\\\")\\n    plt.ylabel(\\\"True label\\\")\\n    plt.xlabel(\\\"Predicted label\\\")\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def confusion_matrix_sklearn(model, predictors, target):\n",
    "    \"\"\"\n",
    "    To plot the confusion_matrix with percentages\n",
    "\n",
    "    model: classifier\n",
    "    predictors: independent variables\n",
    "    target: dependent variable\n",
    "    \"\"\"\n",
    "    y_pred = model.predict(predictors)\n",
    "    cm = confusion_matrix(target, y_pred)\n",
    "    labels = np.asarray(\n",
    "        [\n",
    "            [\"{0:0.0f}\".format(item) + \"\\n{0:.2%}\".format(item / cm.flatten().sum())]\n",
    "            for item in cm.flatten()\n",
    "        ]\n",
    "    ).reshape(2, 2)\n",
    "\n",
    "    plt.figure(figsize=(6, 4))\n",
    "    sns.heatmap(cm, annot=labels, fmt=\"\")\n",
    "    plt.ylabel(\"True label\")\n",
    "    plt.xlabel(\"Predicted label\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Rxw_gopM4jar"
   },
   "source": [
    "###  Defining scorer to be used for hyperparameter tuning\n",
    "\n",
    "- Every prediction of a classification model will be either a TP, FP, FN or TN\n",
    "- For this classification problem, we need to reduce the maintenance cost, which can be reiterated as:\n",
    "  - Maximize (minimum possible maintenance cost/maintenance cost)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "id": "X09SzkBA4jas"
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 22;\n",
       "                var nbb_unformatted_code = \"# defining metric to be used for optimization and with cross-validation\\ndef Minimum_Vs_Model_cost(y_train, y_pred):\\n    \\\"\\\"\\\"\\n    We want the model to optimize the maintenance cost and reduce it to the lowest possible value.\\n    The lowest possible maintenance cost will be achieved when each sample is predicted correctly.\\n\\n    In such a scenario, the maintenance cost will be the total number of failures times the maintenance cost of replacing one generator,\\n    which is given by (TP + FN) * 40 (i.e., the actual positives*40).\\n    For any other scenario,\\n    the maintenance cost associated with the model will be given by (TP * 15 + FP * 5 + FN * 40).\\n\\n    We will use the ratio of these two maintenance costs as the cost function for our model.\\n    The greater the ratio, the lower the associated maintenance cost and the better the model.\\n    \\\"\\\"\\\"\\n    TP = confusion_matrix(y_train, y_pred)[1, 1]\\n    FP = confusion_matrix(y_train, y_pred)[0, 1]\\n    FN = confusion_matrix(y_train, y_pred)[1, 0]\\n    return ((TP + FN) * 15) / (TP * 15 + FP * 5 + FN * 40)\\n\\n\\n# A value of .80 here, will represent that the minimum maintenance cost is 80% of the maintenance cost associated with the model.\\n# Since minimum maintenance cost is constant for any data, when minimum cost will become 100% of maintenance cost associated with the model\\n# Model will have give the least possible maintenance cost.\\n\\n\\n# Type of scoring used to compare parameter combinations\\nscorer = metrics.make_scorer(Minimum_Vs_Model_cost, greater_is_better=True)\\n\\n# Higher the values, the lower the maintenance cost\";\n",
       "                var nbb_formatted_code = \"# defining metric to be used for optimization and with cross-validation\\ndef Minimum_Vs_Model_cost(y_train, y_pred):\\n    \\\"\\\"\\\"\\n    We want the model to optimize the maintenance cost and reduce it to the lowest possible value.\\n    The lowest possible maintenance cost will be achieved when each sample is predicted correctly.\\n\\n    In such a scenario, the maintenance cost will be the total number of failures times the maintenance cost of replacing one generator,\\n    which is given by (TP + FN) * 40 (i.e., the actual positives*40).\\n    For any other scenario,\\n    the maintenance cost associated with the model will be given by (TP * 15 + FP * 5 + FN * 40).\\n\\n    We will use the ratio of these two maintenance costs as the cost function for our model.\\n    The greater the ratio, the lower the associated maintenance cost and the better the model.\\n    \\\"\\\"\\\"\\n    TP = confusion_matrix(y_train, y_pred)[1, 1]\\n    FP = confusion_matrix(y_train, y_pred)[0, 1]\\n    FN = confusion_matrix(y_train, y_pred)[1, 0]\\n    return ((TP + FN) * 15) / (TP * 15 + FP * 5 + FN * 40)\\n\\n\\n# A value of .80 here, will represent that the minimum maintenance cost is 80% of the maintenance cost associated with the model.\\n# Since minimum maintenance cost is constant for any data, when minimum cost will become 100% of maintenance cost associated with the model\\n# Model will have give the least possible maintenance cost.\\n\\n\\n# Type of scoring used to compare parameter combinations\\nscorer = metrics.make_scorer(Minimum_Vs_Model_cost, greater_is_better=True)\\n\\n# Higher the values, the lower the maintenance cost\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# defining metric to be used for optimization and with cross-validation\n",
    "def Minimum_Vs_Model_cost(y_train, y_pred):\n",
    "    \"\"\"\n",
    "    We want the model to optimize the maintenance cost and reduce it to the lowest possible value.\n",
    "    The lowest possible maintenance cost will be achieved when each sample is predicted correctly.\n",
    "\n",
    "    In such a scenario, the maintenance cost will be the total number of failures times the maintenance cost of replacing one generator,\n",
    "    which is given by (TP + FN) * 40 (i.e., the actual positives*40).\n",
    "    For any other scenario,\n",
    "    the maintenance cost associated with the model will be given by (TP * 15 + FP * 5 + FN * 40).\n",
    "\n",
    "    We will use the ratio of these two maintenance costs as the cost function for our model.\n",
    "    The greater the ratio, the lower the associated maintenance cost and the better the model.\n",
    "    \"\"\"\n",
    "    TP = confusion_matrix(y_train, y_pred)[1, 1]\n",
    "    FP = confusion_matrix(y_train, y_pred)[0, 1]\n",
    "    FN = confusion_matrix(y_train, y_pred)[1, 0]\n",
    "    return ((TP + FN) * 15) / (TP * 15 + FP * 5 + FN * 40)\n",
    "\n",
    "\n",
    "# A value of .80 here, will represent that the minimum maintenance cost is 80% of the maintenance cost associated with the model.\n",
    "# Since minimum maintenance cost is constant for any data, when minimum cost will become 100% of maintenance cost associated with the model\n",
    "# Model will have give the least possible maintenance cost.\n",
    "\n",
    "\n",
    "# Type of scoring used to compare parameter combinations\n",
    "scorer = metrics.make_scorer(Minimum_Vs_Model_cost, greater_is_better=True)\n",
    "\n",
    "# Higher the values, the lower the maintenance cost"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "eqCDCbcw4jas"
   },
   "source": [
    "## Model Building with Original Training data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "id": "V-tpzI7g4jas"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Cross-Validation Performance:\n",
      "\n",
      "Logistic Regression: 0.5315403616048175\n",
      "dtree: 0.6560666496504416\n",
      "Random forest: 0.7109742970474746\n",
      "Bagging: 0.6812807591389467\n",
      "Adaboost: 0.6023360315420371\n",
      "GBM: 0.6782623334842277\n",
      "Xgboost: 0.7713551730843292\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 23;\n",
       "                var nbb_unformatted_code = \"models = []  # Empty list to store all the models\\n\\n# Appending models into the list\\n\\nmodels.append(\\n    (\\\"Logistic Regression\\\", LogisticRegression(solver=\\\"newton-cg\\\", random_state=1))\\n)\\nmodels.append((\\\"dtree\\\", DecisionTreeClassifier(random_state=1)))\\nmodels.append((\\\"Random forest\\\", RandomForestClassifier(random_state=1)))\\nmodels.append((\\\"Bagging\\\", BaggingClassifier(random_state=1)))\\nmodels.append((\\\"Adaboost\\\", AdaBoostClassifier(random_state=1)))\\nmodels.append((\\\"GBM\\\", GradientBoostingClassifier(random_state=1)))\\nmodels.append((\\\"Xgboost\\\", XGBClassifier(random_state=1, eval_metric=\\\"logloss\\\")))\\n\\nresults = []  # Empty list to store all model's CV scores\\nnames = []  # Empty list to store name of the models\\nscore = []\\n\\n# loop through all models to get the mean cross validated score\\n\\nprint(\\\"\\\\n\\\" \\\"Cross-Validation Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models:\\n    kfold = StratifiedKFold(\\n        n_splits=5, shuffle=True, random_state=1\\n    )  # Setting number of splits equal to 5\\n    cv_result = cross_val_score(\\n        estimator=model, X=X_train, y=y_train, scoring=scorer, cv=kfold\\n    )\\n    results.append(cv_result)\\n    names.append(name)\\n    print(\\\"{}: {}\\\".format(name, cv_result.mean()))\";\n",
       "                var nbb_formatted_code = \"models = []  # Empty list to store all the models\\n\\n# Appending models into the list\\n\\nmodels.append(\\n    (\\\"Logistic Regression\\\", LogisticRegression(solver=\\\"newton-cg\\\", random_state=1))\\n)\\nmodels.append((\\\"dtree\\\", DecisionTreeClassifier(random_state=1)))\\nmodels.append((\\\"Random forest\\\", RandomForestClassifier(random_state=1)))\\nmodels.append((\\\"Bagging\\\", BaggingClassifier(random_state=1)))\\nmodels.append((\\\"Adaboost\\\", AdaBoostClassifier(random_state=1)))\\nmodels.append((\\\"GBM\\\", GradientBoostingClassifier(random_state=1)))\\nmodels.append((\\\"Xgboost\\\", XGBClassifier(random_state=1, eval_metric=\\\"logloss\\\")))\\n\\nresults = []  # Empty list to store all model's CV scores\\nnames = []  # Empty list to store name of the models\\nscore = []\\n\\n# loop through all models to get the mean cross validated score\\n\\nprint(\\\"\\\\n\\\" \\\"Cross-Validation Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models:\\n    kfold = StratifiedKFold(\\n        n_splits=5, shuffle=True, random_state=1\\n    )  # Setting number of splits equal to 5\\n    cv_result = cross_val_score(\\n        estimator=model, X=X_train, y=y_train, scoring=scorer, cv=kfold\\n    )\\n    results.append(cv_result)\\n    names.append(name)\\n    print(\\\"{}: {}\\\".format(name, cv_result.mean()))\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "models = []  # Empty list to store all the models\n",
    "\n",
    "# Appending models into the list\n",
    "\n",
    "models.append(\n",
    "    (\"Logistic Regression\", LogisticRegression(solver=\"newton-cg\", random_state=1))\n",
    ")\n",
    "models.append((\"dtree\", DecisionTreeClassifier(random_state=1)))\n",
    "models.append((\"Random forest\", RandomForestClassifier(random_state=1)))\n",
    "models.append((\"Bagging\", BaggingClassifier(random_state=1)))\n",
    "models.append((\"Adaboost\", AdaBoostClassifier(random_state=1)))\n",
    "models.append((\"GBM\", GradientBoostingClassifier(random_state=1)))\n",
    "models.append((\"Xgboost\", XGBClassifier(random_state=1, eval_metric=\"logloss\")))\n",
    "\n",
    "results = []  # Empty list to store all model's CV scores\n",
    "names = []  # Empty list to store name of the models\n",
    "score = []\n",
    "\n",
    "# loop through all models to get the mean cross validated score\n",
    "\n",
    "print(\"\\n\" \"Cross-Validation Performance:\" \"\\n\")\n",
    "\n",
    "for name, model in models:\n",
    "    kfold = StratifiedKFold(\n",
    "        n_splits=5, shuffle=True, random_state=1\n",
    "    )  # Setting number of splits equal to 5\n",
    "    cv_result = cross_val_score(\n",
    "        estimator=model, X=X_train, y=y_train, scoring=scorer, cv=kfold\n",
    "    )\n",
    "    results.append(cv_result)\n",
    "    names.append(name)\n",
    "    print(\"{}: {}\".format(name, cv_result.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Training Performance:\n",
      "\n",
      "Logistic Regression: 0.5319050376375217\n",
      "dtree: 1.0\n",
      "Random forest: 0.9989125706829056\n",
      "Bagging: 0.933916226108174\n",
      "Adaboost: 0.6077808654227869\n",
      "GBM: 0.7310202132739138\n",
      "Xgboost: 1.0\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 24;\n",
       "                var nbb_unformatted_code = \"print(\\\"\\\\n\\\" \\\"Training Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models:\\n    model.fit(X_train, y_train)\\n    scores = Minimum_Vs_Model_cost(y_train, model.predict(X_train))\\n    print(\\\"{}: {}\\\".format(name, scores))\";\n",
       "                var nbb_formatted_code = \"print(\\\"\\\\n\\\" \\\"Training Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models:\\n    model.fit(X_train, y_train)\\n    scores = Minimum_Vs_Model_cost(y_train, model.predict(X_train))\\n    print(\\\"{}: {}\\\".format(name, scores))\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"\\n\" \"Training Performance:\" \"\\n\")\n",
    "\n",
    "for name, model in models:\n",
    "    model.fit(X_train, y_train)\n",
    "    scores = Minimum_Vs_Model_cost(y_train, model.predict(X_train))\n",
    "    print(\"{}: {}\".format(name, scores))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Validation Performance:\n",
      "\n",
      "Logistic Regression: 0.5191242416249011\n",
      "dtree: 0.6584141853462696\n",
      "Random forest: 0.7158966897053474\n",
      "Bagging: 0.6809688581314879\n",
      "Adaboost: 0.5885167464114832\n",
      "GBM: 0.6742034943473793\n",
      "Xgboost: 0.7708578143360753\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 65;\n",
       "                var nbb_unformatted_code = \"print(\\\"\\\\n\\\" \\\"Validation Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models:\\n    model.fit(X_train, y_train)\\n    scores = Minimum_Vs_Model_cost(y_val, model.predict(X_val))\\n    print(\\\"{}: {}\\\".format(name, scores))\";\n",
       "                var nbb_formatted_code = \"print(\\\"\\\\n\\\" \\\"Validation Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models:\\n    model.fit(X_train, y_train)\\n    scores = Minimum_Vs_Model_cost(y_val, model.predict(X_val))\\n    print(\\\"{}: {}\\\".format(name, scores))\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"\\n\" \"Validation Performance:\" \"\\n\")\n",
    "\n",
    "for name, model in models:\n",
    "    model.fit(X_train, y_train)\n",
    "    scores = Minimum_Vs_Model_cost(y_val, model.predict(X_val))\n",
    "    print(\"{}: {}\".format(name, scores))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The cross validation training performance scores (customized metric) are similar to the validation perfromance score. This indicates that the default algorithms on original dataset are able to generalize well\n",
    "- There is a tendency for some models (decision tree, random forest, bagging and XGBoost) to overfit the training set; as the training performance score (customized metric) approaches 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 25;\n",
       "                var nbb_unformatted_code = \"# Plotting boxplots for CV scores of all models defined above\\n\\nfig = plt.figure(figsize=(10, 4))\\n\\nfig.suptitle(\\\"Algorithm Comparison\\\")\\nax = fig.add_subplot(111)\\n\\nplt.boxplot(results)\\nax.set_xticklabels(names)\\n\\nplt.show()\";\n",
       "                var nbb_formatted_code = \"# Plotting boxplots for CV scores of all models defined above\\n\\nfig = plt.figure(figsize=(10, 4))\\n\\nfig.suptitle(\\\"Algorithm Comparison\\\")\\nax = fig.add_subplot(111)\\n\\nplt.boxplot(results)\\nax.set_xticklabels(names)\\n\\nplt.show()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plotting boxplots for CV scores of all models defined above\n",
    "\n",
    "fig = plt.figure(figsize=(10, 4))\n",
    "\n",
    "fig.suptitle(\"Algorithm Comparison\")\n",
    "ax = fig.add_subplot(111)\n",
    "\n",
    "plt.boxplot(results)\n",
    "ax.set_xticklabels(names)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- XGBoost (~ 0.77) and Random Forest (~ 0.71) have the best average (& median) training cross validation scores (on the customized metric). This is closely followed by Bagging Classifier (~ 0.68)\n",
    "- XGBoost and AdaBoost each have one outlier as can be observed from the boxplot\n",
    "- The boxplot widths (spread of CV scores) is small for XGBoost, Random Forest and Bagging Classifier as well, indicating these are reliable models to choose for further optimization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "oBKJaFU24jas"
   },
   "source": [
    "## Model Building with Oversampled Training data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Before Oversampling, counts of label '1 (Failures)': 1531\n",
      "Before Oversampling, counts of label '0 (No failures)': 26469 \n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 26;\n",
       "                var nbb_unformatted_code = \"print(\\n    \\\"Before Oversampling, counts of label '1 (Failures)': {}\\\".format(sum(y_train == 1))\\n)\\nprint(\\n    \\\"Before Oversampling, counts of label '0 (No failures)': {} \\\\n\\\".format(\\n        sum(y_train == 0)\\n    )\\n)\";\n",
       "                var nbb_formatted_code = \"print(\\n    \\\"Before Oversampling, counts of label '1 (Failures)': {}\\\".format(sum(y_train == 1))\\n)\\nprint(\\n    \\\"Before Oversampling, counts of label '0 (No failures)': {} \\\\n\\\".format(\\n        sum(y_train == 0)\\n    )\\n)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\n",
    "    \"Before Oversampling, counts of label '1 (Failures)': {}\".format(sum(y_train == 1))\n",
    ")\n",
    "print(\n",
    "    \"Before Oversampling, counts of label '0 (No failures)': {} \\n\".format(\n",
    "        sum(y_train == 0)\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "id": "FKxnygkE4jat"
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 27;\n",
       "                var nbb_unformatted_code = \"# Synthetic Minority Over Sampling Technique\\nsm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)\\nX_train_over, y_train_over = sm.fit_resample(X_train, y_train)\";\n",
       "                var nbb_formatted_code = \"# Synthetic Minority Over Sampling Technique\\nsm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)\\nX_train_over, y_train_over = sm.fit_resample(X_train, y_train)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Synthetic Minority Over Sampling Technique\n",
    "sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)\n",
    "X_train_over, y_train_over = sm.fit_resample(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "id": "uYDlbnUO4jat"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "After Oversampling, counts of label '1 (Failures)': 26469\n",
      "After Oversampling, counts of label '0 (No failures)': 26469 \n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 28;\n",
       "                var nbb_unformatted_code = \"print(\\n    \\\"After Oversampling, counts of label '1 (Failures)': {}\\\".format(\\n        sum(y_train_over == 1)\\n    )\\n)\\nprint(\\n    \\\"After Oversampling, counts of label '0 (No failures)': {} \\\\n\\\".format(\\n        sum(y_train_over == 0)\\n    )\\n)\";\n",
       "                var nbb_formatted_code = \"print(\\n    \\\"After Oversampling, counts of label '1 (Failures)': {}\\\".format(\\n        sum(y_train_over == 1)\\n    )\\n)\\nprint(\\n    \\\"After Oversampling, counts of label '0 (No failures)': {} \\\\n\\\".format(\\n        sum(y_train_over == 0)\\n    )\\n)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\n",
    "    \"After Oversampling, counts of label '1 (Failures)': {}\".format(\n",
    "        sum(y_train_over == 1)\n",
    "    )\n",
    ")\n",
    "print(\n",
    "    \"After Oversampling, counts of label '0 (No failures)': {} \\n\".format(\n",
    "        sum(y_train_over == 0)\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- To handle class imbalance in \"Target\" attribute, synthetic minority oversampling technique was employed to generate synthetic data points for minority class of importance (i.e, class \"1\" or No failures)\n",
    "- After applying SMOTE, we have equal number of class \"1\" and calss \"0\" target outcomes "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Cross-Validation Performance:\n",
      "\n",
      "Logistic Regression: 0.7991885856657728\n",
      "dtree: 0.935799411923789\n",
      "Random forest: 0.9684099076356956\n",
      "Bagging: 0.9567710595379042\n",
      "Adaboost: 0.8302735716914681\n",
      "GBM: 0.8698213334032019\n",
      "Xgboost: 0.9735494688964318\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 29;\n",
       "                var nbb_unformatted_code = \"models_over = []  # Empty list to store all the models\\n\\n# Appending models into the list\\n\\nmodels_over.append(\\n    (\\\"Logistic Regression\\\", LogisticRegression(solver=\\\"newton-cg\\\", random_state=1))\\n)\\nmodels_over.append((\\\"dtree\\\", DecisionTreeClassifier(random_state=1)))\\nmodels_over.append((\\\"Random forest\\\", RandomForestClassifier(random_state=1)))\\nmodels_over.append((\\\"Bagging\\\", BaggingClassifier(random_state=1)))\\nmodels_over.append((\\\"Adaboost\\\", AdaBoostClassifier(random_state=1)))\\nmodels_over.append((\\\"GBM\\\", GradientBoostingClassifier(random_state=1)))\\nmodels_over.append((\\\"Xgboost\\\", XGBClassifier(random_state=1, eval_metric=\\\"logloss\\\")))\\n\\nresults_over = []  # Empty list to store all model's CV scores\\nnames_over = []  # Empty list to store name of the models\\nscore_over = []\\n\\n# loop through all models to get the mean cross validated score\\n\\nprint(\\\"\\\\n\\\" \\\"Cross-Validation Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models_over:\\n    kfold = StratifiedKFold(\\n        n_splits=5, shuffle=True, random_state=1\\n    )  # Setting number of splits equal to 5\\n    cv_result = cross_val_score(\\n        estimator=model, X=X_train_over, y=y_train_over, scoring=scorer, cv=kfold\\n    )\\n    results_over.append(cv_result)\\n    names_over.append(name)\\n    print(\\\"{}: {}\\\".format(name, cv_result.mean()))\";\n",
       "                var nbb_formatted_code = \"models_over = []  # Empty list to store all the models\\n\\n# Appending models into the list\\n\\nmodels_over.append(\\n    (\\\"Logistic Regression\\\", LogisticRegression(solver=\\\"newton-cg\\\", random_state=1))\\n)\\nmodels_over.append((\\\"dtree\\\", DecisionTreeClassifier(random_state=1)))\\nmodels_over.append((\\\"Random forest\\\", RandomForestClassifier(random_state=1)))\\nmodels_over.append((\\\"Bagging\\\", BaggingClassifier(random_state=1)))\\nmodels_over.append((\\\"Adaboost\\\", AdaBoostClassifier(random_state=1)))\\nmodels_over.append((\\\"GBM\\\", GradientBoostingClassifier(random_state=1)))\\nmodels_over.append((\\\"Xgboost\\\", XGBClassifier(random_state=1, eval_metric=\\\"logloss\\\")))\\n\\nresults_over = []  # Empty list to store all model's CV scores\\nnames_over = []  # Empty list to store name of the models\\nscore_over = []\\n\\n# loop through all models to get the mean cross validated score\\n\\nprint(\\\"\\\\n\\\" \\\"Cross-Validation Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models_over:\\n    kfold = StratifiedKFold(\\n        n_splits=5, shuffle=True, random_state=1\\n    )  # Setting number of splits equal to 5\\n    cv_result = cross_val_score(\\n        estimator=model, X=X_train_over, y=y_train_over, scoring=scorer, cv=kfold\\n    )\\n    results_over.append(cv_result)\\n    names_over.append(name)\\n    print(\\\"{}: {}\\\".format(name, cv_result.mean()))\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "models_over = []  # Empty list to store all the models\n",
    "\n",
    "# Appending models into the list\n",
    "\n",
    "models_over.append(\n",
    "    (\"Logistic Regression\", LogisticRegression(solver=\"newton-cg\", random_state=1))\n",
    ")\n",
    "models_over.append((\"dtree\", DecisionTreeClassifier(random_state=1)))\n",
    "models_over.append((\"Random forest\", RandomForestClassifier(random_state=1)))\n",
    "models_over.append((\"Bagging\", BaggingClassifier(random_state=1)))\n",
    "models_over.append((\"Adaboost\", AdaBoostClassifier(random_state=1)))\n",
    "models_over.append((\"GBM\", GradientBoostingClassifier(random_state=1)))\n",
    "models_over.append((\"Xgboost\", XGBClassifier(random_state=1, eval_metric=\"logloss\")))\n",
    "\n",
    "results_over = []  # Empty list to store all model's CV scores\n",
    "names_over = []  # Empty list to store name of the models\n",
    "score_over = []\n",
    "\n",
    "# loop through all models to get the mean cross validated score\n",
    "\n",
    "print(\"\\n\" \"Cross-Validation Performance:\" \"\\n\")\n",
    "\n",
    "for name, model in models_over:\n",
    "    kfold = StratifiedKFold(\n",
    "        n_splits=5, shuffle=True, random_state=1\n",
    "    )  # Setting number of splits equal to 5\n",
    "    cv_result = cross_val_score(\n",
    "        estimator=model, X=X_train_over, y=y_train_over, scoring=scorer, cv=kfold\n",
    "    )\n",
    "    results_over.append(cv_result)\n",
    "    names_over.append(name)\n",
    "    print(\"{}: {}\".format(name, cv_result.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Training Performance:\n",
      "\n",
      "Logistic Regression: 0.7997723770483548\n",
      "dtree: 1.0\n",
      "Random forest: 1.0\n",
      "Bagging: 0.9976255088195387\n",
      "Adaboost: 0.8345542254779346\n",
      "GBM: 0.8726427535276275\n",
      "Xgboost: 0.9979264062735635\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 30;\n",
       "                var nbb_unformatted_code = \"print(\\\"\\\\n\\\" \\\"Training Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models_over:\\n    model.fit(X_train_over, y_train_over)\\n    scores = Minimum_Vs_Model_cost(y_train_over, model.predict(X_train_over))\\n    print(\\\"{}: {}\\\".format(name, scores))\";\n",
       "                var nbb_formatted_code = \"print(\\\"\\\\n\\\" \\\"Training Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models_over:\\n    model.fit(X_train_over, y_train_over)\\n    scores = Minimum_Vs_Model_cost(y_train_over, model.predict(X_train_over))\\n    print(\\\"{}: {}\\\".format(name, scores))\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"\\n\" \"Training Performance:\" \"\\n\")\n",
    "\n",
    "for name, model in models_over:\n",
    "    model.fit(X_train_over, y_train_over)\n",
    "    scores = Minimum_Vs_Model_cost(y_train_over, model.predict(X_train_over))\n",
    "    print(\"{}: {}\".format(name, scores))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Validation Performance:\n",
      "\n",
      "Logistic Regression: 0.5025536261491318\n",
      "dtree: 0.6377187297472456\n",
      "Random forest: 0.802937576499388\n",
      "Bagging: 0.7633824670287044\n",
      "Adaboost: 0.5696092619392186\n",
      "GBM: 0.7294292068198666\n",
      "Xgboost: 0.8065573770491803\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 66;\n",
       "                var nbb_unformatted_code = \"print(\\\"\\\\n\\\" \\\"Validation Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models_over:\\n    model.fit(X_train_over, y_train_over)\\n    scores = Minimum_Vs_Model_cost(y_val, model.predict(X_val))\\n    print(\\\"{}: {}\\\".format(name, scores))\";\n",
       "                var nbb_formatted_code = \"print(\\\"\\\\n\\\" \\\"Validation Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models_over:\\n    model.fit(X_train_over, y_train_over)\\n    scores = Minimum_Vs_Model_cost(y_val, model.predict(X_val))\\n    print(\\\"{}: {}\\\".format(name, scores))\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"\\n\" \"Validation Performance:\" \"\\n\")\n",
    "\n",
    "for name, model in models_over:\n",
    "    model.fit(X_train_over, y_train_over)\n",
    "    scores = Minimum_Vs_Model_cost(y_val, model.predict(X_val))\n",
    "    print(\"{}: {}\".format(name, scores))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The cross validation training performance scores (customized metric) are much higher than validation perfromance score. This indicates that the default algorithms on oversampled dataset are not able to generalize well\n",
    "- It is likely that the algorithms are overfitting the noise in the training sets which explains the trends in the observed performance scores (cross validation training scores ~ training score >> validation score). This will be a concern taking these models to production"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 31;\n",
       "                var nbb_unformatted_code = \"# Plotting boxplots for CV scores of all models defined above\\n\\nfig = plt.figure(figsize=(10, 4))\\n\\nfig.suptitle(\\\"Algorithm Comparison\\\")\\nax = fig.add_subplot(111)\\n\\nplt.boxplot(results_over)\\nax.set_xticklabels(names)\\n\\nplt.show()\";\n",
       "                var nbb_formatted_code = \"# Plotting boxplots for CV scores of all models defined above\\n\\nfig = plt.figure(figsize=(10, 4))\\n\\nfig.suptitle(\\\"Algorithm Comparison\\\")\\nax = fig.add_subplot(111)\\n\\nplt.boxplot(results_over)\\nax.set_xticklabels(names)\\n\\nplt.show()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plotting boxplots for CV scores of all models defined above\n",
    "\n",
    "fig = plt.figure(figsize=(10, 4))\n",
    "\n",
    "fig.suptitle(\"Algorithm Comparison\")\n",
    "ax = fig.add_subplot(111)\n",
    "\n",
    "plt.boxplot(results_over)\n",
    "ax.set_xticklabels(names)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The average (& median) training cross validation scores on oversampled dataset has increased to match training performance scores across algorithms. This indicates potential overfitting of noise in the training datasets"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1aimb6bn4jat"
   },
   "source": [
    "## Model Building with Undersampled data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "id": "DhxfTkvu4jat"
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 32;\n",
       "                var nbb_unformatted_code = \"# Random undersampler for under sampling the data\\nrus = RandomUnderSampler(random_state=1, sampling_strategy=1)\\nX_train_un, y_train_un = rus.fit_resample(X_train, y_train)\";\n",
       "                var nbb_formatted_code = \"# Random undersampler for under sampling the data\\nrus = RandomUnderSampler(random_state=1, sampling_strategy=1)\\nX_train_un, y_train_un = rus.fit_resample(X_train, y_train)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Random undersampler for under sampling the data\n",
    "rus = RandomUnderSampler(random_state=1, sampling_strategy=1)\n",
    "X_train_un, y_train_un = rus.fit_resample(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "id": "jROP_DVF4jau"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "After Undersampling, counts of label '1 (Failures)': 1531\n",
      "After Undersampling, counts of label '0 (No failures)': 1531 \n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 33;\n",
       "                var nbb_unformatted_code = \"print(\\n    \\\"After Undersampling, counts of label '1 (Failures)': {}\\\".format(\\n        sum(y_train_un == 1)\\n    )\\n)\\nprint(\\n    \\\"After Undersampling, counts of label '0 (No failures)': {} \\\\n\\\".format(\\n        sum(y_train_un == 0)\\n    )\\n)\";\n",
       "                var nbb_formatted_code = \"print(\\n    \\\"After Undersampling, counts of label '1 (Failures)': {}\\\".format(\\n        sum(y_train_un == 1)\\n    )\\n)\\nprint(\\n    \\\"After Undersampling, counts of label '0 (No failures)': {} \\\\n\\\".format(\\n        sum(y_train_un == 0)\\n    )\\n)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\n",
    "    \"After Undersampling, counts of label '1 (Failures)': {}\".format(\n",
    "        sum(y_train_un == 1)\n",
    "    )\n",
    ")\n",
    "print(\n",
    "    \"After Undersampling, counts of label '0 (No failures)': {} \\n\".format(\n",
    "        sum(y_train_un == 0)\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Another technique to handle class imbalance in \"Target\" attribute is random undersampling, wherein only random samples from the majority class are chosen for model building. While this helps in dealing with models potentially overfitting, it can often lead to poor performing models due to \"loss of information\" from not considering all datapoints available \n",
    "- After random undersampling, we again have equal number of class \"1\" and class \"0\" (and overall less number of datapoints for model building)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Cross-Validation Performance:\n",
      "\n",
      "Logistic Regression: 0.7724863000214184\n",
      "dtree: 0.7623582706765808\n",
      "Random forest: 0.842602547619812\n",
      "Bagging: 0.810537034879969\n",
      "Adaboost: 0.7875970384722428\n",
      "GBM: 0.8260845219126203\n",
      "Xgboost: 0.8405650028470488\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 34;\n",
       "                var nbb_unformatted_code = \"models_un = []  # Empty list to store all the models\\n\\n# Appending models into the list\\n\\nmodels_un.append(\\n    (\\\"Logistic Regression\\\", LogisticRegression(solver=\\\"newton-cg\\\", random_state=1))\\n)\\nmodels_un.append((\\\"dtree\\\", DecisionTreeClassifier(random_state=1)))\\nmodels_un.append((\\\"Random forest\\\", RandomForestClassifier(random_state=1)))\\nmodels_un.append((\\\"Bagging\\\", BaggingClassifier(random_state=1)))\\nmodels_un.append((\\\"Adaboost\\\", AdaBoostClassifier(random_state=1)))\\nmodels_un.append((\\\"GBM\\\", GradientBoostingClassifier(random_state=1)))\\nmodels_un.append((\\\"Xgboost\\\", XGBClassifier(random_state=1, eval_metric=\\\"logloss\\\")))\\n\\nresults_un = []  # Empty list to store all model's CV scores\\nnames_un = []  # Empty list to store name of the models\\nscore_un = []\\n\\n# loop through all models to get the mean cross validated score\\n\\nprint(\\\"\\\\n\\\" \\\"Cross-Validation Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models_un:\\n    kfold = StratifiedKFold(\\n        n_splits=5, shuffle=True, random_state=1\\n    )  # Setting number of splits equal to 5\\n    cv_result = cross_val_score(\\n        estimator=model, X=X_train_un, y=y_train_un, scoring=scorer, cv=kfold\\n    )\\n    results_un.append(cv_result)\\n    names_un.append(name)\\n    print(\\\"{}: {}\\\".format(name, cv_result.mean()))\";\n",
       "                var nbb_formatted_code = \"models_un = []  # Empty list to store all the models\\n\\n# Appending models into the list\\n\\nmodels_un.append(\\n    (\\\"Logistic Regression\\\", LogisticRegression(solver=\\\"newton-cg\\\", random_state=1))\\n)\\nmodels_un.append((\\\"dtree\\\", DecisionTreeClassifier(random_state=1)))\\nmodels_un.append((\\\"Random forest\\\", RandomForestClassifier(random_state=1)))\\nmodels_un.append((\\\"Bagging\\\", BaggingClassifier(random_state=1)))\\nmodels_un.append((\\\"Adaboost\\\", AdaBoostClassifier(random_state=1)))\\nmodels_un.append((\\\"GBM\\\", GradientBoostingClassifier(random_state=1)))\\nmodels_un.append((\\\"Xgboost\\\", XGBClassifier(random_state=1, eval_metric=\\\"logloss\\\")))\\n\\nresults_un = []  # Empty list to store all model's CV scores\\nnames_un = []  # Empty list to store name of the models\\nscore_un = []\\n\\n# loop through all models to get the mean cross validated score\\n\\nprint(\\\"\\\\n\\\" \\\"Cross-Validation Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models_un:\\n    kfold = StratifiedKFold(\\n        n_splits=5, shuffle=True, random_state=1\\n    )  # Setting number of splits equal to 5\\n    cv_result = cross_val_score(\\n        estimator=model, X=X_train_un, y=y_train_un, scoring=scorer, cv=kfold\\n    )\\n    results_un.append(cv_result)\\n    names_un.append(name)\\n    print(\\\"{}: {}\\\".format(name, cv_result.mean()))\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "models_un = []  # Empty list to store all the models\n",
    "\n",
    "# Appending models into the list\n",
    "\n",
    "models_un.append(\n",
    "    (\"Logistic Regression\", LogisticRegression(solver=\"newton-cg\", random_state=1))\n",
    ")\n",
    "models_un.append((\"dtree\", DecisionTreeClassifier(random_state=1)))\n",
    "models_un.append((\"Random forest\", RandomForestClassifier(random_state=1)))\n",
    "models_un.append((\"Bagging\", BaggingClassifier(random_state=1)))\n",
    "models_un.append((\"Adaboost\", AdaBoostClassifier(random_state=1)))\n",
    "models_un.append((\"GBM\", GradientBoostingClassifier(random_state=1)))\n",
    "models_un.append((\"Xgboost\", XGBClassifier(random_state=1, eval_metric=\"logloss\")))\n",
    "\n",
    "results_un = []  # Empty list to store all model's CV scores\n",
    "names_un = []  # Empty list to store name of the models\n",
    "score_un = []\n",
    "\n",
    "# loop through all models to get the mean cross validated score\n",
    "\n",
    "print(\"\\n\" \"Cross-Validation Performance:\" \"\\n\")\n",
    "\n",
    "for name, model in models_un:\n",
    "    kfold = StratifiedKFold(\n",
    "        n_splits=5, shuffle=True, random_state=1\n",
    "    )  # Setting number of splits equal to 5\n",
    "    cv_result = cross_val_score(\n",
    "        estimator=model, X=X_train_un, y=y_train_un, scoring=scorer, cv=kfold\n",
    "    )\n",
    "    results_un.append(cv_result)\n",
    "    names_un.append(name)\n",
    "    print(\"{}: {}\".format(name, cv_result.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Training Performance:\n",
      "\n",
      "Logistic Regression: 0.772322179250042\n",
      "dtree: 1.0\n",
      "Random forest: 1.0\n",
      "Bagging: 0.9624895222129086\n",
      "Adaboost: 0.8197394253078708\n",
      "GBM: 0.8726961808854266\n",
      "Xgboost: 1.0\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 35;\n",
       "                var nbb_unformatted_code = \"print(\\\"\\\\n\\\" \\\"Training Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models_un:\\n    model.fit(X_train_un, y_train_un)\\n    scores = Minimum_Vs_Model_cost(y_train_un, model.predict(X_train_un))\\n    print(\\\"{}: {}\\\".format(name, scores))\";\n",
       "                var nbb_formatted_code = \"print(\\\"\\\\n\\\" \\\"Training Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models_un:\\n    model.fit(X_train_un, y_train_un)\\n    scores = Minimum_Vs_Model_cost(y_train_un, model.predict(X_train_un))\\n    print(\\\"{}: {}\\\".format(name, scores))\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"\\n\" \"Training Performance:\" \"\\n\")\n",
    "\n",
    "for name, model in models_un:\n",
    "    model.fit(X_train_un, y_train_un)\n",
    "    scores = Minimum_Vs_Model_cost(y_train_un, model.predict(X_train_un))\n",
    "    print(\"{}: {}\".format(name, scores))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Validation Performance:\n",
      "\n",
      "Logistic Regression: 0.4932330827067669\n",
      "dtree: 0.4738743077293523\n",
      "Random forest: 0.7348767737117252\n",
      "Bagging: 0.6691601496089765\n",
      "Adaboost: 0.5512605042016807\n",
      "GBM: 0.6762886597938145\n",
      "Xgboost: 0.7440453686200378\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 67;\n",
       "                var nbb_unformatted_code = \"print(\\\"\\\\n\\\" \\\"Validation Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models_un:\\n    model.fit(X_train_un, y_train_un)\\n    scores = Minimum_Vs_Model_cost(y_val, model.predict(X_val))\\n    print(\\\"{}: {}\\\".format(name, scores))\";\n",
       "                var nbb_formatted_code = \"print(\\\"\\\\n\\\" \\\"Validation Performance:\\\" \\\"\\\\n\\\")\\n\\nfor name, model in models_un:\\n    model.fit(X_train_un, y_train_un)\\n    scores = Minimum_Vs_Model_cost(y_val, model.predict(X_val))\\n    print(\\\"{}: {}\\\".format(name, scores))\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"\\n\" \"Validation Performance:\" \"\\n\")\n",
    "\n",
    "for name, model in models_un:\n",
    "    model.fit(X_train_un, y_train_un)\n",
    "    scores = Minimum_Vs_Model_cost(y_val, model.predict(X_val))\n",
    "    print(\"{}: {}\".format(name, scores))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The performance score (customized metric) have dropped on the validation undersampled dataset than original dataset. This could be likely that the algorithms are overfitting the noise & underfitting the information in the undersampled datasets. This will again be a concern taking these models to production"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAlkAAAEVCAYAAADTivDNAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAqGElEQVR4nO3dfXhcZZ3/8ffHUKgILalkUaACrgjpRqkaUaQKFR/qA7L8dIUsqLBxWbyg4LOwcaXoxmUXnwFlWQKoYPABVHB1wXUjGERsCgVaIm63CFRAg608SSUt398f5047nc4k02ZO5iT5vK4rV865z33OfM+ZM2e+c9/3nFFEYGZmZmb19YxGB2BmZmY2FTnJMjMzM8uBkywzMzOzHDjJMjMzM8uBkywzMzOzHDjJMjMzM8uBkyyzKUzSZZL+OadtHyfp+lGWHy5pTR6PPdlJ+kdJFzc6DjPLl5MssylA0k8lrZO000Q9ZkRcERFvKIkhJL1goh5fmdMkrZD0hKQ1kr4t6UUTFcP2iohPR8R7Gx2HmeXLSZbZJCdpX+DVQABvm6DH3GEiHmcMXwROB04D5gAvBL4HvKWBMY2pIMfOzCaAkyyzye/dwC+Ay4D3jFZR0kclPSjpAUnvLW19kjRb0tckDUm6V9LHJT0jLTtB0k2SPi9pLbAklfWn5Temh7hd0uOSjil5zA9J+n163BNLyi+T9GVJP0rr3CTpOZK+kFrlfiXpJVX2Y3/gFKAjIv4nIv4cEX9KrWvnbOP+/FHSakmvSuX3p3jfUxbrhZJ+LOkxSTdI2qdk+RfTeo9KWibp1SXLlkj6jqTLJT0KnJDKLk/LZ6Zlf0ixLJW0R1q2p6RrJK2VtErS35dt91tpHx+TtFJS+2jPv5lNLCdZZpPfu4Er0t8bR96gy0laBHwQeB3wAuCwsirnAbOB56dl7wZOLFn+CmA18BdAd+mKEfGaNHlQROwSEd9M889J29wL6AQukNRcsuo7gY8DuwN/Bm4Gbk3z3wE+V2WfjwDWRMQvqyyvdX/uAJ4NfAO4Eng52bE5Hjhf0i4l9Y8DPpViW052vEcsBeaTtah9A/i2pJkly49K+7Nb2XqQJcazgbkplpOBJ9OyXmANsCfwDuDTko4oWfdtKe7dgGuA86sfDjObaE6yzCYxSQuAfYBvRcQy4P+Av61S/Z3ApRGxMiL+BJxdsp0m4BjgzIh4LCJ+A3wWeFfJ+g9ExHkRsSEinqQ2w8AnI2I4In4IPA4cULL8uxGxLCLWA98F1kfE1yJiI/BNoGJLFlky8mC1B61xf+6JiEtLHmtuivXPEXE98BRZwjXiPyPixoj4M9AFHCJpLkBEXB4Rf0jH5rPATmX7eXNEfC8inq5w7IbT/rwgIjam4/Fo2vYC4GMRsT4ilgMXl+1Df0T8MO3D14GDqh0TM5t4TrLMJrf3ANdHxMNp/htU7zLcE7i/ZL50endgR+DekrJ7yVqgKtWv1R8iYkPJ/J+A0tah35VMP1lhvrTuFtsFnjvK49ayP+WPRUSM9vib9j8iHgfWkh3TkS7RQUmPSPojWcvU7pXWreDrwHXAlakb998kzUjbXhsRj42yDw+VTP8JmOkxX2bF4STLbJKS9Eyy1qnDJD0k6SHgA8BBkiq1aDwI7F0yP7dk+mGyFpV9SsqeB/y2ZD7qEnh9/ATYe5QxSLXsz7badLxSN+Ic4IE0/upjZM9Fc0TsBjwCqGTdqscutfKdHRHzgFcBbyXr2nwAmCNp1zrug5lNICdZZpPXXwMbgXlk44HmA63Az8jepMt9CzhRUquknYFPjCxI3U3fArol7ZoGdX8QuHwb4vkd2fin3EXE/wJfBnqV3Y9rxzSA/FhJZ9Rpf8q9WdICSTuSjc26JSLuB3YFNgBDwA6SPgHMqnWjkhZKelHq4nyULDncmLb9c+Bf0r69mGxcW/mYLjMrKCdZZpPXe8jGWN0XEQ+N/JENfj6uvNsoIn4EfAnoA1aRDTKHbMA5wGLgCbLB7f1kXY+XbEM8S4Cvpm/IvXM792lbnEa2rxcAfyQbj3Y0cG1aPt79KfcN4CyybsKXkQ2Eh6yr70fAr8m689azbV2rzyEbFP8oMAjcwOZksAPYl6xV67vAWRHx43Hsg5lNIEUUqQfAzCaKpFZgBbBT2bgpKyPpMrJvM3680bGY2eThliyzaUTS0alrrRn4V+BaJ1hmZvlwkmU2vfwD2dih/yMbz/W+xoZjZjZ1ubvQzMzMLAduyTIzMzPLgZMsMzMzsxw4yTIzMzPLgZMsMzMzsxw4yTIzMzPLgZMsMzMzsxw4yTIzMzPLgZMsMzMzsxw4yTIzMzPLgZMsMzMzsxw4yTIzMzPLgZMsMzMzsxw4yTIzMzPLgZMsMzMzsxzs0OgAKtl9991j3333bXQYZmZmZmNatmzZwxHRUl5eyCRr3333ZWBgoNFhmJmZmY1J0r2Vyt1daGZmZpYDJ1lmZmZmOXCSZWZmZpYDJ1lmZmZmOagpyZK0SNLdklZJOqPC8tmSrpV0u6SVkk4sWbabpO9I+pWkQUmH1HMHzMzMzIpozCRLUhNwAfAmYB7QIWleWbVTgLsi4iDgcOCzknZMy74I/FdEHAgcBAzWKXazaaO3t5e2tjaamppoa2ujt7e30SGZmdkYarmFw8HAqohYDSDpSuAo4K6SOgHsKknALsBaYIOkWcBrgBMAIuIp4Km6RW82DfT29tLV1UVPTw8LFiygv7+fzs5OADo6OhocnZmZVVNLd+FewP0l82tSWanzgVbgAeBO4PSIeBp4PjAEXCrpNkkXS3rW+MM2mz66u7vp6elh4cKFzJgxg4ULF9LT00N3d3ejQzMzs1HUkmSpQlmUzb8RWA7sCcwHzk+tWDsALwW+EhEvAZ4AthrTBSDpJEkDkgaGhoZqi95sGhgcHGTBggVblC1YsIDBQfe8m5kVWS1J1hpgbsn83mQtVqVOBK6OzCrgHuDAtO6aiLgl1fsOWdK1lYi4KCLaI6K9pWWrO9ObTVutra309/dvUdbf309ra2uDIjIzs1rUkmQtBfaXtF8azH4scE1ZnfuAIwAk7QEcAKyOiIeA+yUdkOodwZZjucxsDF1dXXR2dtLX18fw8DB9fX10dnbS1dXV6NDMzGwUYw58j4gNkk4FrgOagEsiYqWkk9PyC4FPAZdJupOse/FjEfFw2sRi4IqUoK0ma/UysxqNDG5fvHgxg4ODtLa20t3d7UHvZmYFp4jy4VWN197eHv6BaDMzM5sMJC2LiPbyct/x3czMzCwHTrLMzMzMcuAky8zMzCwHTrLMzMzMclDLz+qYmZmZ1V32a3z1VaQv9DnJMjMzs4aoNSGSVKjkqVZOsszMzKyu5syZw7p16+q6zXq2ejU3N7N27dq6ba8aJ1lmZmZWV+vWrSt0y1Me3ZSVeOC7mZmZWQ7ckmVmZradpvrAbRsfJ1lmZmbbaaoP3LbxcXehmZmZWQ6cZJmZmZnlwEmWmZmZWQ6cZJmZmZnlwAPfzXKUxw356mmibshnZjYdOckyy5FvyGdmNn25u9DMzMwsB27JMstRnDULlsxudBhVxVmzGh2CmdmUVVOSJWkR8EWgCbg4Is4pWz4buBx4XtrmZyLi0pLlTcAA8NuIeGudYjcrPJ39aKNDGFVzczNrlzQ6CjOzqWnMJCslSBcArwfWAEslXRMRd5VUOwW4KyKOlNQC3C3pioh4Ki0/HRgE/LHZppUij8cyM7N81TIm62BgVUSsTknTlcBRZXUC2FXZKNpdgLXABgBJewNvAS6uW9RmZmZmBVdLkrUXcH/J/JpUVup8oBV4ALgTOD0ink7LvgB8FHiaUUg6SdKApIGhoaEawjIzMzMrrlqSrErf8S7vA3kjsBzYE5gPnC9plqS3Ar+PiGVjPUhEXBQR7RHR3tLSUkNYZmZmZsVVS5K1BphbMr83WYtVqROBqyOzCrgHOBA4FHibpN+QdTO+VtLl447azMzMrOBq+XbhUmB/SfsBvwWOBf62rM59wBHAzyTtARwArI6IM4EzASQdDnw4Io6vT+hmZmZWRL59TWbMJCsiNkg6FbiO7BYOl0TESkknp+UXAp8CLpN0J1n34sci4uEc4zYzM8tNHj+JVc9fWCj6T2Lp7EcL/e1qScSSCXicIh6E9vb2GBgYaHQYZmY2TUkqfpLg+LZbveOTtCwi2svL/bM6ZmZmZjnwz+qYmZlZ3RX5B+ibm5sn5HGcZJmZmVld1bursOjdj9W4u9DMzMwsB06yzMzMzHLgJMvMzMwsB06yzMzMzHLgJMvMzMwsB06yzMzMzHLgJMvMzMwsB06yzMzMzHLgJMvMzMwsB77ju5mZWZk4axYsmd3oMKqKs2Y1OgSrgZMsMzOzMjr70UL/jIskYkmjo7CxuLvQbBLo7e2lra2NpqYm2tra6O3tbXRIZmbjJqmmv22tWxRuyTIruN7eXrq6uujp6WHBggX09/fT2dkJQEdHR4OjMzPbfkVuLawHFXEH29vbY2BgoNFhmBVCW1sb5513HgsXLtxU1tfXx+LFi1mxYkUDIyuOPD69FvHaaBNHUqHPgaLHN91IWhYR7VuVF/FJcpJltllTUxPr169nxowZm8qGh4eZOXMmGzdubGBkk4/fmKxWRT9Xih7fdFMtyappTJakRZLulrRK0hkVls+WdK2k2yWtlHRiKp8rqU/SYCo/ffy7Yja9tLa20t/fv0VZf38/ra2tDYrIzMxqMWaSJakJuAB4EzAP6JA0r6zaKcBdEXEQcDjwWUk7AhuAD0VEK/BK4JQK65rZKLq6uujs7KSvr4/h4WH6+vro7Oykq6ur0aGZmdkoahn4fjCwKiJWA0i6EjgKuKukTgC7KhsYsQuwFtgQEQ8CDwJExGOSBoG9ytY1s1GMDG5fvHgxg4ODtLa20t3d7UHvZmYFV0uStRdwf8n8GuAVZXXOB64BHgB2BY6JiKdLK0jaF3gJcEulB5F0EnASwPOe97wawjKbPjo6OpxUmZlNMrWMyar0tZ3y0XZvBJYDewLzgfMlbbodraRdgKuA90fEo5UeJCIuioj2iGhvaWmpISwzMzOz4qolyVoDzC2Z35usxarUicDVkVkF3AMcCCBpBlmCdUVEXD3+kM3MzMyKr5Ykaymwv6T90mD2Y8m6BkvdBxwBIGkP4ABgdRqj1QMMRsTn6he2mZmZWbGNmWRFxAbgVOA6YBD4VkSslHSypJNTtU8Br5J0J/AT4GMR8TBwKPAu4LWSlqe/N+eyJ2ZmZmYFUtPP6kTED4EflpVdWDL9APCGCuv1U3lMl5nZmObMmcO6devqus163h2+ubmZtWvX1m17Zja1+LcLzayw1q1bV+i7Whftx2jNrFhquuO7mZmZmW0bt2TVqLe3l+7u7k03g+zq6vJ9i8xyFmfNgiWzGx1GVXHWrLErmdm05SSrBr29vXR1ddHT08OCBQvo7++ns7MTwImWWY509qOF7y6MJY2OwsyKyt2FNeju7qanp4eFCxcyY8YMFi5cSE9PD93d3Y0OzczMzApKRfyU2N7eHgMDA40OY5OmpibWr1/PjBkzNpUNDw8zc+ZMNm7c2MDIzKY2ScVvySpwfLb9iv7cFj2+6UbSsohoLy93S1YNWltb6e/v36Ksv7+f1tbWBkVkZmZmReckqwZdXV10dnbS19fH8PAwfX19dHZ20tXV1ejQzKY8SYX9a25ubvThMbMC88D3GowMbl+8ePGmbxd2d3d70LtZztwdYmaTmcdkmZmZlSn6mKeixzfdVBuT5ZYsMzOzCop8R393VU8OTrLMzMzK1NpKlEci5haqqcNJlpmZ2XZyQmSj8bcLzczMzHLgJMvMzMwsB06yzMzMzHLgJMvMzMwsB06yzMzMzHJQU5IlaZGkuyWtknRGheWzJV0r6XZJKyWdWOu6ZmZmZlPRmLdwkNQEXAC8HlgDLJV0TUTcVVLtFOCuiDhSUgtwt6QrgI01rGtmZg3i+zyZ5aeWlqyDgVURsToingKuBI4qqxPArsperbsAa4ENNa5rZmYNEhE1/W1rXTOrLcnaC7i/ZH5NKit1PtAKPADcCZweEU/XuK6ZmZnZlFNLklWpLbn8o8obgeXAnsB84HxJs2pcN3sQ6SRJA5IGhoaGagjLzMzMrLhqSbLWAHNL5vcma7EqdSJwdWRWAfcAB9a4LgARcVFEtEdEe0tLS63xm5mZmRVSLUnWUmB/SftJ2hE4FrimrM59wBEAkvYADgBW17iumZmZ2ZQz5rcLI2KDpFOB64Am4JKIWCnp5LT8QuBTwGWS7iTrIvxYRDwMUGndfHbFzMxGzJkzh3Xr1tV1m/X8JmJzczNr166t2/bMikhF/CZIe3t7DAwMNDoMM7NJS1Khv+lX9PjMtoWkZRHRXl4+ZkuWWS18rx0zM7MtOcmyuqg1IfKnVzMzmy7824VmZmZmOXCSZWZmZpYDJ1lmZmZmOXCSZWZmZpYDJ1lmZmZmOXCSZWZmZpYDJ1lmZmZmOXCSZWZmZpYDJ1lmNuX19vbS1tZGU1MTbW1t9Pb2NjokM5sGfMd3M5vSent76erqoqenhwULFtDf309nZycAHR0dDY7OzKYy/0C0TSj/rI5NtLa2Ns477zwWLly4qayvr4/FixezYsWKBkaWsyWzGx3B2JY80ugIzOqi2g9EO8myCeUkyyZaU1MT69evZ8aMGZvKhoeHmTlzJhs3bmxgZPkq+mut6PGZbYtqSZbHZJnZlNba2kp/f/8WZf39/bS2tjYoIjObLpxkmdmU1tXVRWdnJ319fQwPD9PX10dnZyddXV2NDs3MpjgPfDezKW1kcPvixYsZHByktbWV7u5uD3o3s9x5TJZNKI/DMJsYRX+tFT0+s21RbUyWW7JsVHPmzGHdunV13aakum2rubmZtWvX1m17ZlNJPV9r9dbc3NzoEMxyV1OSJWkR8EWgCbg4Is4pW/4R4LiSbbYCLRGxVtIHgPcCAdwJnBgR6+sUv+Vs3bp1hf60WeQ3EbNGqvfr1i1PZttuzIHvkpqAC4A3AfOADknzSutExLkRMT8i5gNnAjekBGsv4DSgPSLayJK0Y+u8D2ZmZmaFU8u3Cw8GVkXE6oh4CrgSOGqU+h1A6W9W7AA8U9IOwM7AA9sbrJmZmdlkUUuStRdwf8n8mlS2FUk7A4uAqwAi4rfAZ4D7gAeBRyLi+irrniRpQNLA0NBQ7XtQgzlz5iCpkH9z5syp676amZlZMdSSZFUa9FKtY/5I4KaIWAsgqZms1Ws/YE/gWZKOr7RiRFwUEe0R0d7S0lJDWLUbGVdUxL96Dyo3MzOzYqglyVoDzC2Z35vqXX7HsmVX4euAeyJiKCKGgauBV21PoGZmZmaTSS1J1lJgf0n7SdqRLJG6prySpNnAYcD3S4rvA14paWdlXwM7Ahgcf9hmZmZmxTbmLRwiYoOkU4HryL4deElErJR0clp+Yap6NHB9RDxRsu4tkr4D3ApsAG4DLqrzPpiZmZkVzrS443uR7+9S5NjA8ZlZxq81s+qm9R3f46xZsGR2o8OoKM6a1egQzMzMLAfTIsnS2Y8W9hOYJGJJo6MwMzOzeqtl4LuZmZmZbSMnWWZmZmY5cJJlZmZmlgMnWWZmZmY5cJJlZmZmlgMnWWZmZmY5cJJlZmZmlgMnWWZmZmY5mBY3IzUzs8ok1b1uUW/+bDbRpk2StS0XkonU3Nzc6BDMbBpzQmSWn2mRZPkiYmZmZhPNY7LMzMzMcuAky8zMzCwH06K70LZfnDULlsxudBhVxVmzGh2CmZlZRU6ybFQ6+9FCj2mTRCxpdBRmZmZbc3ehmZmZWQ5qSrIkLZJ0t6RVks6osPwjkpanvxWSNkqak5btJuk7kn4laVDSIfXeCTMzM7OiGTPJktQEXAC8CZgHdEiaV1onIs6NiPkRMR84E7ghItamxV8E/isiDgQOAgbrGL+ZmZlZIdXSknUwsCoiVkfEU8CVwFGj1O8AegEkzQJeA/QARMRTEfHHcUVsZmZmNgnUkmTtBdxfMr8mlW1F0s7AIuCqVPR8YAi4VNJtki6W9Kwq654kaUDSwNDQUM07YGZmZlZEtSRZlX6PptrXzY4EbirpKtwBeCnwlYh4CfAEsNWYLoCIuCgi2iOivaWlpYawzMzMzIqrliRrDTC3ZH5v4IEqdY8ldRWWrLsmIm5J898hS7rMzMzMprRakqylwP6S9pO0I1kidU15JUmzgcOA74+URcRDwP2SDkhFRwB3jTtqMzMzs4Ib82akEbFB0qnAdUATcElErJR0clp+Yap6NHB9RDxRtonFwBUpQVsNnFi36M3MzMwKSkW8m3d7e3sMDAw0Ogwj3VG9gOfIiKLHZ2ZmU5+kZRHRXl7uO76bmZmZ5cBJlpmZmVkOnGSZmZmZ5WDMge9mUqVbpRVDc3Nzo0MwMzOryEmWjareg8o9UN3MzKYLdxeamZmZ5cBJlpmZmVkOnGSZmZmZ5cBJlpmZmVkOnGSZmZmZ5cBJlpmZmVkOnGSZmZmZ5cBJlpmZmVkOnGSZmZmZ5cBJlpmZmVkOnGSZmZmZ5cBJlpmZmVkOnGSZmZmZ5aCmJEvSIkl3S1ol6YwKyz8iaXn6WyFpo6Q5JcubJN0m6Qf1DN7MzMysqMZMsiQ1ARcAbwLmAR2S5pXWiYhzI2J+RMwHzgRuiIi1JVVOBwbrFrWZmZlZwdXSknUwsCoiVkfEU8CVwFGj1O8AekdmJO0NvAW4eDyBmpmZmU0mtSRZewH3l8yvSWVbkbQzsAi4qqT4C8BHgae3L0QzMzOzyaeWJEsVyqJK3SOBm0a6CiW9Ffh9RCwb80GkkyQNSBoYGhqqISwzMzOz4qolyVoDzC2Z3xt4oErdYynpKgQOBd4m6Tdk3YyvlXR5pRUj4qKIaI+I9paWlhrCMjMzMyuuWpKspcD+kvaTtCNZInVNeSVJs4HDgO+PlEXEmRGxd0Tsm9b7n4g4vi6Rm5mZmRXYDmNViIgNkk4FrgOagEsiYqWkk9PyC1PVo4HrI+KJ3KI1MzMzmyQUUW14VeO0t7fHwMBAo8OwHEiiiOecmZnZ9pK0LCLay8t9x3czMzOzHDjJMjMzM8uBkywzMzOzHDjJMjMzM8uBkywzMzOzHDjJMjMzM8uBkywzMzOzHDjJMjMzM8uBkywzMzOzHDjJqlFvby9tbW00NTXR1tZGb2/v2CuZmZnZtDXmbxdalmB1dXXR09PDggUL6O/vp7OzE4COjo4GR2dmZmZF5JasGnR3d9PT08PChQuZMWMGCxcupKenh+7u7kaHZmZmZgXlH4iuQVNTE+vXr2fGjBmbyoaHh5k5cyYbN25sYGSTj38g2szMphr/QPQ4tLa20t/fv0VZf38/ra2tDYrIzMzMis5JVg26urro7Oykr6+P4eFh+vr66OzspKurq9GhmZmZWUF54HsNRga3L168mMHBQVpbW+nu7vagdzMzM6vKY7JsQnlMlpmZTTUek2VmZmY2gZxkmZmZmeWgpiRL0iJJd0taJemMCss/Iml5+lshaaOkOZLmSuqTNChppaTT678LZmZmZsUzZpIlqQm4AHgTMA/okDSvtE5EnBsR8yNiPnAmcENErAU2AB+KiFbglcAp5euamZmZTUW1tGQdDKyKiNUR8RRwJXDUKPU7gF6AiHgwIm5N048Bg8Be4wvZzMzMrPhqSbL2Au4vmV9DlURJ0s7AIuCqCsv2BV4C3FJl3ZMkDUgaGBoaqiEsMzMzs+KqJclShbJq38E/ErgpdRVu3oC0C1ni9f6IeLTSihFxUUS0R0R7S0tLDWGZmZmZFVctSdYaYG7J/N7AA1XqHkvqKhwhaQZZgnVFRFy9PUGamZmZTTa1JFlLgf0l7SdpR7JE6prySpJmA4cB3y8pE9ADDEbE5+oTspmZmVnxjZlkRcQG4FTgOrKB69+KiJWSTpZ0cknVo4HrI+KJkrJDgXcBry25xcOb6xi/mZmZWSH5Z3VsQvlndczMbKrxz+qYmZmZTaAdGh2ATQ3Z8Lv61nWLl5mZTWZOsqwunBCZmZltyd2FZmZmZjlwkmVmZmaWAydZZmZmZjlwkmVmZmaWAydZZmZmZjlwkmVmZmaWAydZZmZmZjlwkmVmZmaWg0L+dqGkIeDeRsdRxe7Aw40OYhLz8RsfH7/x8fHbfj524+PjNz5FP377RERLeWEhk6wikzRQ6UcgrTY+fuPj4zc+Pn7bz8dufHz8xmeyHj93F5qZmZnlwEmWmZmZWQ6cZG27ixodwCTn4zc+Pn7j4+O3/XzsxsfHb3wm5fHzmCwzMzOzHLgly8zMzCwHDUmyJD1eh220S/rSKMv3lfS3tdavsP5PJd0t6XZJSyXNH2fIdSPpbZLOaHQc5SQtkfRhSSdI2rPR8dSDpI2SlktaIelaSbvVabsnSDq/Htsq2+6rJa1MMT+z3ttPj/GPeWy3ymONHP/bJd0q6VU5PMY2XRsmA0lHSwpJB1ZZ/lNJo35TS9JvJO2eU3zzJb05j21PNEl7SPqGpNWSlkm6OR3/wyU9ks7fOyT9t6S/SOuckJ6fI0q2M/KcvaNxe1M/kuZKukfSnDTfnOb3GWWdKXfOTdqWrIgYiIjTRqmyL7ApyaqhfiXHRcRBwJeBc7c9yq1JahrvNiLimog4px7x5OQEoGKSVY/9n2BPRsT8iGgD1gKnNDqgMRwHfCbF/ORYlbfz+ZiwJIvNx/8g4EzgX+r9ANt5bSi6DqAfOLbRgVQxH5j0SZYkAd8DboyI50fEy8iO+d6pys/S+ftiYClbXj/uJHueRhwL3J5/1BMjIu4HvgKMvFedA1wUEY26B+Z8GnDOFSbJSlnmL1LG/11Jzan85ansZknnSlqRyg+X9IM0fVj6tLBc0m2SdiV7Ql+dyj5QVn8XSZdKujNt++1jhHczsFda91mSLkmtW7dJOiqV7yzpW2l735R0y8gnRUmPS/qkpFuAQyQdL+mXKbZ/l9SU/i5LLSZ3SvpAWvc0SXel7V6Zyja1gkjaR9JP0vKfSHpeKr9M0pck/Tx9wsrl05GkLmUtfv8NHJCK24ErRlpT0qeTT0jqB/5G0hvS83mrpG9L2iVt62WSbkifBq+T9Nw8Yh6H0vPg4HRsb0v/D0jlJ0i6WtJ/SfpfSf82srKkEyX9WtINwKEl5aM9h1+R1Jeew8PSuTco6bLy4CS9F3gn8AlJVyhzbsk5dUyqd3ja5jeAO9O5d246p++Q9A+p3nMl3ajNLXmvlnQO8MxUdkVOx7maWcC6FNsu6VjdmvbtqJLj8E+SfiXpx5J6JX04lddyLVmSjvFP0zE/baztFk16PR0KdJKSrPQ6vDLt/zeBZ5bU/4qkAWUtoGeXbe4jyq5Vv5T0glS/2vlarfxv0vlzezqfdgQ+CRyTzqNjcj8o+Xkt8FREXDhSEBH3RsR5pZUkCdiVdP4mPwMOljQjPWcvAJbnH/KE+jzwSknvBxYAn5X0DElfTufbDyT9UFu+P02tcy4iJvwPeLxC2R3AYWn6k8AX0vQK4FVp+hxgRZo+HPhBmr4WODRN7wLsULq8Qv1/Hdl+mm+uEM9PgfY0/X7g02n608DxaXo34NfAs4APA/+eytuADSXrB/DONN2a4p2R5r8MvBt4GfDjksffLf1/ANiprOwE4PySfX9Pmv474Htp+jLg22SJ9DxgVQ7P48vIPo3tTPYGuCodh03HLtX7DfDRNL07cCPwrDT/MeATwAzg50BLKj8GuKQR52elcxVoSsdzUZqfBeyQpl8HXFXy3KwGZgMzyX65YC7wXOA+oAXYEbipxufwSkDAUcCjwIvSc7oMmF8h3suAd6TptwM/TrHvkR7/uWSvhSeA/VK9k4CPp+mdgAFgP+BDQFfJ/u9a7fWb4/HfSPbG8yvgEeBlqXwHYFbJObUqHaf2VP+ZZG9q/wt8ONWr5VqyJJ2HO6Xt/iGdm1W3W7Q/4HigJ03/HHgp8MGR1xPwYra8Ps0peY5/Cry45HU78vy/my2vt5XO12rldwJ7pemtrmGT+Q84Dfh8lWWHp3N2OXB/OodHztkTgPOBzwFvJWuBPouS1+9U+QPeSPYe+Po0/w7gh2TXseeQJZ4j16wpd84VoiVL0myyA3FDKvoq8Bpl4192jYifp/JvVNnETcDn0qfO3SJiwxgP+TrggpGZiFhXpd4VktaQJQIjn0zeAJwhaTnZBWkm8DyyLP3KtL0VZEnjiI3AVWn6CLLkZGnaxhHA88nemJ8v6TxJi8jeUEnbuULS8WQXxnKHsPm4fD3FMeJ7EfF0RNxF9iZbb68GvhsRf4qIR4FrRqn7zfT/lWRJ301p/98D7EPWCtYG/DiVf5zNTe6N9MwUzx+AOWRJC2RJ1LdTa8jngb8qWecnEfFIRKwH7iLbv1cAP42IoYh4is3HA0Z/Dq+N7ApxJ/C7iLgzIp4GVpJ1iY9mAdAbERsj4nfADcDL07JfRsQ9afoNwLvTft4CPBvYn6x740RJS4AXRcRjYzxeHka6Cw8EFgFfS60CAj4t6Q7gv8laGPcg2+fvR8STKd5rAbbhWgLwnxHx54h4GPj9aNstqA7StSj97wBeA1wOEBF3sOX16Z2SbgVuIzuP55Us6y35f0iarna+Viu/CbhM0t+TJXJTlqQLUuvJ0lQ00l04F7gU+LeyVa4ka208ls3Heqp5E/Ag2fUdsvPi2+m96SGgr6z+lDrndmjkg9dAtVSKiHMk/SdZf+svJL2uhu3Wcu+K48j6yM8hS8r+X1r37RFx9xYbzC781ayPiI0lj/3ViDhzq6Ckg8iy/lPIun3+DngL2QXybcA/Sfqr8vXKlO7Xn0s3P8Z626vWe4A8URLHjyOidCwCkl4ErIyIQ7Zas7GejIj56YPAD8iemy8BnwL6IuJoSfuSJdwjSo/7Rja/zmo9VpWew6fLtvs0Y79+R3vOnyiZFrA4Iq7bagPSa8jOwa9LOjcivjbGY+YmIm5WNii2hey13kLWsjUs6TdkH3iq7fO2nP+Vnr+8Xj91JenZZF1YbZKC7A0myBKorc4/SfuRtT6/PCLWKeuGnllSJapMU2t5RJws6RVk59FyFehLRHWwkqzFGICIOCWdowMV6l7D5g/bI/V/KamN7Drz69HfRiaf9Fy/nuzDdb+yIS9j7eSUOucK0ZIVEY8A6yS9OhW9C7ghtTA9JumVqbziIE5Jf5k+4f8r2cl9IPAYWbN+JdcDp5as3zxKbMNkrSqvlNQKXAcsHkmqJL0kVe0nS4yQNI+sW6eSnwDv0OZvmcxJ/cq7A8+IiKuAfwJeKukZwNyI6AM+StY9uUvZ9n7O5uNyXIpjotwIHK1svMeuwJGpfLRj/wvg0JK+9p0lvRC4G2iRdEgqn1FDQjlh0jl6GvBhSTPIWrJ+mxafUMMmbgEOl/TstP7flCzL6zm8kWwMQpOkFrJk/ZcV6l0HvC/FhaQXKht7uA/w+4j4D6CHrNsJYHik7kRS9k25JrJWxdkptmFJC8laCyE7dkdKmpnGubwFNrVWj3ktGUXF7RbQO4CvRcQ+EbFvakG5B7iV7Nwivam/ONWfRZZwPyJpD7JWh1LHlPy/OU1XO18rlqfr8y0R8QmyH/idy+jXiMnkf4CZkt5XUrZzlboLgP+rUH4mE/tlkgmR3iO/Arw/Iu4j+/LYZ8jOi7ensVl7kHWrlppS51yjWrJ2Tt1wIz5H1m10oaSdybrOTkzLOoH/kPQEWWvBIxW29/50od1I1j3zI7JP+hsk3U7Wz31bSf1/Bi5IXT0bgbOBq6sFGxFPSvos2Se+U4EvAHekk+g3ZH3qXwa+mrovbiNrjt8q1oi4S9LHgetTEjVM1jryJHBpKoPshdcEXJ5aUUTW9//Hsk87pwGXSPoIMFRy3HIXEbcqG0S7nGzs0c/SosvInssn2dzcO7LOkKQTgF5JO6Xij6dPce8AvpT2dwey47wy7/2oVUTcls6nY8ma/b8q6YNkF9qx1n0wdbvdTNZ0fiubm7Hzeg6/S3b8byf7hPfRiHhIW3+t/2Kyrsdb0zk9BPw12cXvI5KGgcfJxkhAduflOyTdGhHH1SnWaka6ayF7DbwnIjYqG3R/raQBNo/ZIiKWSrqGbJ/vJfvQNfI6rOVaUtEY2y2SDjZ/m2vEVcBLyI7lHWTH65cAEXG7pNvIXmerybpZSu2k7As7z2DzN+Gqna/Vys+VtD/Z8/cTsmN4H5uHXfxLRJR2n08aERGS/hr4vKSPku33E2RDTCB9+Yps3x8B3lthGz+amGgn3N8D90XEyBCLL5N9IP09sIZsjOSvyT6Alr6WptQ5V/g7vkvaJSIeT9NnAM+NiNMbHNZWlH0VfkZErJf0l2RP7AvT+BszmyAj14z0ge1G4KT0gWBc15Jq281lJ8ymsJLX0rPJEv5D0/isKafoY7IA3iLpTLJY76W2rplG2BnoS90oAt7nBMusIS5KXfYzycY/jiRC472WVNuumW2bHyj7MsqOwKemaoIFk6Aly8zMzGwyKsTAdzMzM7OpxkmWmZmZWQ6cZJmZmZnlwEmWmZmZWQ6cZJmZmZnlwEmWmZmZWQ7+P1cRuSHaAcg3AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 720x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 68;\n",
       "                var nbb_unformatted_code = \"# Plotting boxplots for CV scores of all models defined above\\n\\nfig = plt.figure(figsize=(10, 4))\\n\\nfig.suptitle(\\\"Algorithm Comparison\\\")\\nax = fig.add_subplot(111)\\n\\nplt.boxplot(results_un)\\nax.set_xticklabels(names)\\n\\nplt.show()\";\n",
       "                var nbb_formatted_code = \"# Plotting boxplots for CV scores of all models defined above\\n\\nfig = plt.figure(figsize=(10, 4))\\n\\nfig.suptitle(\\\"Algorithm Comparison\\\")\\nax = fig.add_subplot(111)\\n\\nplt.boxplot(results_un)\\nax.set_xticklabels(names)\\n\\nplt.show()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plotting boxplots for CV scores of all models defined above\n",
    "\n",
    "fig = plt.figure(figsize=(10, 4))\n",
    "\n",
    "fig.suptitle(\"Algorithm Comparison\")\n",
    "ax = fig.add_subplot(111)\n",
    "\n",
    "plt.boxplot(results_un)\n",
    "ax.set_xticklabels(names)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The algorithms are able to give better performance on the cross validation training scores on undersampled dataset in comparison to original dataset as can be seen from the boxplots. However, the issue is the lack of generalizatbility in carrying forth the performance to the validation set"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "50N658sB4jau"
   },
   "source": [
    "## Model Selection"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "PFFwX4CG4jau"
   },
   "source": [
    "- Models built on original dataset have given generalized performance on cross validation training and validation sets unlike models built on oversampled or undersampled sets\n",
    "- Mean cross validation scores on training sets are highest with XGBoost, Random Forest & Bagging Classifiers (~77, ~71 and ~68% respectively). These models will be tuned further to try to increase performance"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yZGY1eL84jau"
   },
   "source": [
    "## HyperparameterTuning "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best parameters are {'subsample': 0.9, 'scale_pos_weight': 10, 'n_estimators': 250, 'learning_rate': 0.1, 'gamma': 3} with CV score=0.7997478671216852:\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 36;\n",
       "                var nbb_unformatted_code = \"# defining model - XGBoost Hyperparameter Tuning\\nmodel = XGBClassifier(random_state=1, eval_metric=\\\"logloss\\\")\\n\\n# Parameter grid to pass in RandomizedSearchCV\\nparam_grid = {\\n    \\\"n_estimators\\\": np.arange(150, 300, 50),\\n    \\\"scale_pos_weight\\\": [5, 10],\\n    \\\"learning_rate\\\": [0.1, 0.2],\\n    \\\"gamma\\\": [0, 3, 5],\\n    \\\"subsample\\\": [0.8, 0.9],\\n}\\n\\n# Type of scoring used to compare parameter combinations\\nscorer = metrics.make_scorer(Minimum_Vs_Model_cost, greater_is_better=True)\\n\\n# Calling RandomizedSearchCV\\nrandomized_cv = RandomizedSearchCV(\\n    estimator=model,\\n    param_distributions=param_grid,\\n    n_iter=20,\\n    scoring=scorer,\\n    cv=3,\\n    random_state=1,\\n    n_jobs=-1,\\n)\\n\\n# Fitting parameters in RandomizedSearchCV\\nrandomized_cv.fit(X_train, y_train)\\n\\nprint(\\n    \\\"Best parameters are {} with CV score={}:\\\".format(\\n        randomized_cv.best_params_, randomized_cv.best_score_\\n    )\\n)\";\n",
       "                var nbb_formatted_code = \"# defining model - XGBoost Hyperparameter Tuning\\nmodel = XGBClassifier(random_state=1, eval_metric=\\\"logloss\\\")\\n\\n# Parameter grid to pass in RandomizedSearchCV\\nparam_grid = {\\n    \\\"n_estimators\\\": np.arange(150, 300, 50),\\n    \\\"scale_pos_weight\\\": [5, 10],\\n    \\\"learning_rate\\\": [0.1, 0.2],\\n    \\\"gamma\\\": [0, 3, 5],\\n    \\\"subsample\\\": [0.8, 0.9],\\n}\\n\\n# Type of scoring used to compare parameter combinations\\nscorer = metrics.make_scorer(Minimum_Vs_Model_cost, greater_is_better=True)\\n\\n# Calling RandomizedSearchCV\\nrandomized_cv = RandomizedSearchCV(\\n    estimator=model,\\n    param_distributions=param_grid,\\n    n_iter=20,\\n    scoring=scorer,\\n    cv=3,\\n    random_state=1,\\n    n_jobs=-1,\\n)\\n\\n# Fitting parameters in RandomizedSearchCV\\nrandomized_cv.fit(X_train, y_train)\\n\\nprint(\\n    \\\"Best parameters are {} with CV score={}:\\\".format(\\n        randomized_cv.best_params_, randomized_cv.best_score_\\n    )\\n)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# defining model - XGBoost Hyperparameter Tuning\n",
    "model = XGBClassifier(random_state=1, eval_metric=\"logloss\")\n",
    "\n",
    "# Parameter grid to pass in RandomizedSearchCV\n",
    "param_grid = {\n",
    "    \"n_estimators\": np.arange(150, 300, 50),\n",
    "    \"scale_pos_weight\": [5, 10],\n",
    "    \"learning_rate\": [0.1, 0.2],\n",
    "    \"gamma\": [0, 3, 5],\n",
    "    \"subsample\": [0.8, 0.9],\n",
    "}\n",
    "\n",
    "# Type of scoring used to compare parameter combinations\n",
    "scorer = metrics.make_scorer(Minimum_Vs_Model_cost, greater_is_better=True)\n",
    "\n",
    "# Calling RandomizedSearchCV\n",
    "randomized_cv = RandomizedSearchCV(\n",
    "    estimator=model,\n",
    "    param_distributions=param_grid,\n",
    "    n_iter=20,\n",
    "    scoring=scorer,\n",
    "    cv=3,\n",
    "    random_state=1,\n",
    "    n_jobs=-1,\n",
    ")\n",
    "\n",
    "# Fitting parameters in RandomizedSearchCV\n",
    "randomized_cv.fit(X_train, y_train)\n",
    "\n",
    "print(\n",
    "    \"Best parameters are {} with CV score={}:\".format(\n",
    "        randomized_cv.best_params_, randomized_cv.best_score_\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "              colsample_bynode=1, colsample_bytree=1, eval_metric='logloss',\n",
       "              gamma=3, gpu_id=-1, importance_type='gain',\n",
       "              interaction_constraints='', learning_rate=0.1, max_delta_step=0,\n",
       "              max_depth=6, min_child_weight=1, missing=nan,\n",
       "              monotone_constraints='()', n_estimators=250, n_jobs=4,\n",
       "              num_parallel_tree=1, random_state=1, reg_alpha=0, reg_lambda=1,\n",
       "              scale_pos_weight=10, subsample=0.9, tree_method='exact',\n",
       "              validate_parameters=1, verbosity=None)"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 37;\n",
       "                var nbb_unformatted_code = \"# building model with best parameters\\nxgb_tuned = XGBClassifier(\\n    subsample=0.9,\\n    scale_pos_weight=10,\\n    n_estimators=250,\\n    learning_rate=0.1,\\n    gamma=3,\\n    random_state=1,\\n    eval_metric=\\\"logloss\\\")\\n\\n# Fit the model on training data\\nxgb_tuned.fit(X_train, y_train)\";\n",
       "                var nbb_formatted_code = \"# building model with best parameters\\nxgb_tuned = XGBClassifier(\\n    subsample=0.9,\\n    scale_pos_weight=10,\\n    n_estimators=250,\\n    learning_rate=0.1,\\n    gamma=3,\\n    random_state=1,\\n    eval_metric=\\\"logloss\\\",\\n)\\n\\n# Fit the model on training data\\nxgb_tuned.fit(X_train, y_train)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# building model with best parameters\n",
    "xgb_tuned = XGBClassifier(\n",
    "    subsample=0.9,\n",
    "    scale_pos_weight=10,\n",
    "    n_estimators=250,\n",
    "    learning_rate=0.1,\n",
    "    gamma=3,\n",
    "    random_state=1,\n",
    "    eval_metric=\"logloss\",\n",
    ")\n",
    "\n",
    "# Fit the model on training data\n",
    "xgb_tuned.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training performance:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Accuracy</th>\n",
       "      <th>Recall</th>\n",
       "      <th>Precision</th>\n",
       "      <th>F1</th>\n",
       "      <th>Minimum_Vs_Model_cost</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.000</td>\n",
       "      <td>1.000</td>\n",
       "      <td>0.995</td>\n",
       "      <td>0.997</td>\n",
       "      <td>0.998</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Accuracy  Recall  Precision    F1  Minimum_Vs_Model_cost\n",
       "0     1.000   1.000      0.995 0.997                  0.998"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 69;\n",
       "                var nbb_unformatted_code = \"# Calculating different metrics on training set\\nxgboost_random_train = model_performance_classification_sklearn(\\n    xgb_tuned, X_train, y_train\\n)\\nprint(\\\"Training performance:\\\")\\nxgboost_random_train\";\n",
       "                var nbb_formatted_code = \"# Calculating different metrics on training set\\nxgboost_random_train = model_performance_classification_sklearn(\\n    xgb_tuned, X_train, y_train\\n)\\nprint(\\\"Training performance:\\\")\\nxgboost_random_train\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Calculating different metrics on training set\n",
    "xgboost_random_train = model_performance_classification_sklearn(\n",
    "    xgb_tuned, X_train, y_train\n",
    ")\n",
    "print(\"Training performance:\")\n",
    "xgboost_random_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation performance:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Accuracy</th>\n",
       "      <th>Recall</th>\n",
       "      <th>Precision</th>\n",
       "      <th>F1</th>\n",
       "      <th>Minimum_Vs_Model_cost</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.991</td>\n",
       "      <td>0.877</td>\n",
       "      <td>0.962</td>\n",
       "      <td>0.917</td>\n",
       "      <td>0.821</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Accuracy  Recall  Precision    F1  Minimum_Vs_Model_cost\n",
       "0     0.991   0.877      0.962 0.917                  0.821"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 39;\n",
       "                var nbb_unformatted_code = \"# Calculating different metrics on validation set\\nxgboost_random_val = model_performance_classification_sklearn(xgb_tuned, X_val, y_val)\\nprint(\\\"Validation performance:\\\")\\nxgboost_random_val\";\n",
       "                var nbb_formatted_code = \"# Calculating different metrics on validation set\\nxgboost_random_val = model_performance_classification_sklearn(xgb_tuned, X_val, y_val)\\nprint(\\\"Validation performance:\\\")\\nxgboost_random_val\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Calculating different metrics on validation set\n",
    "xgboost_random_val = model_performance_classification_sklearn(xgb_tuned, X_val, y_val)\n",
    "print(\"Validation performance:\")\n",
    "xgboost_random_val"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The best hyperparameters using RandomizedSearch CV for XGBoost model were found to be: subsample 0.9, scale_pos_weight 10, n_estimators 250, learning_rate 0.1 and gamma 3\n",
    "- The average cross validation training performance score (customized metric) using the best parameter XGBoost model is 0.80. This is similar to the performance score (customized metric) on the validation set i.e., 0.82. This indicates the model may generalize with a performance score of ~0.80-0.82\n",
    "- The model does however have a tendency to overfit the training set as can be observed from training performance (customized metric score of 0.998)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "id": "tVZcJ0hv4jau"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best parameters are {'n_estimators': 250, 'min_samples_leaf': 1, 'max_samples': 0.5000000000000001, 'max_features': 'sqrt'} with CV score=0.6920650879825658:\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 40;\n",
       "                var nbb_unformatted_code = \"# defining model - Random Forest Hyperparameter Tuning\\nmodel2 = RandomForestClassifier(random_state=1, oob_score=True, bootstrap=True)\\n\\nparam_grid2 = {\\n    \\\"n_estimators\\\": [150, 250],\\n    \\\"min_samples_leaf\\\": np.arange(1, 3),\\n    \\\"max_features\\\": [\\\"sqrt\\\", \\\"log2\\\"],\\n    \\\"max_samples\\\": np.arange(0.2, 0.6, 0.1),\\n}\\n\\n# Type of scoring used to compare parameter combinations\\nscorer = metrics.make_scorer(Minimum_Vs_Model_cost, greater_is_better=True)\\n\\n# Calling RandomizedSearchCV\\nrandomized_cv2 = RandomizedSearchCV(\\n    estimator=model2,\\n    param_distributions=param_grid2,\\n    n_iter=50,\\n    scoring=scorer,\\n    cv=5,\\n    random_state=1,\\n    n_jobs=-1,\\n)\\n\\n# Fitting parameters in RandomizedSearchCV\\nrandomized_cv2.fit(X_train, y_train)\\nprint(\\n    \\\"Best parameters are {} with CV score={}:\\\".format(\\n        randomized_cv2.best_params_, randomized_cv2.best_score_\\n    )\\n)\";\n",
       "                var nbb_formatted_code = \"# defining model - Random Forest Hyperparameter Tuning\\nmodel2 = RandomForestClassifier(random_state=1, oob_score=True, bootstrap=True)\\n\\nparam_grid2 = {\\n    \\\"n_estimators\\\": [150, 250],\\n    \\\"min_samples_leaf\\\": np.arange(1, 3),\\n    \\\"max_features\\\": [\\\"sqrt\\\", \\\"log2\\\"],\\n    \\\"max_samples\\\": np.arange(0.2, 0.6, 0.1),\\n}\\n\\n# Type of scoring used to compare parameter combinations\\nscorer = metrics.make_scorer(Minimum_Vs_Model_cost, greater_is_better=True)\\n\\n# Calling RandomizedSearchCV\\nrandomized_cv2 = RandomizedSearchCV(\\n    estimator=model2,\\n    param_distributions=param_grid2,\\n    n_iter=50,\\n    scoring=scorer,\\n    cv=5,\\n    random_state=1,\\n    n_jobs=-1,\\n)\\n\\n# Fitting parameters in RandomizedSearchCV\\nrandomized_cv2.fit(X_train, y_train)\\nprint(\\n    \\\"Best parameters are {} with CV score={}:\\\".format(\\n        randomized_cv2.best_params_, randomized_cv2.best_score_\\n    )\\n)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# defining model - Random Forest Hyperparameter Tuning\n",
    "model2 = RandomForestClassifier(random_state=1, oob_score=True, bootstrap=True)\n",
    "\n",
    "param_grid2 = {\n",
    "    \"n_estimators\": [150, 250],\n",
    "    \"min_samples_leaf\": np.arange(1, 3),\n",
    "    \"max_features\": [\"sqrt\", \"log2\"],\n",
    "    \"max_samples\": np.arange(0.2, 0.6, 0.1),\n",
    "}\n",
    "\n",
    "# Type of scoring used to compare parameter combinations\n",
    "scorer = metrics.make_scorer(Minimum_Vs_Model_cost, greater_is_better=True)\n",
    "\n",
    "# Calling RandomizedSearchCV\n",
    "randomized_cv2 = RandomizedSearchCV(\n",
    "    estimator=model2,\n",
    "    param_distributions=param_grid2,\n",
    "    n_iter=50,\n",
    "    scoring=scorer,\n",
    "    cv=5,\n",
    "    random_state=1,\n",
    "    n_jobs=-1,\n",
    ")\n",
    "\n",
    "# Fitting parameters in RandomizedSearchCV\n",
    "randomized_cv2.fit(X_train, y_train)\n",
    "print(\n",
    "    \"Best parameters are {} with CV score={}:\".format(\n",
    "        randomized_cv2.best_params_, randomized_cv2.best_score_\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(max_features='sqrt', max_samples=0.5000000000000001,\n",
       "                       n_estimators=250, random_state=1)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 41;\n",
       "                var nbb_unformatted_code = \"# building model with best parameters\\nrf_tuned = RandomForestClassifier(\\n    n_estimators=250,\\n    min_samples_leaf=1,\\n    max_samples=0.5000000000000001,\\n    max_features=\\\"sqrt\\\",\\n    random_state=1,\\n)\\n\\n# Fit the model on training data\\nrf_tuned.fit(X_train, y_train)\";\n",
       "                var nbb_formatted_code = \"# building model with best parameters\\nrf_tuned = RandomForestClassifier(\\n    n_estimators=250,\\n    min_samples_leaf=1,\\n    max_samples=0.5000000000000001,\\n    max_features=\\\"sqrt\\\",\\n    random_state=1,\\n)\\n\\n# Fit the model on training data\\nrf_tuned.fit(X_train, y_train)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# building model with best parameters\n",
    "rf_tuned = RandomForestClassifier(\n",
    "    n_estimators=250,\n",
    "    min_samples_leaf=1,\n",
    "    max_samples=0.5000000000000001,\n",
    "    max_features=\"sqrt\",\n",
    "    random_state=1,\n",
    ")\n",
    "\n",
    "# Fit the model on training data\n",
    "rf_tuned.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training performance:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Accuracy</th>\n",
       "      <th>Recall</th>\n",
       "      <th>Precision</th>\n",
       "      <th>F1</th>\n",
       "      <th>Minimum_Vs_Model_cost</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.993</td>\n",
       "      <td>0.882</td>\n",
       "      <td>0.998</td>\n",
       "      <td>0.937</td>\n",
       "      <td>0.836</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Accuracy  Recall  Precision    F1  Minimum_Vs_Model_cost\n",
       "0     0.993   0.882      0.998 0.937                  0.836"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 70;\n",
       "                var nbb_unformatted_code = \"# Calculating different metrics on training set\\nrf_random_train = model_performance_classification_sklearn(rf_tuned, X_train, y_train)\\nprint(\\\"Training performance:\\\")\\nrf_random_train\";\n",
       "                var nbb_formatted_code = \"# Calculating different metrics on training set\\nrf_random_train = model_performance_classification_sklearn(rf_tuned, X_train, y_train)\\nprint(\\\"Training performance:\\\")\\nrf_random_train\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Calculating different metrics on training set\n",
    "rf_random_train = model_performance_classification_sklearn(rf_tuned, X_train, y_train)\n",
    "print(\"Training performance:\")\n",
    "rf_random_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation performance:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Accuracy</th>\n",
       "      <th>Recall</th>\n",
       "      <th>Precision</th>\n",
       "      <th>F1</th>\n",
       "      <th>Minimum_Vs_Model_cost</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.985</td>\n",
       "      <td>0.741</td>\n",
       "      <td>0.988</td>\n",
       "      <td>0.847</td>\n",
       "      <td>0.697</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Accuracy  Recall  Precision    F1  Minimum_Vs_Model_cost\n",
       "0     0.985   0.741      0.988 0.847                  0.697"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 44;\n",
       "                var nbb_unformatted_code = \"# Calculating different metrics on validation set\\nrf_random_val = model_performance_classification_sklearn(rf_tuned, X_val, y_val)\\nprint(\\\"Validation performance:\\\")\\nrf_random_val\";\n",
       "                var nbb_formatted_code = \"# Calculating different metrics on validation set\\nrf_random_val = model_performance_classification_sklearn(rf_tuned, X_val, y_val)\\nprint(\\\"Validation performance:\\\")\\nrf_random_val\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Calculating different metrics on validation set\n",
    "rf_random_val = model_performance_classification_sklearn(rf_tuned, X_val, y_val)\n",
    "print(\"Validation performance:\")\n",
    "rf_random_val"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The best hyperparameters using RandomizedSearch CV for Random forest model were found to be: n_estimators 250, min_sample_leaf 1, max_features 'sqrt', max_samples 0.5\n",
    "- The average 5 fold cross validation training performance score (customized metric) using the best parameter Random forest model is 0.692. This is similar to the performance score (customized metric) on the validation set i.e., 0.697. This indicates the model may generalize with a performance score of ~0.69\n",
    "- The model has a slight tendency (although not as much as XGBoost tuned) to overfit the training set as can be observed from training performance (customized metric score of 0.8336)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best parameters are {'n_estimators': 50, 'max_samples': 0.9, 'max_features': 0.8} with CV score=0.7092140237024578:\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 45;\n",
       "                var nbb_unformatted_code = \"# defining model - Bagging Classifier Hyperparameter Tuning\\nmodel3 = BaggingClassifier(random_state=1)\\n\\nparam_grid3 = {\\n    \\\"max_samples\\\": [0.8, 0.9],\\n    \\\"max_features\\\": [0.8, 0.9],\\n    \\\"n_estimators\\\": [40, 50],\\n}\\n\\n# Type of scoring used to compare parameter combinations\\nscorer = metrics.make_scorer(Minimum_Vs_Model_cost, greater_is_better=True)\\n\\n# Calling RandomizedSearchCV\\nrandomized_cv3 = RandomizedSearchCV(\\n    estimator=model3,\\n    param_distributions=param_grid3,\\n    n_iter=50,\\n    scoring=scorer,\\n    cv=5,\\n    random_state=1,\\n    n_jobs=-1,\\n)\\n\\n# Fitting parameters in RandomizedSearchCV\\nrandomized_cv3.fit(X_train, y_train)\\nprint(\\n    \\\"Best parameters are {} with CV score={}:\\\".format(\\n        randomized_cv3.best_params_, randomized_cv3.best_score_\\n    )\\n)\";\n",
       "                var nbb_formatted_code = \"# defining model - Bagging Classifier Hyperparameter Tuning\\nmodel3 = BaggingClassifier(random_state=1)\\n\\nparam_grid3 = {\\n    \\\"max_samples\\\": [0.8, 0.9],\\n    \\\"max_features\\\": [0.8, 0.9],\\n    \\\"n_estimators\\\": [40, 50],\\n}\\n\\n# Type of scoring used to compare parameter combinations\\nscorer = metrics.make_scorer(Minimum_Vs_Model_cost, greater_is_better=True)\\n\\n# Calling RandomizedSearchCV\\nrandomized_cv3 = RandomizedSearchCV(\\n    estimator=model3,\\n    param_distributions=param_grid3,\\n    n_iter=50,\\n    scoring=scorer,\\n    cv=5,\\n    random_state=1,\\n    n_jobs=-1,\\n)\\n\\n# Fitting parameters in RandomizedSearchCV\\nrandomized_cv3.fit(X_train, y_train)\\nprint(\\n    \\\"Best parameters are {} with CV score={}:\\\".format(\\n        randomized_cv3.best_params_, randomized_cv3.best_score_\\n    )\\n)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# defining model - Bagging Classifier Hyperparameter Tuning\n",
    "model3 = BaggingClassifier(random_state=1)\n",
    "\n",
    "param_grid3 = {\n",
    "    \"max_samples\": [0.8, 0.9],\n",
    "    \"max_features\": [0.8, 0.9],\n",
    "    \"n_estimators\": [40, 50],\n",
    "}\n",
    "\n",
    "# Type of scoring used to compare parameter combinations\n",
    "scorer = metrics.make_scorer(Minimum_Vs_Model_cost, greater_is_better=True)\n",
    "\n",
    "# Calling RandomizedSearchCV\n",
    "randomized_cv3 = RandomizedSearchCV(\n",
    "    estimator=model3,\n",
    "    param_distributions=param_grid3,\n",
    "    n_iter=50,\n",
    "    scoring=scorer,\n",
    "    cv=5,\n",
    "    random_state=1,\n",
    "    n_jobs=-1,\n",
    ")\n",
    "\n",
    "# Fitting parameters in RandomizedSearchCV\n",
    "randomized_cv3.fit(X_train, y_train)\n",
    "print(\n",
    "    \"Best parameters are {} with CV score={}:\".format(\n",
    "        randomized_cv3.best_params_, randomized_cv3.best_score_\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "BaggingClassifier(max_features=0.8, max_samples=0.9, n_estimators=50,\n",
       "                  random_state=1)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 48;\n",
       "                var nbb_unformatted_code = \"# building model with best parameters\\nbagging_tuned = BaggingClassifier(\\n    n_estimators=50, max_samples=0.9, max_features=0.8, random_state=1,\\n)\\n\\n# Fit the model on training data\\nbagging_tuned.fit(X_train, y_train)\";\n",
       "                var nbb_formatted_code = \"# building model with best parameters\\nbagging_tuned = BaggingClassifier(\\n    n_estimators=50, max_samples=0.9, max_features=0.8, random_state=1,\\n)\\n\\n# Fit the model on training data\\nbagging_tuned.fit(X_train, y_train)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# building model with best parameters\n",
    "bagging_tuned = BaggingClassifier(\n",
    "    n_estimators=50, max_samples=0.9, max_features=0.8, random_state=1,\n",
    ")\n",
    "\n",
    "# Fit the model on training data\n",
    "bagging_tuned.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training performance:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Accuracy</th>\n",
       "      <th>Recall</th>\n",
       "      <th>Precision</th>\n",
       "      <th>F1</th>\n",
       "      <th>Minimum_Vs_Model_cost</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.999</td>\n",
       "      <td>0.989</td>\n",
       "      <td>1.000</td>\n",
       "      <td>0.994</td>\n",
       "      <td>0.982</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Accuracy  Recall  Precision    F1  Minimum_Vs_Model_cost\n",
       "0     0.999   0.989      1.000 0.994                  0.982"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 49;\n",
       "                var nbb_unformatted_code = \"# Calculating different metrics on train set\\nbagging_random_train = model_performance_classification_sklearn(\\n    bagging_tuned, X_train, y_train\\n)\\nprint(\\\"Training performance:\\\")\\nbagging_random_train\";\n",
       "                var nbb_formatted_code = \"# Calculating different metrics on train set\\nbagging_random_train = model_performance_classification_sklearn(\\n    bagging_tuned, X_train, y_train\\n)\\nprint(\\\"Training performance:\\\")\\nbagging_random_train\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Calculating different metrics on train set\n",
    "bagging_random_train = model_performance_classification_sklearn(\n",
    "    bagging_tuned, X_train, y_train\n",
    ")\n",
    "print(\"Training performance:\")\n",
    "bagging_random_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation performance:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Accuracy</th>\n",
       "      <th>Recall</th>\n",
       "      <th>Precision</th>\n",
       "      <th>F1</th>\n",
       "      <th>Minimum_Vs_Model_cost</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.985</td>\n",
       "      <td>0.745</td>\n",
       "      <td>0.978</td>\n",
       "      <td>0.846</td>\n",
       "      <td>0.699</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Accuracy  Recall  Precision    F1  Minimum_Vs_Model_cost\n",
       "0     0.985   0.745      0.978 0.846                  0.699"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 50;\n",
       "                var nbb_unformatted_code = \"# Calculating different metrics on validation set\\nbagging_random_val = model_performance_classification_sklearn(\\n    bagging_tuned, X_val, y_val\\n)\\nprint(\\\"Validation performance:\\\")\\nbagging_random_val\";\n",
       "                var nbb_formatted_code = \"# Calculating different metrics on validation set\\nbagging_random_val = model_performance_classification_sklearn(\\n    bagging_tuned, X_val, y_val\\n)\\nprint(\\\"Validation performance:\\\")\\nbagging_random_val\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Calculating different metrics on validation set\n",
    "bagging_random_val = model_performance_classification_sklearn(\n",
    "    bagging_tuned, X_val, y_val\n",
    ")\n",
    "print(\"Validation performance:\")\n",
    "bagging_random_val"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The best hyperparameters using RandomizedSearch CV for Bagging Classifier were found to be: n_estimator 50, max_samples 0.9, max_features 0.8\n",
    "- The average 5 fold cross validation training performance score (customized metric) using the best parameter Bagging classifier is 0.71. This is similar to the performance score (customized metric) on the validation set i.e., 0.70. This indicates the model may generalize with a performance score of ~0.69-0.71\n",
    "- The model does however have a tendency to overfit the training set as can be observed from training performance (customized metric score of 0.982)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "D9JNnpxa4jau"
   },
   "source": [
    "## Model Performance comparison and choosing the final model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training performance comparison:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>XGBoost Tuned with Random search</th>\n",
       "      <th>Random forest Tuned with Random search</th>\n",
       "      <th>Bagging Tuned with Random Search</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Accuracy</th>\n",
       "      <td>1.000</td>\n",
       "      <td>0.993</td>\n",
       "      <td>0.999</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Recall</th>\n",
       "      <td>1.000</td>\n",
       "      <td>0.882</td>\n",
       "      <td>0.989</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Precision</th>\n",
       "      <td>0.995</td>\n",
       "      <td>0.998</td>\n",
       "      <td>1.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>F1</th>\n",
       "      <td>0.997</td>\n",
       "      <td>0.937</td>\n",
       "      <td>0.994</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Minimum_Vs_Model_cost</th>\n",
       "      <td>0.998</td>\n",
       "      <td>0.836</td>\n",
       "      <td>0.982</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       XGBoost Tuned with Random search  \\\n",
       "Accuracy                                          1.000   \n",
       "Recall                                            1.000   \n",
       "Precision                                         0.995   \n",
       "F1                                                0.997   \n",
       "Minimum_Vs_Model_cost                             0.998   \n",
       "\n",
       "                       Random forest Tuned with Random search  \\\n",
       "Accuracy                                                0.993   \n",
       "Recall                                                  0.882   \n",
       "Precision                                               0.998   \n",
       "F1                                                      0.937   \n",
       "Minimum_Vs_Model_cost                                   0.836   \n",
       "\n",
       "                       Bagging Tuned with Random Search  \n",
       "Accuracy                                          0.999  \n",
       "Recall                                            0.989  \n",
       "Precision                                         1.000  \n",
       "F1                                                0.994  \n",
       "Minimum_Vs_Model_cost                             0.982  "
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 72;\n",
       "                var nbb_unformatted_code = \"# training performance comparison\\n\\nmodels_train_comp_df = pd.concat(\\n    [xgboost_random_train.T, rf_random_train.T, bagging_random_train.T,], axis=1,\\n)\\nmodels_train_comp_df.columns = [\\n    \\\"XGBoost Tuned with Random search\\\",\\n    \\\"Random forest Tuned with Random search\\\",\\n    \\\"Bagging Tuned with Random Search\\\",\\n]\\nprint(\\\"Training performance comparison:\\\")\\nmodels_train_comp_df\";\n",
       "                var nbb_formatted_code = \"# training performance comparison\\n\\nmodels_train_comp_df = pd.concat(\\n    [xgboost_random_train.T, rf_random_train.T, bagging_random_train.T,], axis=1,\\n)\\nmodels_train_comp_df.columns = [\\n    \\\"XGBoost Tuned with Random search\\\",\\n    \\\"Random forest Tuned with Random search\\\",\\n    \\\"Bagging Tuned with Random Search\\\",\\n]\\nprint(\\\"Training performance comparison:\\\")\\nmodels_train_comp_df\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# training performance comparison\n",
    "\n",
    "models_train_comp_df = pd.concat(\n",
    "    [xgboost_random_train.T, rf_random_train.T, bagging_random_train.T,], axis=1,\n",
    ")\n",
    "models_train_comp_df.columns = [\n",
    "    \"XGBoost Tuned with Random search\",\n",
    "    \"Random forest Tuned with Random search\",\n",
    "    \"Bagging Tuned with Random Search\",\n",
    "]\n",
    "print(\"Training performance comparison:\")\n",
    "models_train_comp_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation performance comparison:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>XGBoost Tuned with Random search</th>\n",
       "      <th>Random forest Tuned with Random search</th>\n",
       "      <th>Bagging Tuned with Random Search</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Accuracy</th>\n",
       "      <td>0.991</td>\n",
       "      <td>0.985</td>\n",
       "      <td>0.985</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Recall</th>\n",
       "      <td>0.877</td>\n",
       "      <td>0.741</td>\n",
       "      <td>0.745</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Precision</th>\n",
       "      <td>0.962</td>\n",
       "      <td>0.988</td>\n",
       "      <td>0.978</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>F1</th>\n",
       "      <td>0.917</td>\n",
       "      <td>0.847</td>\n",
       "      <td>0.846</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Minimum_Vs_Model_cost</th>\n",
       "      <td>0.821</td>\n",
       "      <td>0.697</td>\n",
       "      <td>0.699</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       XGBoost Tuned with Random search  \\\n",
       "Accuracy                                          0.991   \n",
       "Recall                                            0.877   \n",
       "Precision                                         0.962   \n",
       "F1                                                0.917   \n",
       "Minimum_Vs_Model_cost                             0.821   \n",
       "\n",
       "                       Random forest Tuned with Random search  \\\n",
       "Accuracy                                                0.985   \n",
       "Recall                                                  0.741   \n",
       "Precision                                               0.988   \n",
       "F1                                                      0.847   \n",
       "Minimum_Vs_Model_cost                                   0.697   \n",
       "\n",
       "                       Bagging Tuned with Random Search  \n",
       "Accuracy                                          0.985  \n",
       "Recall                                            0.745  \n",
       "Precision                                         0.978  \n",
       "F1                                                0.846  \n",
       "Minimum_Vs_Model_cost                             0.699  "
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 73;\n",
       "                var nbb_unformatted_code = \"# training performance comparison\\n\\nmodels_val_comp_df = pd.concat(\\n    [xgboost_random_val.T, rf_random_val.T, bagging_random_val.T,], axis=1,\\n)\\nmodels_val_comp_df.columns = [\\n    \\\"XGBoost Tuned with Random search\\\",\\n    \\\"Random forest Tuned with Random search\\\",\\n    \\\"Bagging Tuned with Random Search\\\",\\n]\\nprint(\\\"Validation performance comparison:\\\")\\nmodels_val_comp_df\";\n",
       "                var nbb_formatted_code = \"# training performance comparison\\n\\nmodels_val_comp_df = pd.concat(\\n    [xgboost_random_val.T, rf_random_val.T, bagging_random_val.T,], axis=1,\\n)\\nmodels_val_comp_df.columns = [\\n    \\\"XGBoost Tuned with Random search\\\",\\n    \\\"Random forest Tuned with Random search\\\",\\n    \\\"Bagging Tuned with Random Search\\\",\\n]\\nprint(\\\"Validation performance comparison:\\\")\\nmodels_val_comp_df\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# training performance comparison\n",
    "\n",
    "models_val_comp_df = pd.concat(\n",
    "    [xgboost_random_val.T, rf_random_val.T, bagging_random_val.T,], axis=1,\n",
    ")\n",
    "models_val_comp_df.columns = [\n",
    "    \"XGBoost Tuned with Random search\",\n",
    "    \"Random forest Tuned with Random search\",\n",
    "    \"Bagging Tuned with Random Search\",\n",
    "]\n",
    "print(\"Validation performance comparison:\")\n",
    "models_val_comp_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The XGBoost Tuned model with Random Search is giving the highest performance score (Minimum_Vs_Model_cost) of 0.821 on the Validation Set. Although this algorithm is giving much higher performance on training set (0.998) indicating overfitting, we still observe the following -\n",
    "  - The average cross validation Training performance score (Minimum_Vs_Model_cost) with this model is 0.80, similar to the validation score of 0.821\n",
    "  - The accuracy, precision and F1 scores of the training & validation models are very much comparable\n",
    "    \n",
    "    \n",
    "- We will choose this tuned model to see if it can generalize well on the testing dataset to give a likewise high performance score (Minimum_Vs_Model_cost) ~ 0.8"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "d_pDMFAz4jav"
   },
   "source": [
    "## Test set final performance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 51;\n",
       "                var nbb_unformatted_code = \"# Loading the dataset\\ntest = pd.read_csv(\\\"test.csv\\\")\";\n",
       "                var nbb_formatted_code = \"# Loading the dataset\\ntest = pd.read_csv(\\\"test.csv\\\")\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Loading the dataset\n",
    "test = pd.read_csv(\"test.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 52;\n",
       "                var nbb_unformatted_code = \"X_test = test.drop([\\\"Target\\\"], axis=1)\\ny_test = test[\\\"Target\\\"]\";\n",
       "                var nbb_formatted_code = \"X_test = test.drop([\\\"Target\\\"], axis=1)\\ny_test = test[\\\"Target\\\"]\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "X_test = test.drop([\"Target\"], axis=1)\n",
    "y_test = test[\"Target\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    9453\n",
       "1     547\n",
       "Name: Target, dtype: int64"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 53;\n",
       "                var nbb_unformatted_code = \"y_test.value_counts()\";\n",
       "                var nbb_formatted_code = \"y_test.value_counts()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "y_test.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The test data has likewise 94.53% \"0\" or \"No failures\" and 5.47% \"1\" or \"Failures\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 10000 entries, 0 to 9999\n",
      "Data columns (total 40 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   V1      9989 non-null   float64\n",
      " 1   V2      9993 non-null   float64\n",
      " 2   V3      10000 non-null  float64\n",
      " 3   V4      10000 non-null  float64\n",
      " 4   V5      10000 non-null  float64\n",
      " 5   V6      10000 non-null  float64\n",
      " 6   V7      10000 non-null  float64\n",
      " 7   V8      10000 non-null  float64\n",
      " 8   V9      10000 non-null  float64\n",
      " 9   V10     10000 non-null  float64\n",
      " 10  V11     10000 non-null  float64\n",
      " 11  V12     10000 non-null  float64\n",
      " 12  V13     10000 non-null  float64\n",
      " 13  V14     10000 non-null  float64\n",
      " 14  V15     10000 non-null  float64\n",
      " 15  V16     10000 non-null  float64\n",
      " 16  V17     10000 non-null  float64\n",
      " 17  V18     10000 non-null  float64\n",
      " 18  V19     10000 non-null  float64\n",
      " 19  V20     10000 non-null  float64\n",
      " 20  V21     10000 non-null  float64\n",
      " 21  V22     10000 non-null  float64\n",
      " 22  V23     10000 non-null  float64\n",
      " 23  V24     10000 non-null  float64\n",
      " 24  V25     10000 non-null  float64\n",
      " 25  V26     10000 non-null  float64\n",
      " 26  V27     10000 non-null  float64\n",
      " 27  V28     10000 non-null  float64\n",
      " 28  V29     10000 non-null  float64\n",
      " 29  V30     10000 non-null  float64\n",
      " 30  V31     10000 non-null  float64\n",
      " 31  V32     10000 non-null  float64\n",
      " 32  V33     10000 non-null  float64\n",
      " 33  V34     10000 non-null  float64\n",
      " 34  V35     10000 non-null  float64\n",
      " 35  V36     10000 non-null  float64\n",
      " 36  V37     10000 non-null  float64\n",
      " 37  V38     10000 non-null  float64\n",
      " 38  V39     10000 non-null  float64\n",
      " 39  V40     10000 non-null  float64\n",
      "dtypes: float64(40)\n",
      "memory usage: 3.1 MB\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 54;\n",
       "                var nbb_unformatted_code = \"X_test.info()\";\n",
       "                var nbb_formatted_code = \"X_test.info()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "X_test.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- There are 11 & 7 missing values for attributes \"V1\" and \"V2\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 55;\n",
       "                var nbb_unformatted_code = \"imputer = SimpleImputer(strategy=\\\"median\\\")\\nimpute = imputer.fit(X_test)\\nX_test = imputer.transform(X_test)\";\n",
       "                var nbb_formatted_code = \"imputer = SimpleImputer(strategy=\\\"median\\\")\\nimpute = imputer.fit(X_test)\\nX_test = imputer.transform(X_test)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "imputer = SimpleImputer(strategy=\"median\")\n",
    "impute = imputer.fit(X_test)\n",
    "X_test = imputer.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Test Performance:\n",
      "\n",
      "XGboost_tuned: 0.791988416988417\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 56;\n",
       "                var nbb_unformatted_code = \"print(\\\"\\\\n\\\" \\\"Test Performance:\\\" \\\"\\\\n\\\")\\n\\nfinal_model = XGBClassifier(\\n    subsample=0.9,\\n    scale_pos_weight=10,\\n    n_estimators=250,\\n    learning_rate=0.1,\\n    gamma=3,\\n    random_state=1,\\n    eval_metric=\\\"logloss\\\")\\nname = \\\"XGboost_tuned\\\"\\n\\nfinal_model.fit(X_train, y_train)\\nfinal_scores = Minimum_Vs_Model_cost(y_test, final_model.predict(X_test))\\nprint(\\\"{}: {}\\\".format(name, final_scores))\";\n",
       "                var nbb_formatted_code = \"print(\\\"\\\\n\\\" \\\"Test Performance:\\\" \\\"\\\\n\\\")\\n\\nfinal_model = XGBClassifier(\\n    subsample=0.9,\\n    scale_pos_weight=10,\\n    n_estimators=250,\\n    learning_rate=0.1,\\n    gamma=3,\\n    random_state=1,\\n    eval_metric=\\\"logloss\\\",\\n)\\nname = \\\"XGboost_tuned\\\"\\n\\nfinal_model.fit(X_train, y_train)\\nfinal_scores = Minimum_Vs_Model_cost(y_test, final_model.predict(X_test))\\nprint(\\\"{}: {}\\\".format(name, final_scores))\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"\\n\" \"Test Performance:\" \"\\n\")\n",
    "\n",
    "final_model = XGBClassifier(\n",
    "    subsample=0.9,\n",
    "    scale_pos_weight=10,\n",
    "    n_estimators=250,\n",
    "    learning_rate=0.1,\n",
    "    gamma=3,\n",
    "    random_state=1,\n",
    "    eval_metric=\"logloss\",\n",
    ")\n",
    "name = \"XGboost_tuned\"\n",
    "\n",
    "final_model.fit(X_train, y_train)\n",
    "final_scores = Minimum_Vs_Model_cost(y_test, final_model.predict(X_test))\n",
    "print(\"{}: {}\".format(name, final_scores))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Accuracy</th>\n",
       "      <th>Recall</th>\n",
       "      <th>Precision</th>\n",
       "      <th>F1</th>\n",
       "      <th>Minimum_Vs_Model_cost</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.990</td>\n",
       "      <td>0.850</td>\n",
       "      <td>0.957</td>\n",
       "      <td>0.900</td>\n",
       "      <td>0.792</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Accuracy  Recall  Precision    F1  Minimum_Vs_Model_cost\n",
       "0     0.990   0.850      0.957 0.900                  0.792"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 57;\n",
       "                var nbb_unformatted_code = \"xgboost_test = model_performance_classification_sklearn(final_model, X_test, y_test)\\nxgboost_test\";\n",
       "                var nbb_formatted_code = \"xgboost_test = model_performance_classification_sklearn(final_model, X_test, y_test)\\nxgboost_test\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "xgboost_test = model_performance_classification_sklearn(final_model, X_test, y_test)\n",
    "xgboost_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 75;\n",
       "                var nbb_unformatted_code = \"# creating confusion matrix\\nconfusion_matrix_sklearn(final_model, X_test, y_test)\";\n",
       "                var nbb_formatted_code = \"# creating confusion matrix\\nconfusion_matrix_sklearn(final_model, X_test, y_test)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# creating confusion matrix\n",
    "confusion_matrix_sklearn(final_model, X_test, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The XGBoost tuned model is generalizing well on the test data with a Minimum_Vs_Model_cost of 0.792 (the cross validation training average score was 0.799 and the validation score was 0.821)\n",
    "- The model is able to make predictions resulting in a maintenance cost ~ (1/0.792 or ~1.26) times the minimum maintenance cost possible"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x864 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 74;\n",
       "                var nbb_unformatted_code = \"feature_names = data.columns\\nimportances = final_model.feature_importances_\\nindices = np.argsort(importances)\\n\\nplt.figure(figsize=(12, 12))\\nplt.title(\\\"Feature Importances\\\")\\nplt.barh(range(len(indices)), importances[indices], color=\\\"violet\\\", align=\\\"center\\\")\\nplt.yticks(range(len(indices)), [feature_names[i] for i in indices])\\nplt.xlabel(\\\"Relative Importance\\\")\\nplt.show()\";\n",
       "                var nbb_formatted_code = \"feature_names = data.columns\\nimportances = final_model.feature_importances_\\nindices = np.argsort(importances)\\n\\nplt.figure(figsize=(12, 12))\\nplt.title(\\\"Feature Importances\\\")\\nplt.barh(range(len(indices)), importances[indices], color=\\\"violet\\\", align=\\\"center\\\")\\nplt.yticks(range(len(indices)), [feature_names[i] for i in indices])\\nplt.xlabel(\\\"Relative Importance\\\")\\nplt.show()\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "feature_names = data.columns\n",
    "importances = final_model.feature_importances_\n",
    "indices = np.argsort(importances)\n",
    "\n",
    "plt.figure(figsize=(12, 12))\n",
    "plt.title(\"Feature Importances\")\n",
    "plt.barh(range(len(indices)), importances[indices], color=\"violet\", align=\"center\")\n",
    "plt.yticks(range(len(indices)), [feature_names[i] for i in indices])\n",
    "plt.xlabel(\"Relative Importance\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The top attributes which have the maximum importance for making accurate failure/ no-failure predictions are \"V18\", \"V39\", \"V26\", \"V3\" & \"V10\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TM6VZTRn4jav"
   },
   "source": [
    "## Pipelines to build the final model\n",
    "\n",
    "- Pipelines can be used to put the final model in production"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 59;\n",
       "                var nbb_unformatted_code = \"# As we already know the final model, we will not be splitting test set into test and validation sets\\n\\nX_train_pipeline = train.drop(\\\"Target\\\", axis=1)\\ny_train_pipeline = train[\\\"Target\\\"]\";\n",
       "                var nbb_formatted_code = \"# As we already know the final model, we will not be splitting test set into test and validation sets\\n\\nX_train_pipeline = train.drop(\\\"Target\\\", axis=1)\\ny_train_pipeline = train[\\\"Target\\\"]\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# As we already know the final model, we will not be splitting train set into train and validation sets\n",
    "\n",
    "X_train_pipeline = train.drop(\"Target\", axis=1)\n",
    "y_train_pipeline = train[\"Target\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 60;\n",
       "                var nbb_unformatted_code = \"X_test_pipeline = test.drop(\\\"Target\\\", axis=1)\\ny_test_pipeline = test[\\\"Target\\\"]\";\n",
       "                var nbb_formatted_code = \"X_test_pipeline = test.drop(\\\"Target\\\", axis=1)\\ny_test_pipeline = test[\\\"Target\\\"]\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "X_test_pipeline = test.drop(\"Target\", axis=1)\n",
    "y_test_pipeline = test[\"Target\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {
    "id": "Zzg12gvx4jav"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),\n",
       "                ('XGB',\n",
       "                 XGBClassifier(base_score=0.5, booster='gbtree',\n",
       "                               colsample_bylevel=1, colsample_bynode=1,\n",
       "                               colsample_bytree=1, eval_metric='logloss',\n",
       "                               gamma=3, gpu_id=-1, importance_type='gain',\n",
       "                               interaction_constraints='', learning_rate=0.1,\n",
       "                               max_delta_step=0, max_depth=6,\n",
       "                               min_child_weight=1, missing=nan,\n",
       "                               monotone_constraints='()', n_estimators=250,\n",
       "                               n_jobs=4, num_parallel_tree=1, random_state=1,\n",
       "                               reg_alpha=0, reg_lambda=1, scale_pos_weight=10,\n",
       "                               subsample=0.9, tree_method='exact',\n",
       "                               validate_parameters=1, verbosity=None))])"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 61;\n",
       "                var nbb_unformatted_code = \"model_pipeline = Pipeline(\\n    steps=[\\n        (\\\"imputer\\\", SimpleImputer(strategy=\\\"median\\\")),\\n        (\\\"XGB\\\", XGBClassifier(\\n    subsample=0.9,\\n    scale_pos_weight=10,\\n    n_estimators=250,\\n    learning_rate=0.1,\\n    gamma=3,\\n    random_state=1,\\n    eval_metric=\\\"logloss\\\")),\\n    ]\\n)\\n# Fit the model on training data\\nmodel_pipeline.fit(X_train_pipeline, y_train_pipeline)\";\n",
       "                var nbb_formatted_code = \"model_pipeline = Pipeline(\\n    steps=[\\n        (\\\"imputer\\\", SimpleImputer(strategy=\\\"median\\\")),\\n        (\\n            \\\"XGB\\\",\\n            XGBClassifier(\\n                subsample=0.9,\\n                scale_pos_weight=10,\\n                n_estimators=250,\\n                learning_rate=0.1,\\n                gamma=3,\\n                random_state=1,\\n                eval_metric=\\\"logloss\\\",\\n            ),\\n        ),\\n    ]\\n)\\n# Fit the model on training data\\nmodel_pipeline.fit(X_train_pipeline, y_train_pipeline)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model_pipeline = Pipeline(\n",
    "    steps=[\n",
    "        (\"imputer\", SimpleImputer(strategy=\"median\")),\n",
    "        (\n",
    "            \"XGB\",\n",
    "            XGBClassifier(\n",
    "                subsample=0.9,\n",
    "                scale_pos_weight=10,\n",
    "                n_estimators=250,\n",
    "                learning_rate=0.1,\n",
    "                gamma=3,\n",
    "                random_state=1,\n",
    "                eval_metric=\"logloss\",\n",
    "            ),\n",
    "        ),\n",
    "    ]\n",
    ")\n",
    "# Fit the model on training data\n",
    "model_pipeline.fit(X_train_pipeline, y_train_pipeline)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0])"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 62;\n",
       "                var nbb_unformatted_code = \"# transforming and predicting on test data\\nmodel_pipeline.predict(X_test_pipeline)\";\n",
       "                var nbb_formatted_code = \"# transforming and predicting on test data\\nmodel_pipeline.predict(X_test_pipeline)\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# transforming and predicting on test data\n",
    "model_pipeline.predict(X_test_pipeline)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7997076023391813"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "            setTimeout(function() {\n",
       "                var nbb_cell_id = 63;\n",
       "                var nbb_unformatted_code = \"Minimum_Vs_Model_cost(y_test_pipeline, model_pipeline.predict(X_test_pipeline))\";\n",
       "                var nbb_formatted_code = \"Minimum_Vs_Model_cost(y_test_pipeline, model_pipeline.predict(X_test_pipeline))\";\n",
       "                var nbb_cells = Jupyter.notebook.get_cells();\n",
       "                for (var i = 0; i < nbb_cells.length; ++i) {\n",
       "                    if (nbb_cells[i].input_prompt_number == nbb_cell_id) {\n",
       "                        if (nbb_cells[i].get_text() == nbb_unformatted_code) {\n",
       "                             nbb_cells[i].set_text(nbb_formatted_code);\n",
       "                        }\n",
       "                        break;\n",
       "                    }\n",
       "                }\n",
       "            }, 500);\n",
       "            "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "Minimum_Vs_Model_cost(y_test_pipeline, model_pipeline.predict(X_test_pipeline))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The pipeline performance is as expected (Minimum_Vs_Model_cost 0.799) indicating it was built accurately to replicate the final chosen model after necessary pre processing "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "c5hPmHyR4jaw"
   },
   "source": [
    "# Business Insights and Conclusions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FnBbg6sH4jaw"
   },
   "source": [
    "- A machine learning model has been built to minimize the total maintenance cost of machinery/processes used for wind energy production\n",
    "    - The final tuned model (XGBoost) was chosen after building ~7 different machine learning algorithms & further optimizing for target class imbalance (having few \"failures\" and many \"no failures\" in dataset) as well as finetuning the algorithm performance (hyperparameter and cross validation techniques)\n",
    "\n",
    "    - A pipeline was additionally built to productionise the final chosen model\n",
    "   \n",
    "   \n",
    "- The model is expected to generalize well in terms of predictions & expected to result in a maintenance cost ~1.26 times minimum possible maintenance cost. Having no model in place for predictions could potentially result in costs as high as ~2.67 minimum possible maintenance cost. Hence, productionising the model has a large cost saving advantage\n",
    "\n",
    "- The main attributes of importance for predicting failures vs. no failures were found to be \"V18\", \"V39\", \"V26\", \"V3\" & \"V10\" in order of decreasing importance. This added knowledge can be used to refine the process of collecting more frequent sensor information to be used in improving the machine learning model to further decrease maintenance costs "
   ]
  }
 ],
 "metadata": {
  "colab": {
   "name": "Learner's_notebook.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
