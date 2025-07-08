# EnVision
## Project Description
This university project simulated a real-world business case in the area of energy consultation. We began by identifying a problem, substantiating it with relevant literature, and ultimately solving it using data science methods. This project was done in April and Mai 2024.
## 1. Problem
### 1.1 Background
Buildings in the EU account for 36% of greenhouse gas emissions and 40% of energy consumption. In Germany, residential housing alone contributes to 15% of total national emissions (Statista, 2024). To meet its climate targets, such as 65% GHG reduction by 2030 and net, zero by 2045—the government promotes energy-efficient refurbishments, supported by mandatory energy consultations.

The role of energy consultants is therefore critical in guiding homeowners through sustainable renovations. Demand for such services is rising, with over 280,000 energy consultations in 2022 and a 25% annual increase in certified consultants.

### 1.2 Problem Spotlight
Despite growing demand, energy consultants face persistent challenges in acquiring and retaining clients. Two key barriers are ineffective Customer Relationship Management (CRM) and limited marketing and managerial skills. The absence of these capabilities can significantly hinder business success and, in competitive markets, even lead to failure (Radipere & Scheers, 2014). Additionally, customer acquisition is notably time-consuming: based on our interviews, consultants spend an average of 3 out of 13 total hours per consultation,nearly one quarter of their time, solely on acquiring clients.
## 2. Solution & Objectives
The goal of this project was to develop machine learning models that support energy consultants in optimizing their sales and marketing strategies. Specifically, we pursued two objectives:
1. Identify target customer profiles using unsupervised learning (k-means clustering) to uncover shared characteristics among homeowners who are likely to book an energy consultation. These insights can inform the development of tailored marketing campaigns aligned with the preferences of each customer segment.
2. Predict consultation likelihood using supervised learning models trained to estimate which homeowners are most likely to book a consultation. In addition to making accurate predictions, the models also reveal which factors are most influential in driving this decision. These predictions are then applied to a separate dataset of potential leads ([link here]), enabling consultants to focus their outreach on the most promising contacts. This targeted approach can significantly reduce time spent on cold outreach and improve conversion efficiency.

## 3. Methodology
### 3.1 Data Collection
To better understand which homeowners are most likely to participate in energy consultations, we designed a structured survey targeting a diverse group of individuals across Germany between 2020 and 2024. The survey was distributed via networks of building cooperatives and homeowners’ associations, enabling direct access to a broad and demographically varied audience. This approach helped ensure high response rates and accurate insights into residential preferences, behaviors, and motivations.
The survey captured quantitative data using three types of input variables:

**Numeric continuous** (e.g., age, income)

**Numeric discrete** (e.g., energy awareness on a 1–5 scale)

**Categorical** (e.g., house type, occupation status)

The goal was to generate a rich dataset that reflects the diversity of German homeowners in both urban and rural contexts, allowing us to uncover meaningful patterns for customer targeting.
### 3.2 Research Focus
Based on a thorough literature review, we explored which factors most strongly influence a homeowner’s decision to book an energy consultation. The research built upon foundational work by scholars such as Galvin (2014), Janda (2011), Stern (2000), and Heinzle (2012), who investigated the roles of structural housing characteristics, energy usage behavior, and environmental attitudes contributing to sustainable choices.
### 3.3 Key Variables
| Key Variables | Reason |
|-----------|-------|
| House Type, Age, and Size | Critical physical attributes determining urgency and scope for energy optimization. |
| Income       | Influences financial capability for renovations and likelihood of leveraging subsidies for energy-efficient improvements. |
| Occupation Status and Education Level | Indicates access to information and comprehension of energy-saving measures and subsidies. |
| Location     | Urban or rural settings impact availability of consultancy services and reflect energy usage patterns and conservation opportunities. |
| Energy Bill and Source      | Reflects current expenditure and incentive for efficiency improvements, openness to alternative energy solutions. |
| Knowledge and Awareness of Energy Issues | Reflects existing understanding of energy matters, correlating with tendency to seek expert advice. |
| Attitudes Towards Energy Reduction and Investment Willingness | Psychological drivers indicating readiness to act on efficiency measures. |
| Belief in Climate Change      | Ideological factor motivating actions towards sustainable living and pursuit of energy consultations. |
| Past Renovations and Environmental Concern  | Historical actions and environmental mindfulness suggest behavior beneficial to future consultation engagements. |
| Dependent Variable: History of Booking Energy Consultation | Direct indicator of the target customer for energy consultants. |

## 4. Creation of Datasets
As the scope of the course was only a few weeks, we were unable to gather sufficient data in the short time. Therefore we opted to construct our own datasets, based on the 32 interviews we had conducted about the topic:
### 1. Training Dataset (Link)
This dataset was designed for training and evaluating our machine learning models. It includes multiple homeowner profiles grouped into three segments based on observed patterns from the interviews, as well as one additional group with randomized data to simulate noise and increase realism. The dataset contains a labeled outcome variable indicating whether a homeowner had booked an energy consultation or not. The goal was to simulate a realistic distribution of potential customer types.
### 3. Lead List Dataset (Link)
The second dataset simulates a real-world contact list, such as those purchased from online providers. It mirrors the structure of the training dataset but includes personal contact information (full name, email address, phone number) and does not contain the outcome label. This resembles a typical challenge faced by energy consultants, who often work with low-quality lead lists. Our predictive models are applied to this dataset to identify and prioritize the most promising leads, helping consultants save time and improve outreach efficiency.

## 4. Creating Models
### 4.1 Data Cleaning and Preprocessing [BILDER HINZUFÜGEN]
The dataset underwent a structured preprocessing pipeline to ensure high data quality and optimize performance for machine learning models.

First, we verified data integrity. No missing values were found, so no imputation or row removal was needed. We then analyzed numerical variables for outliers using boxplots. While outliers were present in some variables (e.g. house size, age), they were few in number (1.8% of the data), reflected realistic values, and did not indicate data entry errors. As a result, we retained them to preserve the dataset’s integrity.

Categorical variables were transformed using one-hot encoding. Depending on the model, we either retained or dropped one dummy variable to avoid multicollinearity (“dummy trap”). For example, logistic regression and k-means required dropping one category, while decision trees and random forests were not sensitive to this.

To check variables for interrelationships, a correlation matrix was built. Though there were some moderately strong correlations, none of them was above the set threshold of 0.9. Therefore no necessity to delete variables was seen.

We also prepared the target variable for binary classification. The outcome variable booked_energy_consultation was mapped to a Boolean format:<br>
Yes → True<br>No and Considered but not used → False.

For feature scaling, we applied MinMaxScaler and StandardScaler selectively:

- MinMaxScaler was used for models that rely on distance calculations, such as k-nearest neighbors and neural networks.
- StandardScaler was used for models that assume a normal distribution, such as SVMs or PCA. [CHECK IF USED]

These steps are implemented in the notebook [NOTEBOOK HERE]
## 5. How to Run the Project
Follow these steps to set up and run the project on your local machine:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-rep
```
### 2. Set up the environment
Install all required packages:

```bash
pip install -r requirements.txt
```

Alternatively, if you are using a virtual environment or conda:
```bash
# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook
```

Navigate to the `notebooks/` folder and run the files in the following order: [ANPASSEN]

1. `01_create_datasets.ipynb`  
2. `02_cleaning_preprocessing.ipynb`  
3. `03_adapt_dataset.ipynb`  
4. *(...other notebooks as needed)*  
5. `10_prediction_on_new_data.ipynb`

### 4. Make predictions on new data

In the final notebook (`LINK HIER REIN`), the trained model is applied to the lead list (`LINK HIER REIN`) to predict which homeowners are most likely to book an energy consultation. The top leads can then be exported for targeted outreach.
## 6. Example Output / Screenshots

## Sources
Statista 2024
Radipere & Scheers, 2014
Galvin (2014), Janda (2011), Stern (2000), and Heinzle (2012)
