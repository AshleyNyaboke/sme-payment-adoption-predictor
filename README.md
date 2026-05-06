# 🇰🇪 SME Digital Payment Adoption Predictor

> Built specifically for the **Absa Bank Kenya & Airtel Money Kenya** 
> strategic partnership on digital payments for SMEs.



## 🎯 Project Overview

Millions of Kenyan SMEs still rely on cash for daily transactions despite 
the availability of digital payment infrastructure. This project uses 
machine learning to predict which SMEs are most likely to adopt digital 
paybill payments — helping financial institutions like Absa and Airtel 
target the right businesses with the right interventions at the right time.

**Live Dashboard:** ("https://styling-uncapped-flaxseed.ngrok-free.dev")



## 💡 Business Context

| Company | Relevance |
|---|---|
| **Absa Bank Kenya** | Investing KES 3 billion annually in digital banking. Paybill 303030 is a key growth channel for SME payments |
| **Airtel Money Kenya** | Recently separated into Airtel Money Kenya Limited. Partners with Absa to allow payments via *334# and My Airtel App with 100% cashback incentives |

Instead of marketing digital payments to every SME blindly — which is 
expensive — this model helps both companies prioritize the highest 
probability adopters, saving marketing budget and improving conversion rates.

---
 📊 Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | 86.00% |
| **Random Forest** | **89.00% ✅ Best Model** |

### Detailed Scorecard (Random Forest)

| Metric | Non-Adopters | Adopters |
|---|---|---|
| Precision | 86% | 90% |
| Recall | 74% | 95% |
| F1-Score | 79% | 93% |

### Key Finding
Mobile money transaction frequency is the single strongest predictor 
of SME digital payment adoption — stronger than revenue, business age, 
or location. When the model predicts an SME will adopt, it is right 
90% of the time, and it catches 95% of all actual adopters. This is 
a directly actionable insight for Absa and Airtel's marketing teams.


---

##  Feature Importance

The top drivers of SME digital payment adoption discovered by the model:

1.  **Mobile Money Frequency** — strongest predictor by far
2.  **Owns Smartphone** — digital payments need a digital device
3.  **Has Bank Account** — already in the financial system
4.  **Urban Location** — city businesses adopt faster
5.  **Distance to Nearest Bank** — further away = more motivation to go digital

---


## 🚀 How To Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/AshleyNyaboke/sme-payment-adoption-predictor.git
cd sme-payment-adoption-predictor
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

---

## 📦 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data generation and manipulation |
| Scikit-learn | Machine learning models |
| Matplotlib | Data visualization |
| Streamlit | Interactive dashboard |
| Google Colab | Development environment |

---

## 📈 Dataset

Since real SME transaction data is proprietary, this project uses a 
synthetic dataset of **2,000 Kenyan SMEs** built using patterns from:
- CBK FinAccess Survey
- KNBS SME Reports
- World Bank Kenya Financial Inclusion Data

Features include county, business sector, monthly revenue, mobile money 
usage, smartphone ownership, bank account status, and more — all grounded 
in real Kenyan market conditions.

---

## 👩🏾‍💻 About The Author

**Ashley Nyaboke Kibwogo**  
Data Scientist | Nairobi, Kenya  
Building data solutions for Kenya's financial inclusion challenges.

🔗 [LinkedIn](https://www.linkedin.com/in/ashley-nyaboke/)  
📧 [Email](ashleynyaboke333@gmail.com)

---

## 📬 Get In Touch

If you work in data, fintech, or financial inclusion in Kenya and would 
like to collaborate or discuss this project — I'd love to connect!

---

*Built with 💙 for Kenya's digital payments future*
