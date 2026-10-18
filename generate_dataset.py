"""Dataset académico 100% sintético para F05 Streamlit (semilla fija).
Ejecuta: python generate_dataset.py
Requiere: numpy
"""
from pathlib import Path
import csv
import numpy as np

SEED = 20261017
N_UNIQUE = 990
N_DUPLICATES = 10
rng = np.random.default_rng(SEED)

age = rng.integers(21, 69, N_UNIQUE)
employment_type = rng.choice(['asalariado','independiente','otro'], N_UNIQUE, p=[.69,.25,.06])
monthly_income = np.clip(rng.lognormal(np.log(23000), .62, N_UNIQUE), 7000, 140000).round(0).astype(int)
loan_amount = np.clip(monthly_income*rng.uniform(1.8, 9, N_UNIQUE), 12000, 460000).round(-2).astype(int)
loan_term = rng.choice([12,18,24,36,48,60],N_UNIQUE,p=[.08,.1,.24,.34,.17,.07])
credit_history_months = np.minimum((age-18)*12,rng.gamma(3.5,37,N_UNIQUE).astype(int)+3)
credit_history_months = np.maximum(credit_history_months, 6)
months_on_book = np.array([int(rng.integers(1, min(int(t), int(h))+1)) for t,h in zip(loan_term,credit_history_months)])
# Variable latente conductual para que existan correlaciones plausibles, sin separación perfecta.
fragility = rng.normal(0, 1, N_UNIQUE) + np.where(employment_type == 'independiente',.18,0)
fragility += -.000014*(monthly_income-23000)
fragility = np.clip(fragility,-3.2,3.5)
previous_delinquencies = np.clip(rng.poisson(np.exp(-.65+.43*fragility)),0,9)
# El MOP académico es 0-4 (NO equivale a códigos oficiales de Buró).
# Corregimos probabilidades y generamos MOP con logits ordinales para robustez.
severity = .25*fragility + .23*previous_delinquencies + rng.normal(0,.85,N_UNIQUE)
worst_mop_12m = np.select([severity<-.1,severity<.7,severity<1.4,severity<2.3],[0,1,2,3],default=4).astype(int)
bureau_inquiries_6m = np.clip(rng.poisson(np.exp(.5+.23*fragility)),0,15)
bureau_inquiries_3m = rng.binomial(bureau_inquiries_6m, .53)
revolving_accounts = np.clip(rng.poisson(np.clip(2.4 + .000025*(monthly_income-23000), .4,5.6)),0,12)
credit_utilization = np.clip(rng.beta(2.3,2.4,N_UNIQUE)+.105*fragility+.018*previous_delinquencies,0,1).round(3)
dti_ratio = np.clip(rng.beta(2.4,4.5,N_UNIQUE)+.065*fragility+.014*previous_delinquencies, .04,.93).round(3)
interest_rate = np.clip(19.8 + 3.8*fragility + 2.8*previous_delinquencies + rng.normal(0,4.1,N_UNIQUE), 10,52).round(2)
# Incumplimiento FUTURO simulado en horizonte de 12 meses: correlación probabilística,
# no se utiliza información del horizonte para las variables explicativas.
logit = (-2.05 + .68*(dti_ratio-.35)/.2 + .52*(credit_utilization-.5)/.25
         + .35*(worst_mop_12m-1.0) + .17*previous_delinquencies
         + .18*(bureau_inquiries_3m-1) -.22*(credit_history_months-100)/100
         + .11*(revolving_accounts-2.5) + .15*(interest_rate-23)/10
         -.12*(monthly_income-23000)/25000 + rng.normal(0,.65,N_UNIQUE))
prob = 1/(1+np.exp(-np.clip(logit,-12,12)))
default_12m = rng.binomial(1,prob)
headers = ['customer_id','age','monthly_income','employment_type','loan_amount',
           'loan_term','interest_rate','dti_ratio','credit_utilization',
           'previous_delinquencies','months_on_book','bureau_inquiries_3m',
           'bureau_inquiries_6m','revolving_accounts','credit_history_months',
           'worst_mop_12m','default_12m']
cols = [np.array([f'CR{i:04d}' for i in range(1,N_UNIQUE+1)]),age,monthly_income,
        employment_type,loan_amount,loan_term,interest_rate,dti_ratio,
        credit_utilization,previous_delinquencies,months_on_book,
        bureau_inquiries_3m,bureau_inquiries_6m,revolving_accounts,
        credit_history_months,worst_mop_12m,default_12m]
rows = [[v.item() if hasattr(v,'item') else v for v in row] for row in zip(*cols)]
# Casos de calidad controlados: nulos en 3 variables + 10 duplicados exactos.
for key, count in [('monthly_income',12),('employment_type',9),('credit_utilization',8)]:
    for idx in rng.choice(N_UNIQUE, count, replace=False):
        rows[int(idx)][headers.index(key)] = ''
for idx in rng.choice(N_UNIQUE, N_DUPLICATES, replace=False):
    rows.append(rows[int(idx)].copy())
rng.shuffle(rows)
out = Path(__file__).resolve().parent / 'data' / 'credit_risk_customers.csv'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)
print(f'{out} | filas={len(rows)}, columnas={len(headers)}, defaults={sum(int(r[-1]) for r in rows)}')
