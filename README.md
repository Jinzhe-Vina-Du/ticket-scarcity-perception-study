# Ticket Scarcity Perception Study

An independent research project examining how perceived ticket scarcity affects 
university students' purchase urgency, decision-making behavior, and psychological 
responses.

## Research Overview

**Sample:** 125 university students (survey conducted via Wenjuanxing)  
**Methods:** Likert-scale survey, Pearson correlation analysis, descriptive statistics  
**Tools:** Python (pandas, scipy, matplotlib, seaborn)

## Key Construct: β Inertia Coefficient

This study introduces the **β inertia coefficient** — defined as the gap between 
a respondent's felt urgency (Q10) and their actual likelihood to act on that urgency 
(Q11). A positive β indicates that scarcity triggers psychological urgency without 
converting to purchase action.

**Finding:** Mean β = 0.93 (SD = 1.27), suggesting that urgency and action are 
systematically decoupled under scarcity conditions among this sample.

## Files

- `scarcity_analysis.py` — Full analysis pipeline (data cleaning, correlation, visualizations)
- `figures/` — Output charts (distribution, heatmap, mean scores, group comparisons)

## Key Findings

- Urgency mean = 3.38 vs. Urgency-to-Action mean = 2.45 (gap = 0.93)
- Perceived value shows the strongest correlation with both urgency (r=0.57) and action (r=0.61)
- Fear of loss significantly predicts urgency (r=0.54, p<0.01)
- ~63% of respondents report wanting the ticket *more* under scarcity conditions
