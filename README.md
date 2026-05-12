This repository contains the code used for statistical analysis in the Master's Thesis titled: "His Depression, Her Depression: A Corpus-Based Analysis of Gendered Linguistic Markers of Depression"

The analysis was conducted on a custom database created by running data from the Distress Analysis Interview Corpus (DAIC-WOZ) (Gratch et al., 2014; DeVault et al., 2014) through two Linguistic Inquiry and Word Count (LIWC) dictionaries: the standard English LIWC (2022), and the Absolutist word dictionary (Al-Mosaiwi and Johnstone, 2018).
Each statistical method was tailored towards the research questions and the contents of the database (binary gender scores, binary depression scores, continuous ratings per selected linguistic marker of depression),

The code contains the following:

================================================================================
  Linguistic Markers of Depression — Full Analysis Pipeline
================================================================================
  Research Questions:
    RQ1: Do depressed and non-depressed people differ in marker use?
    RQ2: Do depressed men and women differ in marker use?

  Sections:
    0. Configuration
    1. Load & Inspect Data
    2. Descriptive Statistics
    3. Visualisations
    4. RQ1 — t-tests / Mann-Whitney U + FDR correction + effect sizes
    5. RQ2 — MANOVA + follow-up ANOVAs (depressed participants only)
    6. Logistic Regression (per gender + full sample)
    7. Radar Chart (group profiles)
================================================================================
