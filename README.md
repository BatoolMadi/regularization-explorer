# Regularization Explorer

This project explores how Logistic Regression coefficients change as regularization strength varies.

## What was done
- Trained Logistic Regression models with L1 and L2 regularization
- Used 20 values of C (logarithmically spaced from 0.001 to 100)
- Recorded coefficient values for each feature
- Visualized the regularization path

## Output
- A plot showing how coefficients change across different values of C

![Regularization Plot](regularization_plot.png)

## Interpretation
The regularization path plot clearly illustrates how coefficients change as regularization strength varies. As C decreases, coefficients shrink significantly in both models. In the L1 plot, several coefficients are driven exactly to zero, showing that L1 performs feature selection by eliminating less important variables. In contrast, the L2 plot shows that coefficients are reduced gradually but remain non-zero, meaning all features are retained. Some features remain stable across all values of C, indicating strong predictive power, while others shrink rapidly, suggesting weaker influence. Based on this behavior, L1 is more suitable when model simplicity and feature selection are desired, whereas L2 is preferable when preserving all features while controlling overfitting is important.