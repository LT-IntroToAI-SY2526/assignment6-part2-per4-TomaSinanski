# Assignment 6 Part 2 - Writeup

---

## Question 1: Feature Importance

Based on your house price model, rank the four features from most important to least important. Explain how you determined this ranking.

**YOUR ANSWER:**
1. Most Important: Bedrooms
2. Bathrooms 
3. Age
4. Least Important: Square Feet

**Explanation:**
The terminal displayed the coefficent for each feature and feature with the largest coefficient is the most important and the smallest is least important.



---

## Question 2: Interpreting Coefficients

Choose TWO features from your model and explain what their coefficients mean in plain English. For example: "Each additional bedroom increases the price by $___"

**Feature 1:**
Each additional bedroom increases the price by $6649.

**Feature 2:**
Each additional bathroom increases the price by $3859.

---

## Question 3: Model Performance

What was your model's R² score? What does this tell you about how well your model predicts house prices? Is there room for improvement?

**YOUR ANSWER:**
The R^2 score was 0.9936 which means that 99.36% of the house's value is determined by the 4 features we used. This means the prediction data closly matched the testing date. I would say there is not much room for improvement.




---

## Question 4: Adding Features

If you could add TWO more features to improve your house price predictions, what would they be and why?

**Feature 1:**
Location

**Why it would help:**
Different locations make houses more desirable and therefor increase their price.

**Feature 2:**
Extra stuff(pools, backyards, garages)

**Why it would help:**
People likely want these things and are willing to pay more for them.

---

## Question 5: Model Trust

Would you trust this model to predict the price of a house with 6 bedrooms, 4 bathrooms, 3000 sq ft, and 5 years old? Why or why not? (Hint: Think about the range of your training data)

**YOUR ANSWER:**
No because those values are out of our range so we cannot assume the same trends could occur with higher values.


