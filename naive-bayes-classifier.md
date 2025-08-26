
# Naive Bayes (Bernoulli) from scratch in C

I’ve been implementing small machine learning algorithms from scratch to really understand how they work. This time, let’s look at **Naive Bayes**, specifically the **Bernoulli variant**.


## What is Naive Bayes?

Naive Bayes is a simple but surprisingly effective algorithm for classification.  

- **Bayes’ theorem** tells us how to update probabilities when we see new evidence:  

  $$
  P(y \mid x) \propto P(y) \cdot P(x \mid y)
  $$

- In Naive Bayes, we assume that each feature is **independent** given the class. This assumption is rarely true in practice, but it simplifies things and often works well.

- For **Bernoulli Naive Bayes**, features are binary (yes/no, true/false). For example: *Is the fruit yellow? Is it sweet? Is it long?*

So for a binary class problem, the formula looks like:

$$
P(y \mid x) \propto P(y) \prod_{f} P(x_f \mid y)
$$

Here:
- $P(y)$ is the prior probability of the class.  
- $P(x_f \mid y)$ is the probability that feature $f$ has value 1 given the class.  

We use **Laplace smoothing** to avoid zero probabilities, and work in **log-space** to keep calculations stable.



## A tiny dataset

Imagine we want to predict if a fruit is a mango ($y=1$).  
Each fruit has three yes/no features:

1. Yellow  
2. Sweet  
3. Long  

Our dataset might look like this (1 = yes, 0 = no):

| Yellow | Sweet | Long | IsMango |
|--------|-------|------|---------|
| 1      | 1     | 0    | 1       |
| 0      | 0     | 1    | 0       |
| 1      | 1     | 0    | 1       |
| 0      | 0     | 1    | 0       |
| 1      | 0     | 0    | 1       |


## Solving this dataset by hand (step by step)

We'll compute everything with tiny whole numbers and fractions so it's easy to follow.

1) Count how many examples are mangoes (y=1) vs not (y=0)
- y=1: 3 rows
- y=0: 2 rows
- With Laplace smoothing (add 1 to each class, total outcomes = 2):
  - $P(y=1) = \dfrac{3+1}{5+2} = \dfrac{4}{7} \approx 0.571$
  - $P(y=0) = \dfrac{2+1}{5+2} = \dfrac{3}{7} \approx 0.429$

2) For each feature, count 1s within each class, then smooth
- For y=1 (3 samples):
  - $P(\text{Yellow}=1 \mid y=1) = \dfrac{3+1}{3+2} = \dfrac{4}{5} = 0.8$; $P(\text{Yellow}=0 \mid y=1) = \dfrac{1}{5} = 0.2$
  - $P(\text{Sweet}=1 \mid y=1) = \dfrac{2+1}{3+2} = \dfrac{3}{5} = 0.6$; $P(\text{Sweet}=0 \mid y=1) = \dfrac{2}{5} = 0.4$
  - $P(\text{Long}=1 \mid y=1) = \dfrac{0+1}{3+2} = \dfrac{1}{5} = 0.2$; $P(\text{Long}=0 \mid y=1) = \dfrac{4}{5} = 0.8$
- For y=0 (2 samples):
  - $P(\text{Yellow}=1 \mid y=0) = \dfrac{0+1}{2+2} = \dfrac{1}{4} = 0.25$; $P(\text{Yellow}=0 \mid y=0) = \dfrac{3}{4} = 0.75$
  - $P(\text{Sweet}=1 \mid y=0) = \dfrac{0+1}{2+2} = \dfrac{1}{4} = 0.25$; $P(\text{Sweet}=0 \mid y=0) = \dfrac{3}{4} = 0.75$
  - $P(\text{Long}=1 \mid y=0) = \dfrac{2+1}{2+2} = \dfrac{3}{4} = 0.75$; $P(\text{Long}=0 \mid y=0) = \dfrac{1}{4} = 0.25$

3) Make a prediction for a new fruit
- Suppose the fruit is Yellow=1, Sweet=1, Long=0.
- Score for y=1:
  - $\dfrac{4}{7} \times \dfrac{4}{5} \times \dfrac{3}{5} \times \dfrac{4}{5} = \dfrac{192}{875} \approx 0.219$
- Score for y=0:
  - $\dfrac{3}{7} \times \dfrac{1}{4} \times \dfrac{1}{4} \times \dfrac{1}{4} = \dfrac{3}{448} \approx 0.0067$
- Normalize to get probabilities:
  - $P(y=1 \mid x) \approx \dfrac{0.219}{0.219 + 0.0067} \approx 0.97$
  - $P(y=0 \mid x) \approx 0.03$
- So we'd predict y=1 (it's a mango).

This is exactly what the C code automates: count, smooth, multiply the relevant feature probabilities, and pick the class with the larger final score.

## The classifier in C

Here’s the core idea in code: count frequencies, apply Laplace smoothing, accumulate probabilities in log-space, and finally normalize.

```c
float calculate_posterior_probability(
    float dataset_x[5][3], float dataset_y[5], float input[3]
) {
    // 1. Count how many samples belong to each class
    // 2. Compute class priors with Laplace smoothing
    // 3. For each feature, compute likelihoods:
    //    P(x_f=1 | y=1), P(x_f=0 | y=1), and similarly for y=0
    // 4. Accumulate log probabilities
    // 5. Convert back from log-space and normalize
}
```

## What is Laplace smoothing?

When working with probabilities, a big problem comes up if we never see a certain feature in the training data.  
For example: if in our dataset no mango was ever labeled "Long," then  
$$P(\text{Long}=1 \mid \text{Mango}) = 0.$$

If we multiply by zero, the whole probability for that class collapses to zero, which is almost never what we want.

**Laplace smoothing** fixes this by pretending we saw each possible outcome *one extra time*.  
So instead of using:

$$ P = \frac{\text{count}}{\text{total}} $$

we use:

$$ P = \frac{\text{count} + 1}{\text{total} + \text{number of outcomes}} $$

This tiny adjustment keeps probabilities from being exactly zero or one. It makes the model more robust, especially with small datasets.

## Key points

* **Naive assumption**: We pretend features are independent given the class. This is rarely true but makes the model fast and often effective.
* **Laplace smoothing**: Adding +1 in counts prevents probabilities from becoming zero when data is small.
* **Log-space arithmetic**: Multiplying small numbers can cause underflow. Working with logs avoids this.
* **Variants**:

  * Bernoulli NB (binary features)
  * Multinomial NB (count features, like word frequencies)
  * Gaussian NB (continuous features)

---

That’s it. A compact and efficient classifier that you can implement in a few dozen lines of C, yet captures a powerful idea: simple assumptions can lead to strong results.