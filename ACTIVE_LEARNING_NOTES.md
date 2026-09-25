# Active Learning & AI-Assisted Annotation — Engineering Reference

> Pragmatic engineering notes on Uncertainty Sampling, Score Aggregation, Diversity Sampling, Softmax Miscalibration, and Annotation Quality Assurance.

---

## 1. Uncertainty Sampling — Theory & Mathematics

### 1.1 Core Concept

In pool-based active learning, a model has access to a large pool of unlabeled data but a limited budget to request human labels. **Uncertainty sampling** is the most widely used query strategy: it selects the instances the model is most confused about, under the premise that resolving that confusion yields the highest information gain per label.

The active learning loop operates as follows:

1. Train the model on the current labeled set.
2. Run inference on the entire unlabeled pool.
3. **Score** each instance by its uncertainty.
4. **Sample** a batch of high-uncertainty instances.
5. Send that batch to annotators.
6. Add the newly labeled data to the training set.
7. Repeat.

### 1.2 Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $x$ | An unlabeled data instance |
| $y$ | A class label from $Y = \{y_1, y_2, \dots, y_K\}$ |
| $P(y \mid x)$ | Model's predicted probability that $x$ belongs to class $y$ |
| $\hat{y}_1$ | The class with the highest predicted probability |
| $\hat{y}_2$ | The class with the second-highest predicted probability |

Classes are sorted in descending order of probability:

$$P(\hat{y}_1 \mid x) \ge P(\hat{y}_2 \mid x) \ge \dots \ge P(\hat{y}_K \mid x)$$

### 1.3 Least Confidence Sampling

**Intuition:** Select the instance for which the model's confidence in its *best guess* is the lowest. If the model cannot even commit to its top choice, it clearly does not understand the instance.

**Formula:**

$$x^*_{LC} = \arg\max_{x} \left( 1 - P(\hat{y}_1 \mid x) \right)$$

**Strengths:**
- Trivially simple to implement. Most object detectors already emit a single confidence score per prediction; that score *is* $P(\hat{y}_1 \mid x)$.
- Computationally negligible — no need to access the full probability distribution.

**Weaknesses:**
- It discards all information beyond the top class. Consider two instances with predictions `[0.4, 0.39, 0.21]` and `[0.4, 0.01, 0.59]`. Least Confidence treats both identically (confidence = 0.4), even though the first instance is genuinely ambiguous between two classes while the second has a clear runner-up that simply outranks the top prediction (this would be caught by the full distribution).
- For binary classification, this strategy is equivalent to Margin and Entropy. It only diverges in multi-class settings.

### 1.4 Margin Sampling

**Intuition:** Select the instance where the gap between the first and second most likely classes is the smallest. A tiny margin means the model is caught in a tug-of-war between two competing hypotheses.

**Formula:**

$$x^*_{M} = \arg\min_{x} \left( P(\hat{y}_1 \mid x) - P(\hat{y}_2 \mid x) \right)$$

Equivalently, to express it as a maximization of an uncertainty score:

$$U_M(x) = 1 - \left( P(\hat{y}_1 \mid x) - P(\hat{y}_2 \mid x) \right)$$

**Strengths:**
- Superior to Least Confidence for multi-class problems because it explicitly captures how closely the model is torn between its two top hypotheses.
- Excels at finding instances that lie directly on the decision boundary between two specific classes (e.g., distinguishing `truck` from `bus` in autonomous driving).

**Weaknesses:**
- Still ignores all classes beyond the top two. If the model assigns probability `[0.35, 0.33, 0.32]` (genuinely confused across all three classes), Margin only sees the gap `0.35 - 0.33 = 0.02` and misses the fact that the third class is equally competitive.

### 1.5 Entropy Sampling

**Intuition:** Use Shannon Entropy from information theory to measure the total disorder in the predicted probability distribution. Maximum entropy occurs when the distribution is perfectly uniform (the model has zero preference for any class).

**Formula:**

$$x^*_E = \arg\max_{x} \left( - \sum_{k=1}^{K} P(y_k \mid x) \log_2 P(y_k \mid x) \right)$$

For a $K$-class problem, the theoretical maximum entropy is $\log_2 K$ bits (achieved when $P(y_k \mid x) = \frac{1}{K}$ for all $k$).

**Strengths:**
- The most comprehensive uncertainty measure. It evaluates the *entire* probability distribution, making it the most robust strategy for complex multi-class taxonomies (e.g., traffic sign classification with 40+ sign types).
- Theoretically grounded in information theory — it directly quantifies how many bits of information are needed to resolve the model's confusion.

**Weaknesses:**
- Requires access to the full softmax output vector, not just the top-1 confidence. Some production detectors only expose the top-1 score.
- Slightly more expensive to compute over very large label spaces (though negligible compared to model inference time).

### 1.6 The Canonical Divergence Example

To demonstrate exactly where and why these three strategies disagree, consider a clean 3-class problem with two instances:

| | Class 1 | Class 2 | Class 3 |
|---|---------|---------|---------|
| **Instance X** | 0.50 | 0.50 | 0.00 |
| **Instance Y** | 0.50 | 0.25 | 0.25 |

**Least Confidence** ($1 - P(\hat{y}_1)$):
- X: $1 - 0.5 = 0.5$
- Y: $1 - 0.5 = 0.5$
- **Result: Tie.** It only evaluates the top prediction and cannot distinguish between them.

**Margin Sampling** ($P(\hat{y}_1) - P(\hat{y}_2)$):
- X: $0.5 - 0.5 = 0.0$ (perfectly torn between top two)
- Y: $0.5 - 0.25 = 0.25$
- **Result: Picks X.** The model's top two classes are in a dead heat.

**Entropy Sampling** ($-\sum p \log_2 p$):
- X: $-(0.5 \times \log_2 0.5 + 0.5 \times \log_2 0.5) = -(0.5 \times -1 + 0.5 \times -1) = 1.0$ bits
- Y: $-(0.5 \times \log_2 0.5 + 0.25 \times \log_2 0.25 + 0.25 \times \log_2 0.25) = -(0.5 \times -1 + 0.25 \times -2 + 0.25 \times -2) = 1.5$ bits
- **Result: Picks Y.** The confusion is distributed across *all three* classes, yielding higher total information uncertainty.

**Key insight:** Margin found the binary confusion (X). Entropy found the holistic confusion (Y). Neither is universally "better" — the right choice depends on the structure of your classification problem.

### 1.7 When to Use Which Strategy

| Strategy | Best Fit | Avoid When |
|----------|----------|------------|
| **Least Confidence** | Binary classification; detector outputs only a single confidence score | Multi-class with >3 classes |
| **Margin** | Errors concentrate on a specific pair of confusable classes (`truck`/`bus`, `pedestrian`/`rider`) | Many classes are simultaneously competitive |
| **Entropy** | Schema has dozens of classes at similar hierarchy level (traffic signs, fine-grained vehicle types) | Binary classification (all three are equivalent) |

---

## 2. Score Aggregation for Object Detection

### 2.1 The Problem

Classification uncertainty is defined per instance. In object detection, the "instance" is an image, but uncertainty is calculated per **bounding box prediction**. A single image may contain $N$ predicted bounding boxes, each with its own uncertainty score $u_i$.

Annotation platforms like CVAT assign work at the **image level** (one image = one annotation unit). Therefore, we need an aggregation function to collapse $\{u_1, u_2, \dots, u_N\}$ into a single image-level score $U(X)$.

### 2.2 Sum Aggregation

$$U_{sum}(X) = \sum_{i=1}^{N} u_i$$

**Behavior:** The image's total cumulative uncertainty. Heavily favors dense scenes.

**Strengths:**
- When labeling cost is **per image** (flat rate per frame), Sum maximizes the number of labeled objects per dollar spent, since dense images yield more bounding box annotations per unit cost.
- Naturally prioritizes crowded intersections, parking lots, and rush-hour traffic — scenes where the model is likely to struggle with overlapping objects.

**Weaknesses:**
- Biased toward object count rather than genuine model confusion. An image with 100 easy objects and marginal uncertainty per box will outscore an image with 1 genuinely confusing object.
- Highly susceptible to false positive noise. If the model generates many low-confidence spurious detections (background clutter), Sum will prioritize those noisy images.

### 2.3 Mean Aggregation

$$U_{mean}(X) = \frac{1}{N} \sum_{i=1}^{N} u_i$$

**Behavior:** The expected (average) uncertainty of a detection within the image. Normalizing by $N$ removes the density bias.

**Strengths:**
- Treats sparse and dense images on an equal footing. A highway scene with 3 objects and an intersection with 50 objects are compared fairly.
- Good default choice when you have no strong prior about whether dense or sparse scenes are more valuable.

**Weaknesses:**
- **Dilution effect.** If an image contains 1 extremely confusing object but 20 perfectly clear objects (low $u_i$), the 20 clear objects drag the average down to near zero. The model will skip the image entirely, missing the chance to learn from the one hard case.
- Sensitive to NMS thresholds: changing the NMS IoU threshold changes $N$, which changes the denominator and shifts the ranking.

### 2.4 Max Aggregation

$$U_{max}(X) = \max_{i=1}^{N} u_i$$

**Behavior:** The image's uncertainty is defined strictly by its worst-case (most confusing) bounding box.

**Strengths:**
- Immune to dilution. If even one highly uncertain object exists in the image, the image gets a high score regardless of how many easy objects surround it.
- Ideal for hunting edge cases and rare failure modes — the exact instances that cause safety-critical failures in autonomous driving.

**Weaknesses:**
- Discards all information about the rest of the image. An image with 1 hard object and 10 moderately hard objects scores identically to an image with 1 hard object and 10 trivial objects.
- Highly vulnerable to outlier region proposals. A single bad anchor or spurious detection can spike the max score without reflecting genuine model uncertainty.

### 2.5 The Economic Argument

The "best" aggregation function is not a pure engineering decision — it is an economic one.

Two independent studies reached opposite conclusions:
- **Brust et al. (VISAPP 2019):** Sum produces the best learning efficiency.
- **Probst et al. (2022):** Margin + Max is optimal; Mean degrades badly; Entropy underperforms random sampling.

The reconciliation is straightforward: **if labeling cost is per-image**, Sum wins because each selected image contains more objects to label "for free." **If labeling cost is per-bounding-box** (the common outsourcing model in Vietnam and Southeast Asia), Sum's advantage vanishes entirely because dense images cost proportionally more.

> **Rule:** Always determine your cost model *before* choosing your aggregation function.

---

## 3. The Redundancy Trap & Diversity Sampling

### 3.1 The Problem

Uncertainty sampling, applied naively to a batch of size $K$, will select the $K$ most uncertain instances from the pool. In video data at 30 FPS, consecutive frames are nearly identical. If frame 1201 confuses the model, frames 1202–1230 will also confuse it with nearly identical scores.

**Result:** Selecting Top-200 by uncertainty score yields 200 frames from the same 1-second clip. The labeling cost is 200 images; the information value is approximately 1 image.

This is the **redundancy trap** — the fatal flaw of pure uncertainty sampling at scale.

### 3.2 Choosing K (Batch Size)

There is no magic number for $K$. It is a trade-off:

| Consideration | Guidance |
|---------------|----------|
| **Theoretical ideal** | $K = 1$. Select 1 instance, label it, retrain, repeat. Zero redundancy. Completely impractical due to retraining cost. |
| **Rule of thumb** | $K$ = 1–5% of the unlabeled pool per round. |
| **Operational constraint** | $K$ = Number of images your annotation team can process in one sprint (e.g., 5,000 images/week). |
| **Retraining cadence** | $K$ is bounded by how often you can afford to retrain the model. Weekly retraining → $K$ = weekly annotation capacity. |

### 3.3 Solution: Hybrid Strategy (Uncertainty + Diversity)

Never use uncertainty alone for large batches. The standard industrial approach is a two-stage pipeline:

#### Method A: K-Means on Feature Embeddings (Industry Standard)

1. **Coarse filter (Uncertainty):** Run uncertainty scoring and select the Top-$M$ instances, where $M \gg K$ (typically $M = 3K$ to $5K$). Example: need $K = 1{,}000$, select $M = 5{,}000$ most uncertain images.
2. **Feature extraction:** Extract the output of the model's penultimate layer (the layer before the classification head) as a feature vector (embedding) for each of the $M$ images.
3. **Clustering:** Run K-Means on the $M$ feature vectors, forcing exactly $K$ clusters.
4. **Selection:** From each cluster, select the single instance closest to the cluster centroid.

**Result:** $K$ instances that are all genuinely difficult (they passed the uncertainty filter) but are semantically diverse (they belong to different clusters in feature space).

#### Method B: Feature-Space NMS (Fast, Greedy)

This operates identically to Non-Maximum Suppression in object detection, but applied to images in feature space:

1. Compute uncertainty score and feature vector for all images.
2. Sort all images by uncertainty in descending order.
3. Select the top image (guaranteed pick).
4. Compute cosine similarity between the selected image's feature vector and all remaining images.
5. **Suppress:** Remove all images with similarity $> \tau$ (e.g., $\tau = 0.9$). These are redundant duplicates.
6. Move to the next surviving image in the sorted list and repeat until $K$ images are selected.

**Result:** You sweep from most uncertain downward, but any image that "looks like" an already-selected image is discarded.

#### Comparison

| Method | Time Complexity | Quality | When to Use |
|--------|----------------|---------|-------------|
| K-Means | $O(M \cdot K \cdot d)$ per iteration | High — global diversity guarantee | Default choice for production |
| Feature NMS | $O(M \cdot K \cdot d)$ worst case, fast in practice | Good — greedy, may miss some global diversity | When you need speed or M is very large |

---

## 4. The Softmax Deception — Why Raw Probabilities Lie

### 4.1 The Core Problem

Every formula in Section 1 relies on $P(y \mid x)$ — the model's predicted probabilities. In deep learning, these probabilities come from the softmax function applied to raw logits $z$:

$$P(y_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

The problem: **modern deep neural networks are systematically miscalibrated.** They are pathologically overconfident. This is not a minor nuance — it fundamentally undermines the entire premise of uncertainty sampling.

### 4.2 How Cross-Entropy Loss Creates the Disease

During training with standard Cross-Entropy Loss, the optimizer does not stop when the model classifies correctly. The loss function continues to penalize the model until the predicted probability of the correct class approaches `1.0`.

To make softmax output `1.0`, the logit for the correct class must be infinitely larger than all other logits. The optimizer achieves this by **scaling up the model weights** — not by learning better features, but by amplifying the magnitude of the existing logit gap.

**Concrete numerical example:**

| Stage | Raw Logits | Softmax Output | Model's Reported Confidence |
|-------|-----------|----------------|----------------------------|
| **Early training** (honest) | `[2.0, 1.0, 0.5]` | `[0.62, 0.23, 0.14]` | 62% |
| **After overtraining** (inflated) | `[20.0, 10.0, 5.0]` | `[0.99995, 0.00004, 0.00001]` | 99.99% |

The ratio between logits is *identical* in both cases. The model has not learned anything new. It has simply inflated its weights to satisfy the loss function. Yet softmax now reports 99.99% confidence.

**Impact on active learning:** The model outputs `0.99` for genuinely difficult or out-of-distribution images. Least Confidence, Margin, and Entropy all look at `0.99` and conclude: "The model is certain — do not label this image." You have just lost the most valuable data points in your pool.

### 4.3 The Out-of-Distribution Catastrophe

Softmax is a closed-world function. It forces the total probability to sum to 1.0 across the known classes. If you feed an image of a shoe to a model trained only on dogs and cats, the raw logits for both classes will be very low. But softmax will *still* normalize them to sum to 1.0, potentially outputting `[0.55, 0.45]` — and the model will declare "55% confident it's a dog."

In active learning for autonomous driving, this means:
- Novel objects the model has never seen (a fallen tree, a wheelchair, an animal) will receive artificially high confidence scores.
- These are precisely the instances that should be prioritized for labeling.
- The uncertainty sampling pipeline will skip them entirely.

### 4.4 Three Practical Solutions

#### Solution 1: Temperature Scaling (Zero-Cost Fix)

Divide raw logits by a temperature constant $T > 1$ before applying softmax:

$$P(y_i) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}$$

- When $T = 1$: Standard softmax (the default).
- When $T > 1$: The distribution is "softened." Extreme outputs like `[0.99, 0.01]` are compressed to something like `[0.7, 0.3]`.
- When $T \to \infty$: The distribution approaches uniform.

**How to choose $T$:** Use a held-out calibration set. Optimize $T$ to minimize the Expected Calibration Error (ECE) on that set. This is a single scalar optimization — it takes minutes.

**Limitation:** Temperature scaling is a post-hoc global adjustment. It cannot fix instances where the model is confidently *wrong* in one region of feature space but correctly calibrated in another.

#### Solution 2: Energy-Based Scoring (Skip Softmax Entirely)

Instead of using softmax probabilities, compute uncertainty directly from raw logits using the energy score:

$$E(x) = -\log \sum_{k} e^{z_k}$$

If all logit values are low, the energy is high (the model has never learned features for this input). If logit values are high and peaked, the energy is low (the model recognizes this input well).

**Advantage:** The exponential amplification problem of softmax is avoided because we never normalize to a probability distribution.

#### Solution 3: Monte Carlo Dropout (Gold Standard, Expensive)

Instead of trusting a single forward pass, enable Dropout at inference time and run the image through the model $T$ times (typically $T = 10$–$30$). Each forward pass produces a different softmax distribution because Dropout randomly zeroes different neurons.

Compute the **variance** of the predictions across the $T$ runs:

$$\text{Var}(x) = \frac{1}{T} \sum_{t=1}^{T} \left( P_t(y \mid x) - \bar{P}(y \mid x) \right)^2$$

- **High variance:** The model's prediction changes drastically depending on which neurons are dropped. This indicates **epistemic uncertainty** — the model genuinely does not know.
- **Low variance:** The model consistently predicts the same thing regardless of dropout. It is confident for the right reasons.

**Cost:** $T \times$ the inference cost of a single forward pass. For large pools, this can be prohibitive. Common compromise: apply MC Dropout only to the Top-$M$ candidates pre-filtered by a cheaper scoring method.

### 4.5 Academic Reference

The foundational paper on this phenomenon is:
> Guo et al., *"On Calibration of Modern Neural Networks"*, ICML 2017.

Key finding: older, simpler models (shallow CNNs, Random Forests) were well-calibrated. Modern deep networks (ResNets, Transformers) become *progressively more miscalibrated* as depth and width increase. The larger and deeper the network, the more delusional its confidence.

---

## 5. Three Failure Modes of Active Learning

### 5.1 Cold Start

In the first few rounds of the active learning loop, the model is too immature for its uncertainty estimates to be meaningful. A model trained on 50 images has no reliable sense of what it does and does not know.

**Mitigation:** In the initial rounds, use **representative sampling** (e.g., random sampling or stratified sampling) to build a broad foundation. Switch to uncertainty-based sampling only after the model has seen enough data to produce calibrated predictions.

### 5.2 Collective Outlier Trap

Some images will always produce high uncertainty scores, but labeling them provides zero learning value:
- Images where the target object is 95% occluded.
- Pedestrians at extreme distance (4 pixels tall).
- Lens flare, severe motion blur, or sensor corruption.
- Night scenes with near-total darkness.

The model correctly reports high uncertainty on these images — it genuinely cannot classify them. But even with a human label, the model cannot learn useful features from these degraded inputs.

**Mitigation:** Apply a minimum quality filter before uncertainty scoring. Discard images below a threshold of visual quality (e.g., based on sharpness, brightness, minimum object pixel area).

### 5.3 Model Coupling

The dataset selected by active learning is specifically tailored to patch the weaknesses of Model A. Six months later, the team upgrades to Model B (a newer architecture). Model B has *different* weaknesses, and the dataset curated for Model A may perform worse than a random sample for Model B.

**Mitigation:** 
- Always maintain a randomly sampled baseline subset (e.g., 20% of the labeling budget) alongside the actively selected subset. This ensures the dataset retains general representativeness.
- Version and track datasets independently from models. As the course slide states: *"Annotated data often outlives models."*

---

## 6. Detecting Rubber-Stamping (Annotation QA)

### 6.1 The Problem

When annotators are tasked with reviewing AI pre-labels, there is a strong psychological incentive to simply click "Accept" on everything. The AI's output *looks* reasonable, and rejecting it requires effort and justification. An annotator with an Accept Rate of 99% is either:
- (a) An exceptionally skilled reviewer working with a near-perfect model, or
- (b) A rubber-stamper who is not actually reviewing the labels.

In practice, (b) is far more common. Here is how to distinguish them.

### 6.2 Method 1: Honeypot Injection

Secretly insert **gold frames** with deliberately corrupted AI labels into the annotator's work queue. Corruptions include:
- Deleting a bounding box from a clearly visible object.
- Changing the class label (e.g., `truck` → `bus`).
- Shifting a bounding box 20–30 pixels off-center.
- Adding a phantom box in empty space.

**Evaluation:** If the annotator accepts the honeypot without correction, they are provably not reviewing the labels. This is the strongest possible evidence of rubber-stamping.

**Implementation:** Honeypots should constitute 2–5% of the work queue and should be distributed unpredictably (not clustered at the beginning or end of a batch).

### 6.3 Method 2: False Negative Analysis (Added Rate)

A rubber-stamper only examines the boxes the model *has* drawn. They never proactively scan the image for objects the model *missed*. The "Added" metric (annotator manually drew a new box for a missed object) is the clearest behavioral signal.

**Evaluation:** In complex traffic scenes in Vietnam (motorbikes, street vendors, partially occluded pedestrians), a model will inevitably miss objects. If an annotator's Added rate is 0% or near-0% while their peers average 5–10%, the annotator is not scanning for false negatives.

### 6.4 Method 3: Time-to-Action Analysis

Measure the elapsed time between when the image appears on screen and when the annotator submits their review.

**Benchmarks:**
- A complex urban traffic scene with 10–30 objects requires a minimum of 15–30 seconds for a thorough review (scanning all objects, checking edges, looking for missed items).
- If the median review time is 1–3 seconds per image, the annotator is physically incapable of having reviewed the labels. Flag the entire batch for re-review.

**Implementation:** Log timestamps server-side (not client-side) to prevent manipulation. Compute per-annotator distributions and flag outliers.

### 6.5 Method 4: Shadow QA (Blind Re-Review)

Randomly sample 5% of the images an annotator marked as "Accepted" and route them to an independent senior QA reviewer for blind re-review (the QA reviewer does not see the original annotator's decision).

**Evaluation:** If the QA reviewer finds errors (Edited/Deleted/Added) in 15–20% of the sampled images, the original annotator's 99% Accept Rate is statistically meaningless. Compute the **residual error rate** (errors that survived the annotator's review) as the true quality metric.

---

## 7. Key References

1. Settles, B. *Active Learning Literature Survey.* CS Tech Report 1648, University of Wisconsin–Madison, 2009.
2. Guo, C. et al. *On Calibration of Modern Neural Networks.* ICML, 2017.
3. Brust, C.-A. et al. *Active Learning for Deep Object Detection.* VISAPP, 2019.
4. Probst, D. et al. *Evaluating Zero-cost Active Learning for Object Detection.* arXiv:2212.04211, 2022.
5. Haussmann, E. et al. (NVIDIA). *Scalable Active Learning for Object Detection.* IEEE IV, 2020.
6. Fort, K. & Sagot, B. *Influence of Pre-annotation on POS-tagged Corpus Development.* LAW IV @ ACL, 2010.
7. Küppers, F. et al. *Multivariate Confidence Calibration for Object Detection.* CVPR-W, 2020.
