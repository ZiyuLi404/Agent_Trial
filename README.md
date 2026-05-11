# Agent_Trial
# Problem Formulation

## 1. Research Problem

This project studies how to evaluate clinical AI agents that may change over time during a clinical trial.

Traditional randomized controlled trials usually assume that the intervention remains fixed throughout the study. However, modern clinical AI agents are often updated continuously. Their behavior may change because of model upgrades, prompt modifications, retrieval changes, tool-use changes, planner updates, or workflow redesign. Therefore, treating the AI agent as a fixed intervention may lead to biased or invalid evaluation results.

The goal of this project is to design a **version-aware adaptive trial framework** for evaluating non-stationary clinical AI agents. The framework should detect meaningful changes in agent behavior, split the trial into version-specific epochs, and estimate the treatment effect of each agent version using concurrent control data.

---

## 2. Clinical Task Definition

The clinical task is defined as:

> Sequential diagnostic workup / triage.

In this task, a patient case is presented to the system step by step. At each step, the agent receives the currently available clinical information and decides what to do next.

### Input

The input is the current known information about a patient case, including:

```text
patient demographics
symptoms
medical history
available test results
previous actions already taken
current clinical context
```

Example input:

```text
A 55-year-old male presents with chest pain and shortness of breath.
He has a history of hypertension.
No ECG or troponin test has been performed yet.
```

### Output

The agent may output one or more of the following:

```text
next question to ask
next physical examination to perform
next diagnostic test to order
preliminary diagnosis
triage recommendation
final diagnosis
```

Example output:

```text
Order an ECG and troponin test.
The patient may have acute coronary syndrome.
Recommend urgent evaluation.
```

---

## 3. Intervention Definition

The intervention is not defined as only the language model itself.

Instead, the intervention is defined as a complete **clinical AI system**.

This system may include:

```text
large language model
prompt
retrieval module
external tools
planner
clinical workflow
user interface
test-ordering policy
triage recommendation policy
```

Therefore, an update to any important component of this system may change the behavior of the intervention.

We define the treatment intervention as:

> A versioned clinical AI system that assists with sequential diagnostic workup and triage.

The control arm can be one of the following:

```text
clinician alone
frozen baseline agent
standard non-updating clinical AI system
```

For the first prototype, we use:

```text
control = frozen baseline agent
treatment = live updating agent
```

This makes the experiment easier to simulate and implement.

---

## 4. Agent Version Definition

Let the clinical AI agent at time \(t\) be represented as a time-varying policy:

\[
\pi_t(a \mid x)
\]

where:

```text
x = current patient context
a = agent action
π_t = agent policy at time t
```

Because the agent may change over time, we do not assume that:

\[
\pi_1 = \pi_2 = \cdots = \pi_T
\]

Instead, we assume that the agent may have different versions:

\[
\pi^{(1)}, \pi^{(2)}, \pi^{(3)}, \ldots, \pi^{(V)}
\]

Each version corresponds to a period of relatively stable behavior.

We call this period an **epoch**.

For example:

```text
Epoch 1: version v1 is active
Epoch 2: version v2 is active
Epoch 3: version v3 is active
```

---

## 5. Update Definition

An update means that the clinical AI system has changed in a way that may affect its behavior.

We define three types of updates.

### 5.1 Declared Update

A declared update occurs when the developer or system provider explicitly announces that the agent has changed.

Examples:

```text
model upgraded from v1 to v2
new tool-use policy released
new retrieval database added
new clinical workflow deployed
```

### 5.2 Configuration Update

A configuration update occurs when an internal component changes, even if the base LLM is the same.

Examples:

```text
prompt changed
retrieval strategy changed
planner changed
tool policy changed
UI workflow changed
threshold for triage recommendation changed
```

### 5.3 Hidden Behavioral Update

A hidden behavioral update occurs when there is no public announcement, but the agent’s behavior changes noticeably.

For example, on the same fixed set of anchor cases, the agent may start giving different diagnoses, ordering different tests, or making different triage recommendations.

This type of update is important because real-world clinical AI systems may change without clear public documentation.

---

## 6. Outcome Definition

We evaluate the agent using several clinical outcomes.

For the first prototype, we use three main outcomes.

### 6.1 Diagnosis Accuracy

Diagnosis accuracy measures whether the agent reaches the correct diagnosis.

```text
Diagnosis accuracy = 1 if the diagnosis is correct
Diagnosis accuracy = 0 otherwise
```

Higher is better.

### 6.2 Unsafe Miss Rate

Unsafe miss rate measures whether the agent misses a dangerous condition or gives an unsafe recommendation.

For example:

```text
sending home a patient with possible acute coronary syndrome
failing to order urgent testing for a high-risk case
missing sepsis
missing stroke
```

Unsafe miss is coded as:

```text
unsafe_miss = 1 if the agent makes an unsafe miss
unsafe_miss = 0 otherwise
```

Lower is better.

### 6.3 Test Cost / Number of Tests

Test cost measures how many diagnostic tests the agent orders, or the total cost of ordered tests.

For example:

```text
ECG = 1 unit
troponin = 1 unit
CT scan = 3 units
MRI = 4 units
```

Lower test cost is not always better, because ordering too few tests may increase unsafe misses.

Therefore, this outcome should be interpreted together with diagnosis accuracy and unsafe miss rate.

---

## 7. Treatment Effect Definition

For each agent version \(v\), we estimate the treatment effect using only the cases from the same epoch.

Let:

```text
Y_i = outcome for patient case i
A_i = treatment assignment
A_i = 1 means treatment group
A_i = 0 means control group
E_v = epoch where version v is active
```

The treatment effect for version \(v\) is:

\[
\tau_v = \mathbb{E}[Y_i \mid A_i = 1, i \in E_v] - \mathbb{E}[Y_i \mid A_i = 0, i \in E_v]
\]

In words:

> The treatment effect of version \(v\) is the difference between the treatment group and the control group among cases enrolled during the same version-specific epoch.

This is important because version \(v\) should only be compared with concurrent control cases from the same time period.

---

## 8. Why Concurrent Control Is Necessary

If the agent changes over time, using old control data may be misleading.

For example:

```text
Version v1 is used in January.
Version v2 is used in March.
Patient cases in January may be different from patient cases in March.
Clinical practice may also change over time.
```

If we compare March treatment cases with January control cases, the estimated treatment effect may reflect time trends or patient distribution changes rather than the true effect of the new agent version.

Therefore, for each version, the primary comparison should use:

```text
treatment cases from the version-v epoch
control cases from the same version-v epoch
```

Historical data may be used only for secondary analysis or efficiency improvement, not as the main validity basis.

---

## 9. Notation Table

| Symbol | Meaning |
|---|---|
| \(i\) | Patient case index |
| \(t\) | Time index |
| \(x_i\) | Clinical context of patient case \(i\) |
| \(a_i\) | Agent action for case \(i\) |
| \(Y_i\) | Outcome for case \(i\) |
| \(A_i\) | Trial arm assignment |
| \(A_i = 1\) | Treatment group |
| \(A_i = 0\) | Control group |
| \(\pi_t\) | Agent policy at time \(t\) |
| \(\pi^{(v)}\) | Agent policy for version \(v\) |
| \(E_v\) | Epoch where version \(v\) is active |
| \(\tau_v\) | Treatment effect of version \(v\) |

---

## 10. Summary

This project evaluates a clinical AI agent that may continuously change over time. The agent is modeled as a time-varying policy rather than a fixed intervention. When the agent changes, the trial should detect the change, split the study into version-specific epochs, and estimate each version's effect using concurrent control data.

In the first prototype, the project will use a frozen baseline agent as the control arm and a live updating agent as the treatment arm. The main outcomes will be diagnosis accuracy, unsafe miss rate, and test cost.
