"""Plot the ROC and precision-recall curves, then pick the decision threshold
that balances precision and recall the way the problem needs."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, precision_score, recall_score, f1_score,
                             confusion_matrix, accuracy_score, balanced_accuracy_score,
                             classification_report)

# --- 1. Example data and model -----------------------
X, y = make_classification(n_samples=2000, weights=[0.85, 0.15], class_sep=1.0,
                           n_informative=4, flip_y=0.02, random_state=0)  # 15% positives
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)
model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)

# predict: probabilities (for ROC / PR curves) and classes (for all other metrics)
y_score = model.predict_proba(X_te)[:, 1]          # probabilities
y_pred = (y_score >= 0.5).astype(int)              # 0/1 classes with threshold 0.5

# --- 2. Confusion matrix and metrics needs y_pred (0/1)--------------------------------------------------
# con labels=[0, 1] l'ordine è [[VN, FP], [FN, VP]]

vn, fp, fn, vp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
print("Matrice di confusione:\n", confusion_matrix(y_te, y_pred, labels=[0, 1]))
print(f"VP={vp}  FN={fn}  FP={fp}  VN={vn}\n")

# --- 3. metrics scikit-learn ---------------------------------------------
print(f"Accuracy           {accuracy_score(y_te, y_pred):.3f}")
print(f"Precision          {precision_score(y_te, y_pred):.3f}")
print(f"Recall/Sensibility {recall_score(y_te, y_pred):.3f}")
print(f"F1                 {f1_score(y_te, y_pred):.3f}")
print(f"Balanced accuracy  {balanced_accuracy_score(y_te, y_pred):.3f}")

# Specificità: sklearn non ha una funzione dedicata -> recall della classe 0
print(f"Specificità        {recall_score(y_te, y_pred, pos_label=0):.3f}")

# --- 4. metrics manual-----------------------------
print("\n manually:")
print(f"Accuracy    (VP+VN)/tot = {(vp + vn) / (vp + vn + fp + fn):.3f}")
print(f"Precisione  VP/(VP+FP)  = {vp / (vp + fp):.3f}")
print(f"Recall      VP/(VP+FN)  = {vp / (vp + fn):.3f}")
print(f"Specificity VN/(VN+FP)  = {vn / (vn + fp):.3f}")
p, r = vp / (vp + fp), vp / (vp + fn)
print(f"F1          2PR/(P+R)   = {2 * p * r / (p + r):.3f}")
print(f"FPR         FP/(FP+VN)  = {fp / (fp + vn):.3f}")
print(f"NPV         VN/(VN+FN)  = {vn / (vn + fn):.3f}")

# --- 5. Report complete ----------------------------------
print("\n" + classification_report(y_te, y_pred, target_names=["class 0", "class 1"]))

# --- 6. Plot ROC and precision-recall curves need probabilities----------------------------------

"""
roc_curve tries every possible threshold: one for each distinct value in y_score
For each threshold it computes two numbers, which become one point of the curve:ì
-fpr = false positive rate = FP / (FP + VN), the x-axis;
-tpr = true positive rate = recall, the y-axis.
-roc_thr holds the thresholds, from highest to lowest.
"""
fpr, tpr, roc_thr = roc_curve(y_te, y_score)

"""
the same, but for each threshold it computes precision and recall
"""
prec, rec, pr_thr = precision_recall_curve(y_te, y_score)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))


#draw the roc curve, fpr on x,tpr on y
ax[0].plot(fpr, tpr, label=f"model (AUC = {roc_auc_score(y_te, y_score):.2f})")
ax[0].plot([0, 1], [0, 1], "--", color="gray", label="random (AUC = 0.5)") #draw dashed line from (0,0) to (1,1),a reference line
ax[0].set(xlabel="False positive rate (1 - specificity)", ylabel="Recall (sensitivity)",
          title="ROC curve")
ax[0].legend()

#the precision recall curve, recall on x, precision on y
ax[1].plot(rec, prec, label=f"model (AP = {average_precision_score(y_te, y_score):.2f})")
#horizontal line at y_test mean
ax[1].axhline(y_te.mean(), ls="--", color="gray", label=f"random (= {y_te.mean():.2f} positives)")
ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
ax[1].legend()

# --- 7. The knob: the decision threshold --------------------------------------
print(f"{'threshold':>9} {'precision':>9} {'recall':>7} {'f1':>6}")

#try 9 thresholds, foreach compute precision and recall
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    y_pred_t = (y_score >= t).astype(int)
    print(f"{t:>9.1f} {precision_score(y_te, y_pred_t, zero_division=0):>9.2f} "
          f"{recall_score(y_te, y_pred_t):>7.2f} {f1_score(y_te, y_pred_t):>6.2f}")
#zero_division=0: with very high thresholds the model may predict no positives at all. 
#Precision is then 0/0, undefined; this setting makes it 0 instead of showing a warning.

#choose between
#1. best F1 - best balance between precision and recall
# compute f1 for all thresholds
#+ 1e-12 adds a tiny number to avoid dividing by zero when P and R are both 0
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
#get threshold where at the position where f1 is max np.argmax(f1s)- best balance thresold
t_f1 = pr_thr[np.argmax(f1s)]    

#As the threshold rises, recall falls and precision rises

#2. get threshold where recall is bigger than 90%,[-1] means the last position
t_rec = pr_thr[np.where(rec[:-1] >= 0.90)[0][-1]]   

#3. get threshold where precision is bigger than 90%, second [0] means the first position
t_prec = pr_thr[np.where(prec[:-1] >= 0.90)[0][0]]        

for name, t, style in [("max F1", t_f1, "o"), ("recall >= 0.90", t_rec, "s"), ("precision >= 0.90", t_prec, "^")]:
    y_pred_t = (y_score >= t).astype(int)
    p, r = precision_score(y_te, y_pred_t), recall_score(y_te, y_pred_t)
    print(f"{name:<18} threshold={t:.2f}  precision={p:.2f}  recall={r:.2f}")
    ax[1].plot(r, p, style, markersize=9, label=f"{name} (t={t:.2f})")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("roc_pr_curves.png", dpi=130)
plt.show()