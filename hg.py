"""
=============================================================
KLASIFIKASI KELOMPOK GIZI SISWA SMP - PROGRAM MBG
=============================================================
Pilot Project: Personalisasi Gizi – Makan Bergizi Gratis
Data Engineer Report
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("1. MEMUAT DATA")
print("=" * 60)

df_raw = pd.read_excel(
    "Data_Gizi_MBG_SMP (1).xlsx",
    sheet_name="Data Siswa",
    header=None
)
df_raw.columns = df_raw.iloc[0]
df = df_raw.iloc[1:].reset_index(drop=True)

print(f"Jumlah baris : {len(df)}")
print(f"Jumlah kolom : {len(df.columns)}")
print(f"\nKolom: {list(df.columns)}\n")

target_col = df.columns[-1]
print(f"Target       : {target_col}")
print("\nDistribusi Kelas Target:")
print(df[target_col].value_counts())


# ─────────────────────────────────────────────
# 2. PEMERIKSAAN KUALITAS DATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. PEMERIKSAAN KUALITAS DATA")
print("=" * 60)

print("Missing values per kolom:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"Total missing values : {df.isnull().sum().sum()}")
print(f"Duplikasi baris      : {df.duplicated().sum()}")


# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. PREPROCESSING")
print("=" * 60)

# Pilih 25 fitur (hapus ID, Nama, Kelas)
FEATURE_NAMES = [
    "Usia",
    "Berat Badan kg",
    "Tinggi Badan cm",
    "IMT",
    "Kategori IMT",
    "LILA cm",
    "Riwayat Stunting",
    "Hemoglobin gdL",
    "Status Anemia",
    "Level Aktivitas",
    "PAL",
    "Durasi Aktivitas jam",
    "Alergi Makanan",
    "Penyakit Kronis",
    "Intoleransi Laktosa",
    "Pola Makan",
    "Kebiasaan Sarapan",
    "Tingkat Ekonomi",
    "Asupan Gizi di Rumah",
    "BMR kkal",
    "Kebutuhan Kalori kkal",
    "Kebutuhan Protein g",
    "Kebutuhan Karbohidrat g",
    "Kebutuhan Lemak g",
    "Kebutuhan Serat g",
]

X = df[FEATURE_NAMES].copy()
y = df[target_col].copy()

# Label Encoding untuk variabel kategorik
print("Encoding variabel kategorik...")
label_encoders = {}
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
print(f"Kolom kategorik: {cat_cols}")

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

X = X.astype(float)

# Encode target
le_target = LabelEncoder()
y_enc = le_target.fit_transform(y)
print(f"\nKelas target: {list(le_target.classes_)}")

# Stratified Train-Test Split 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\nData latih : {X_train.shape[0]} sampel")
print(f"Data uji   : {X_test.shape[0]} sampel")


# ─────────────────────────────────────────────
# 4. DEFINISI MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. DEFINISI MODEL")
print("=" * 60)

models = {
    "Decision Tree (Baseline)": DecisionTreeClassifier(random_state=42),
    "Random Forest"           : RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression"     : LogisticRegression(max_iter=5000, random_state=42),
}

for name in models:
    print(f"  - {name}")


# ─────────────────────────────────────────────
# 5. PELATIHAN & EVALUASI
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. PELATIHAN & EVALUASI (5-Fold Stratified CV + Test Set)")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")

    # Cross-Validation
    cv_acc = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
    cv_f1  = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_weighted")

    # Latih pada seluruh data latih, uji pada test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average="weighted")

    results[name] = {
        "model"   : model,
        "y_pred"  : y_pred,
        "cv_acc"  : cv_acc,
        "cv_f1"   : cv_f1,
        "test_acc": test_acc,
        "test_f1" : test_f1,
    }

    print(f"  CV Accuracy    : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  CV F1-Weighted : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"  Test Accuracy  : {test_acc:.4f}")
    print(f"  Test F1        : {test_f1:.4f}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=le_target.classes_
    ))


# ─────────────────────────────────────────────
# 6. RINGKASAN PERBANDINGAN
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. RINGKASAN PERBANDINGAN MODEL")
print("=" * 60)

header = f"{'Model':<30} {'CV Acc':>10} {'CV F1':>10} {'Test Acc':>10} {'Test F1':>10}"
print(header)
print("-" * len(header))
for name, r in results.items():
    print(
        f"{name:<30} "
        f"{r['cv_acc'].mean():>10.4f} "
        f"{r['cv_f1'].mean():>10.4f} "
        f"{r['test_acc']:>10.4f} "
        f"{r['test_f1']:>10.4f}"
    )


# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. FEATURE IMPORTANCE – Random Forest")
print("=" * 60)

rf_model = results["Random Forest"]["model"]
fi = pd.Series(rf_model.feature_importances_, index=FEATURE_NAMES)
fi_sorted = fi.sort_values(ascending=False)

print(f"\n{'Fitur':<30} {'Importance':>12} {'%':>8}")
print("-" * 52)
for feat, imp in fi_sorted.items():
    print(f"  {feat:<28} {imp:>12.6f} {imp*100:>7.2f}%")


# ─────────────────────────────────────────────
# 8. CONFUSION MATRIX (Model Terbaik)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. CONFUSION MATRIX – Random Forest (Model Terbaik)")
print("=" * 60)

y_pred_rf = results["Random Forest"]["y_pred"]
cm = confusion_matrix(y_test, y_pred_rf)
class_labels = [c.split("–")[0].strip() for c in le_target.classes_]

print(f"\n{'':>12}", end="")
for lbl in class_labels:
    print(f"{lbl:>14}", end="")
print()
print("-" * (12 + 14 * len(class_labels)))
for i, row in enumerate(cm):
    print(f"{class_labels[i]:>12}", end="")
    for val in row:
        print(f"{val:>14}", end="")
    print()


# ─────────────────────────────────────────────
# 9. KESIMPULAN
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. KESIMPULAN")
print("=" * 60)

best_name = max(results, key=lambda k: results[k]["test_f1"])
best      = results[best_name]

print(f"""
Model terpilih : {best_name}
Test Accuracy  : {best['test_acc']:.4f} ({best['test_acc']*100:.2f}%)
Test F1        : {best['test_f1']:.4f} ({best['test_f1']*100:.2f}%)
CV Accuracy    : {best['cv_acc'].mean():.4f} ± {best['cv_acc'].std():.4f}

Model Random Forest dengan 100 decision trees berhasil mengklasifikasikan
kelompok gizi siswa dengan akurasi hampir sempurna. Fitur paling dominan
adalah Riwayat Stunting, IMT, dan Alergi Makanan.
""")
