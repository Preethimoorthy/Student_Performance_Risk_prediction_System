
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Student Performance and Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance over University and Department")
st.subheader("Student Risk Prediction System")

# Load dataset
try:
    df = pd.read_csv("student_performance.csv")
except FileNotFoundError:
    st.error(
        "student_performance.csv was not found. "
        "Save your notebook dataset in the same folder as app.py."
    )
    st.stop()

features = [
    "attendance",
    "study_hours",
    "previous_marks",
    "assignments"
]

required_columns = features + ["passed", "department"]

if not all(col in df.columns for col in required_columns):
    st.error("Dataset is missing one or more required columns.")
    st.write("Required columns:", required_columns)
    st.stop()

df = df.dropna(subset=required_columns).copy()
df["passed"] = pd.to_numeric(df["passed"], errors="coerce")
df = df.dropna(subset=["passed"])
df["passed"] = df["passed"].astype(int)

if len(df) < 2 or df["passed"].nunique() < 2:
    st.error("The dataset needs at least two records and both pass/fail classes.")
    st.stop()

if not df["passed"].isin([0, 1]).all():
    st.error("Use 0 for Fail and 1 for Pass in the 'passed' column.")
    st.stop()

# Train Logistic Regression using the academic features
model = LogisticRegression(max_iter=1000)
model.fit(df[features], df["passed"])

# Navigation
section = st.sidebar.radio(
    "Select Section",
    [
        "University Performance",
        "Department Performance",
        "Student Risk Prediction"
    ]
)

# 1. Overall university performance
if section == "University Performance":
    st.header("Overall University Performance")

    total = len(df)
    passed = int((df["passed"] == 1).sum())
    failed = int((df["passed"] == 0).sum())

    pass_pct = passed / total * 100
    fail_pct = failed / total * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students", total)
    c2.metric("Passed Students", passed)
    c3.metric("Failed Students", failed)

    c4, c5 = st.columns(2)
    c4.metric("Pass Percentage", f"{pass_pct:.2f}%")
    c5.metric("Fail Percentage", f"{fail_pct:.2f}%")

    summary = pd.DataFrame({
        "Result": ["Passed", "Failed"],
        "Students": [passed, failed],
        "Percentage": [pass_pct, fail_pct]
    })

    st.bar_chart(summary.set_index("Result")["Percentage"])
    st.dataframe(summary, use_container_width=True)

# 2. Department-wise performance
elif section == "Department Performance":
    st.header("Department-wise Performance")

    stats = df.groupby("department").agg(
        Total_Students=("passed", "count"),
        Passed=("passed", lambda x: (x == 1).sum()),
        Failed=("passed", lambda x: (x == 0).sum())
    )

    stats["Pass_Percentage"] = (
        stats["Passed"] / stats["Total_Students"] * 100
    )
    stats["Fail_Percentage"] = (
        stats["Failed"] / stats["Total_Students"] * 100
    )

    st.dataframe(stats.round(2), use_container_width=True)

    st.subheader("Department Comparison")
    st.bar_chart(
        stats[["Pass_Percentage", "Fail_Percentage"]]
    )

    best = stats["Pass_Percentage"].idxmax()
    st.write(f"Highest pass percentage in this dataset: **{best}**")

# 3. Individual student prediction and risk
elif section == "Student Risk Prediction":
    st.header("Individual Student Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.slider("Attendance (%)", 0, 100, 60)
        study_hours = st.slider("Study Hours per Day", 0, 12, 2)

    with col2:
        previous_marks = st.slider("Previous Marks (%)", 0, 100, 50)
        assignments = st.slider("Assignments Completed (%)", 0, 100, 50)

    if st.button("Predict Performance"):
        student = pd.DataFrame([{
            "attendance": attendance,
            "study_hours": study_hours,
            "previous_marks": previous_marks,
            "assignments": assignments
        }], columns=features)

        predicted_result = int(model.predict(student)[0])

        failure_index = list(model.classes_).index(0)
        risk = model.predict_proba(student)[0][failure_index] * 100

        if predicted_result == 1:
            st.success("Predicted Result: PASS")
        else:
            st.error("Predicted Result: FAIL")

        st.metric("Estimated Risk of Failure", f"{risk:.2f}%")

        if risk < 30:
            st.success("Risk Level: Low")
        elif risk < 60:
            st.warning("Risk Level: Medium")
        else:
            st.error("Risk Level: High")

        st.caption(
            "Risk is the model-estimated probability of failure, "
            "not a guarantee of the student's actual result."
        )
