import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# --- إعدادات الصفحة ---
st.set_page_config(page_title="Steel Plates Faults Classifier", layout="wide")

st.title("🚀 نظام تصنيف عيوب الألواح الفولاذية")
st.markdown("""
هذا التطبيق يقوم بتحليل وتصنيف العيوب بناءً على الخصائص الفيزيائية، مع إظهار تحليل احتمالي لكل أنواع العيوب.
""")

# --- 1. تحميل البيانات ---
@st.cache_data 
def load_data():
    URL_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA"
    features = [
        'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
        'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
        'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
        'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index',
        'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index',
        'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index',
        'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas'
    ]
    faults = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    df = pd.read_csv(URL_PATH, sep=r"\s+", header=None)
    df.columns = features + faults
    return df, features, faults

df, features, faults = load_data()

# --- 2. معالجة البيانات وتدريب النموذج ---
y_multi = df[faults]
y = y_multi.idxmax(axis=1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X = df[features]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# تدريب النموذج (استخدام balanced للتعامل مع مشكلة Other_Faults)
@st.cache_resource
def train_model(X_t, y_t):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_t, y_t)
    return model

model = train_model(X_train, y_train)

# --- 3. قسم التنبؤ التفاعلي (Live Prediction) ---
st.divider()
st.subheader("🔮 التنبؤ وتحليل نسبة الثقة")
st.write("أدخل قيم الخصائص أدناه لرؤية كيف يوزع النموذج احتمالية الخطأ على كل الأنواع:")

# الحصول على أهم 5 ميزات
feat_importances = pd.Series(model.feature_importances_, index=features)
top_5_features = feat_importances.nlargest(5).index.tolist()

# إنشاء مدخلات المستخدم
input_data = {}
cols = st.columns(len(top_5_features))
for i, feat in enumerate(top_5_features):
    val = cols[i].number_input(f"{feat}", value=float(df[feat].mean()))
    input_data[feat] = val

# ملء بقية الميزات بالقيم المتوسطة
for feat in features:
    if feat not in input_data:
        input_data[feat] = df[feat].mean()

if st.button("تحليل العينة وتوزيع الاحتمالات"):
    input_df = pd.DataFrame([input_data])[features]
    
    # التنبؤ بالنوع
    prediction = model.predict(input_df)
    res = le.inverse_transform(prediction)[0]
    
    # حساب الاحتمالات لكل الفئات
    probs = model.predict_proba(input_df)[0]
    
    # عرض النتيجة الأساسية
    st.success(f"التصنيف النهائي المتوقع: **{res}**")
    
    # إنشاء DataFrame للاحتمالات لعرضها
    prob_df = pd.DataFrame({
        'نوع العيب': le.classes_,
        'نسبة الثقة (%)': probs * 100
    }).sort_values(by='نسبة الثقة (%)', ascending=False)

    # تقسيم العرض لنتائج نصية ورسم بياني
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.write("📋 **تفاصيل النسب:**")
        st.dataframe(prob_df.style.format({'نسبة الثقة (%)': '{:.2f}%'}))
        
    with col_res2:
        st.write("📊 **التحليل البياني للاحتمالات:**")
        fig_prob, ax_prob = plt.subplots()
        colors = ['red' if x == res else 'skyblue' for x in prob_df['نوع العيب']]
        sns.barplot(x='نسبة الثقة (%)', y='نوع العيب', data=prob_df, palette=colors, ax=ax_prob)
        st.pyplot(fig_prob)

# --- 4. الرسوم البيانية العامة (أسفل الصفحة) ---
st.divider()
st.subheader("📊 إحصائيات النموذج العامة")
tab1, tab2 = st.tabs(["أهمية الميزات", "مصفوفة الارتباك"])

with tab1:
    fig_feat, ax_feat = plt.subplots()
    feat_importances.nlargest(10).plot(kind='barh', color='lightgreen', ax=ax_feat)
    st.pyplot(fig_feat)

with tab2:
    y_pred = model.predict(X_test)
    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    st.pyplot(fig_cm)