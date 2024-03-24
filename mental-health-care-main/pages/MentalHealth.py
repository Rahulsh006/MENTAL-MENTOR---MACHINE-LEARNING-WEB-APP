import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
from PIL import Image


st.title("Let's Get Predict Your Mind  ")
#image = Image.open('medicine.png')
#st.image(image, caption='Just Relax.....Let's Get Alright ')


classifiers = st.sidebar.selectbox("Used Classifier",("SVC","LogisticRegression","Decision Tree","Random Forest","NaiveBayes"))
#st.write(classifiers)
st.write("Let's see a sample data entry")
st.write("Refer the following and fill your data")
data = pd.read_csv("survey.csv")
data.drop(columns=['Timestamp', 'Country', 'state', 'comments'], inplace = True)
data.drop(data[data['Age'] < 0].index, inplace = True) 
data.drop(data[data['Age'] > 100].index, inplace = True)
data['dailylife_interfere'] = data['dailylife_interfere'].fillna('Don\'t know' )
data['self_employed'] = data['self_employed'].fillna('No')
data['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male',
                     'Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                      'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make',], 'Male', inplace = True)

data['Gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female',
                     'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)',
                     'woman',], 'Female', inplace = True)

data["Gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary',
                     'fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                      'Agender', 'A little about you', 'Nah', 'All',
                      'ostensibly male, unsure what that really means',
                      'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?',
                      'Guy (-ish) ^_^', 'Trans woman',], 'Other', inplace = True)
list_col=['Age', 'Gender', 'self_employed', 'family_history', 'treatment',
       'dailylife_interfere', 'no_employees', 'remote_work', 'tech_company',
       'benefits', 'care_options', 'wellness_program', 'seek_help',
       'anonymity', 'leave', 'mental_health_consequence',
       'phys_health_consequence', 'coworkers', 'supervisor',
       'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence']

n_f = data.select_dtypes(include=[np.number]).columns
c_f = data.select_dtypes(include=[object]).columns

label_encoder = LabelEncoder()
for col in c_f:
    label_encoder.fit(data[col])
    data[col] = label_encoder.transform(data[col])   
st.write(data.head(2))
X = data.drop("treatment",axis=1)
y = data["treatment"]

def add_parameters_csv(clf_name):
    p = dict()
    if clf_name == "SVC":
        C = st.sidebar.slider("C",0.01,15.0)
        p["C"] = C
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        p["max_depth"] = max_depth
        p["n_estimators"] = n_estimators
    elif clf_name == "LogisticRegression":
        max_iter = st.sidebar.slider("max_iter",100,300)
        C = st.sidebar.slider("C",1,5)
        p["max_iter"] = max_iter
        p["C"] = C
    elif clf_name == "Decision Tree":
        min_samples_split = st.sidebar.slider("min_samples_split",2,5)
        p["min_samples_split"] = min_samples_split
    else:
        st.write("No Parameters selection")
    return p
p = add_parameters_csv(classifiers)

def get_Classifier_csv(clf_name,p):
    if clf_name == "SVC":
        clf = SVC(C=p["C"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=p["n_estimators"],max_depth=p["max_depth"],random_state=1200)
    elif clf_name == "LogisticRegression":
        clf = LogisticRegression(C=p["C"],max_iter=p["max_iter"])
    elif clf_name == "NaiveBayes":
        clf = GaussianNB()
    else:
        clf = DecisionTreeClassifier(min_samples_split=p["min_samples_split"])
    return clf
clf = get_Classifier_csv(classifiers,p)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1200)
clf.fit(X_train,y_train)
y_pred_test = clf.predict(X_test)
#st.write(f"classifier Used={classifiers}")
acc = accuracy_score(y_test,y_pred_test)
#st.write(f"accuracy score={acc}")
st.write("1. Enter your age ")
age = st.number_input("Age",0,100)
st.write("2. Enter your gender ")
st.write("Female:0 , Male : 1 , Other:2")
Gender = st.number_input("Gender",0,2)
st.write("3. Are you self employed or not?")
st.write(" No:0 , Yes:1")
self_employed = st.number_input("Self employed",0,1)
st.write(" 4. Do you have a family history of mental illness?")
st.write(" No:0 , Yes:1")
family_history = st.number_input("Family history",0,1)
st.write("5. Did your mental discomfort interferes your field of activity in any means?")
st.write(" Often:0 , Rarely:1 , Never:2 , Sometimes:3 , Don't know:4")
dailylife_interfere = st.number_input("Activity field interfere",0,4)
st.write("6. Number of people you may interact in a day?")
no_interactions = st.number_input("no of interactions",0,1000)
st.write("7. Do you work remotely (outside of an office) at least 50% of the time?")
st.write(" No:0 , Yes:1")
remote_work = st.number_input("remote_work",0,2)
st.write("8. Do you experiance any pressure from your activity(study or work) field?")
st.write(" Yes:0 , No:1")
workorstudy_pressure = st.number_input("Field",0,1)
st.write("9. Does any mental health benefits provided in your field?")
st.write(" Yes:0 , Don't know:1 ,No:2")
benefits = st.number_input("benefits",0,2)
st.write("10. Are you aware of the mental health care programmes and benefits availed at your field?")
st.write(" NotSure:0 ,No:1 ,Yes:2")
care_options = st.number_input("Care options",0,2)
st.write("11. Do you attended any mental wellness program conducted ?")
st.write(" No:0,Don't Know:1,Yes:2")
wellness_program = st.number_input("Wellness program",0,2)
st.write("12. Does your department provide resources to learn more about mental health issues and how to seek help?")
st.write(" Yes:0,Don't know:1,Yes:2")
seek_help = st.number_input("Seek help",0,2)
st.write("13. Did your anonymity protected if you choose to take advantage of mental health treatment resources?")
st.write(" Yes:0 , Don't know:1 , N0:2")
anonymity= st.number_input("Anonymity",0,2)
st.write("14. How easy is it for you to take medical leave for a mental health condition?")
st.write(" Easy:0 , Don't know:1 ,Somewhat difficult:2 ,Very difficult:3 ,Very easy:4")
leave= st.number_input("Leave",0,4)
st.write("15. Do you feel bad on sharing your mental health issue with any one of your colleague?")
st.write(" No:0 ,Maybe:1 , Yes:2 ")
mental_health_consequence = st.number_input("Sharing to individual",0,2)
st.write("16. Do you feel bad on discussing your mental health issue with in your colleague group?")
st.write(" No:0 , Yes:1 , Maybe:2")
phys_health_consequence= st.number_input("Share with group of colleagues",0,2)
st.write("17. Would you be willing to discuss a mental health issue with your friends?")
st.write(" Some of them:0 , No:1 , Yes:2")
colleagues= st.number_input("Friends",0,2)
st.write("18. Would you be willing to discuss a mental health issue with your family?")
st.write(" Yes:0 , No:1 , Some of them:2")
family= st.number_input("Family",0,2)
st.write("19. Would you bring up any general mental health talks with a anyone in a casual talk?")
st.write(" No:0 , Yes:1 , Maybe:2")
in_a_group= st.number_input("Mental health talks",0,2)
st.write("20. Would you bring up your physical health issue with a anyone in a casual talk?")
st.write(" Maybe:0 , No:1 , Yes:2")
individual= st.number_input("physical health talks",0,2)
st.write("21. Do you feel that your colleagues and family takes mental health as seriously as physical health?")
st.write("mental vs physical Yes:0 , Don't know:1 , No:2")
mental_vs_physical= st.number_input("Mental vs physical",0,2)
st.write("22. Have you experianced negative consequences from colleagues on your mental health conditions?")
st.write(" No:0 , Yes:1")
obs_consequence= st.number_input("Negative consequence",0,1)

data1={'Age':age, 'Gender':Gender, 'self_employed':self_employed, 'family_history':family_history,
       'dailylife_interfere':dailylife_interfere, 'no_interactions':no_interactions, 'remote_work':remote_work, 'workorstudy_pressure':workorstudy_pressure,
       'benefits':benefits, 'care_options':care_options, 'wellness_program':wellness_program, 'seek_help':seek_help,
       'anonymity':anonymity, 'leave':leave, 'mental_health_consequence':mental_health_consequence,
       'phys_health_consequence':phys_health_consequence, 'colleagues':colleagues, 'family':family,
       'in_a_group':in_a_group, 'individual':individual,
       'mental_vs_physical':mental_vs_physical, 'obs_consequence':obs_consequence}
df2 = pd.DataFrame(data1,index=["Name"])
y_pred_test1 = clf.predict(df2)
if(y_pred_test1==0):
    st.write("YOU MAY BETTER GO FOR A TREATMENT.")
    st.write("https://www.bbrfoundation.org/blog/everyday-mental-health-tips")
    st.write("https://www.mayoclinic.org/diseases-conditions/mental-illness/diagnosis-treatment/drc-20374974")
    st.write("Some inspiration videos:")
    st.write("https://www.mayoclinic.org/diseases-conditions/mental-illness/diagnosis-treatment/drc-20374974")
    st.write("https://www.youtube.com/watch?v=-GXfLY4-d8w")
else:
    st.write("YOU DO NOT NEED ANY TREATMENT NOW. :smile:")
    st.write("Just relax , watch the mentioned videos.....Also go through the exercices and tips provided. ")
    st.write("Some inspirational videos are:")
    st.write("https://www.mayoclinic.org/diseases-conditions/mental-illness/diagnosis-treatment/drc-20374974")
    st.write("https://www.youtube.com/watch?v=-GXfLY4-d8w")

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
