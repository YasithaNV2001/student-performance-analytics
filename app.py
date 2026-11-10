"""Streamlit demo: estimate a student's risk of failing Math, and explain it.

    streamlit run app.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402

from src import early_warning as ew  # noqa: E402

NOTE = 'A tool to offer support early, not to label students.'
REPO_URL = 'https://github.com/YasithaNV2001/student-performance-analytics'

st.set_page_config(page_title='Student Early Warning', page_icon='🎓', layout='centered')


@st.cache_resource
def get_model():
    """Load the saved model (train it first if the file is missing) and its test AUC."""
    if ew.MODEL_PATH.exists():
        pipe = ew.load()
    else:
        pipe, _ = ew.train()
        ew.save(pipe)
    return pipe, ew.holdout_auc(pipe)


@st.cache_data
def get_students():
    """All feature rows (for input ranges), the typical student used for defaults, and split sizes."""
    X, y = ew.load_data()
    X_tr, X_te, _, _ = ew.split(X, y)
    return X, ew.typical_student(X), len(X_tr), len(X_te)


pipe, auc = get_model()
X, typical, n_train, n_test = get_students()
t = typical.iloc[0]

# ---- Inputs ---------------------------------------------------------------
sb = st.sidebar
sb.header('About the student')
inputs = {
    'G1': sb.slider('First-term grade (G1, out of 20)', 0, 20, int(t['G1']), key='G1'),
    'failures': sb.select_slider('Past failed classes', options=[0, 1, 2, 3], value=int(t['failures']),
                                 help='3 means three or more', key='failures'),
    'absences': sb.number_input('Absences this year', 0, int(X['absences'].max()), int(t['absences']),
                                key='absences'),
    'studytime': sb.select_slider('Weekly study time', options=[1, 2, 3, 4], value=int(t['studytime']),
                                  format_func=ew.VALUE_LABELS['studytime'].get, key='studytime'),
    'higher': sb.radio('Plans for higher education', ['yes', 'no'], index=['yes', 'no'].index(t['higher']),
                       horizontal=True, key='higher'),
}
sb.subheader('Other factors')
inputs.update({
    'age': sb.slider('Age', int(X['age'].min()), int(X['age'].max()), int(t['age']), key='age'),
    'goout': sb.slider('Going out with friends (1 = very little, 5 = a lot)', 1, 5, int(t['goout']),
                       key='goout'),
    'famsup': sb.radio('Study support from family', ['yes', 'no'], index=['yes', 'no'].index(t['famsup']),
                       horizontal=True, key='famsup'),
    'schoolsup': sb.radio('Extra support from school', ['yes', 'no'],
                          index=['yes', 'no'].index(t['schoolsup']), horizontal=True, key='schoolsup'),
})

student = typical.copy()
for col, value in inputs.items():
    student.loc[0, col] = value

# ---- Prediction -----------------------------------------------------------
st.title('Student early-warning demo')
st.write('Estimates the risk that a student fails Math (final grade below 10 out of 20) using the '
         'first-term grade and the factors on the left. Fields not on the left are set to a typical student.')

risk = ew.fail_risk(pipe, student)
typical_risk = ew.fail_risk(pipe, typical)
diff = round((risk - typical_risk) * 100)
st.metric('Risk of failing Math', f'{risk:.0%}',
          delta=f'{diff:+d} points vs a typical student' if diff else None,
          delta_color='inverse')
if not diff:
    st.caption(f'Same as a typical student ({typical_risk:.0%}).')
st.info(NOTE)

# ---- Explanation ----------------------------------------------------------
st.subheader('Why this estimate?')
plt.figure()
shap.plots.waterfall(ew.explain(pipe, student), max_display=10, show=False)
st.pyplot(plt.gcf(), clear_figure=True)
plt.close('all')
st.caption('Red bars raise the risk and blue bars lower it. The chart starts from the model\'s '
           'baseline, E[f(X)], which is about 50% because failing and passing students get equal '
           'weight in training, and adds each factor until it reaches this student\'s risk, f(x). '
           'It shows patterns the model found in past students\' records, not causes, '
           'so some effects can look surprising.')

with st.expander('About the model'):
    st.write(f'Random Forest trained on {n_train} of the {len(X)} students in the UCI Student Performance '
             f'(Math) dataset and tested on the other {n_test} (AUC {auc:.2f}). It uses every recorded field except '
             f'the second-term and final grades, so it can be used after the first term. '
             f'Details and the full analysis: [GitHub]({REPO_URL}).')
    others = [c for c in X.columns if c not in inputs]
    st.write('Values used for the fields not on the left (the most common answer, or the median):')
    st.dataframe({'Field': [ew.LABELS[c] for c in others],
                  'Value': [ew.display_value(c, t[c]) for c in others]},
                 hide_index=True, width='stretch')
