import warnings
warnings.filterwarnings('ignore')
from data_treatment import load_data_new, data_clean2
import pandas as pd

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



if __name__ == '__main__':
    sql = "select * from bidata.trail_pigeon"
    df = load_data_new(sql, filename="df_20190215.csv")

    student_sql = """SELECT student_id,count(student_id) from trail_pigeon 
                    GROUP BY student_id"""
    student_mul = load_data_new(student_sql, filename="student_mul.csv")
    student_ids = student_mul[student_mul["count(student_id)"]==1]

    df = df[df["student_id"].isin(student_mul["student_id"])]

    label_by_contract = "is_pigeon"
    labels = label_by_contract

    # 数据预处理
    df = data_clean2(df, min_date="2018-05-01",
                                     mid_date="2018-09-15",
                                     max_date="2018-09-30", label=labels)


    print("data_count", len(df))

    drop_features = [
        "order_id",
        "lesson_plan_id",
        "student_id",
        "student_no",
        "recent_scores",
        "order_apply_time",
        "adjust_start_time"
    ]
    df = df.drop(drop_features, axis=1)



    #单因素方差分析
    df_anova_single = pd.DataFrame()
    i = 0
    columns = list(df.columns)
    columns.remove("is_pigeon")
    for x in columns:
        s = 'is_pigeon ~ ' + x
        model = ols(s, df).fit()
        anovat = anova_lm(model)
        print(anovat.loc[x, :])
        df_anova_single = pd.concat([df_anova_single, pd.DataFrame(anovat.loc[x, :]).T], axis=0)
        df_anova_single.sort_values(by="PR(>F)", ascending=True, inplace=True, kind='quicksort', na_position='last')


    #多因素方差分析
    formula = 'is_pigeon ~ '
    for x in columns[:-1]:
        formula = formula + x + " + "
    formula = formula + columns[-1]
    anova_results = anova_lm(ols(formula, df).fit())
    print(anova_results)

    anova_results.sort_values(by="PR(>F)", ascending=True, inplace=True,
                                kind='quicksort', na_position='last')

    #卡方检验
    model1 = SelectKBest(chi2, k=len(columns)-2)  # 选择k个最佳特征
    # iris.data是特征数据，iris.target是标签数据，该函数可以选择出k个特征
    dm = df.drop("is_pigeon", axis=1)
    # dm.to_csv("load_data/dm.csv")
    model1.fit_transform(dm, df["is_pigeon"])
    #p值和得分 卡方95%对应的临界值为3.74
    ka_pvalue = pd.DataFrame(model1.pvalues_).T
    ka_scores = pd.DataFrame(model1.scores_).T

    ka = pd.concat([ka_pvalue, ka_scores], axis=0)
    ka.index = ["p_value", "scores"]
    ka.columns = dm.columns
    ka = ka.T
    ka.sort_values(by="p_value", ascending=True, inplace=True,
                              kind='quicksort', na_position='last')

    df_anova_single.to_csv("data_analysis/单因素方差分析.csv")
    anova_results.to_csv("data_analysis/多因素方差分析.csv")
    ka.to_csv("data_analysis/卡方检验.csv")













