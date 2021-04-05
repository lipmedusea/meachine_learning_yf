from data_treatment import load_data_new,data_transform_new
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if __name__ == '__main__':
    #加载数据
    sql = "select * from trail_boost"
    df = load_data_new(sql)
    print(len(df))
    #数据预处理
    df["order_apply_date"] = pd.to_datetime(df["order_apply_date"])
    df_select = df[["order_apply_date","contract_status_gpconcat"]]

    df_select["date"]=[x.strftime('%Y-%m-%d') for x in df_select["order_apply_date"]]
    df_select["is_sucess_by_contract"] = np.where(df_select["contract_status_gpconcat"].isnull(),0,1)

    ds=df_select["is_sucess_by_contract"].groupby([df_select["date"], df_select["is_sucess_by_contract"]]).count().unstack()
    ds.to_excel("train_data/ds.xlsx",encoding="utf-8")

    plt.plot(ds.index,ds[1])
    plt.show()

