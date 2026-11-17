import os
import duckdb
def load_data(tx_path="train_transaction.csv",id_path="train_identity.csv"):

    con = duckdb.connect()

    tx_path = os.path.join("data",tx_path)
    id_path = os.path.join("data",id_path)
    
  

    if os.path.exists(tx_path) and os.path.exists(id_path):
        print("loadin datasets")

        query = f"""
            SELECT 
                t.TransactionID,
                t.isFraud,
                t.TransactionAmt,
                t.TransactionDT,
                t.ProductCD,
                t.card4,
                t.dist1,
                t.C1,
                t.V1,
                COALESCE(i.DeviceType, 'missing') AS DeviceType,
                COALESCE(i.DeviceInfo, 'unknown') AS DeviceInfo
            FROM '{tx_path}' t
            LEFT JOIN '{id_path}' i ON t.TransactionID = i.TransactionID
            LIMIT 50000
        """

        df = con.execute(query).df()
    else:
        print("Data files not detected on disk. Using built-in synthetic IEEE-CIS sample.")
        tx_df, id_df = generate_mock_ieee_data()
        con.register("tx_table", tx_df)
        con.register("id_table", id_df)
        query = """
            SELECT 
                t.TransactionID,
                t.isFraud,
                t.TransactionAmt,
                t.TransactionDT,
                t.ProductCD,
                t.card4,
                t.dist1,
                t.C1,
                t.V1,
                COALESCE(i.DeviceType, 'missing') AS DeviceType,
                COALESCE(i.DeviceInfo, 'unknown') AS DeviceInfo
            FROM tx_table t
            LEFT JOIN id_table i ON t.TransactionID = i.TransactionID
        """
        df = con.execute(query).df()
    return df

def generate_mock_ieee_data():
    """Generates a small synthetic replica matching the IEEE-CIS schema."""
    np.random.seed(42)
    n = 2000
    
    # 1. Transaction Table (parent)
    tx_ids = np.arange(1000, 1000 + n)
    transaction_df = pd.DataFrame({
        "TransactionID": tx_ids,
        "isFraud": np.random.choice([0, 1], size=n, p=[0.965, 0.035]), # ~3.5% fraud rate
        "TransactionAmt": np.random.exponential(scale=65.0, size=n),
        "TransactionDT": np.sort(np.random.randint(86400, 86400 * 30, size=n)),
        "ProductCD": np.random.choice(["W", "C", "R", "H", "S"], size=n),

        #normal distribution of n numbers ranging from 1000 to 9999
        "card1": np.random.randint(1000, 9999, size=n),
        "card4": np.random.choice(["visa", "mastercard", "discover"], size=n),

        #use exponential distribution replacing 40% of data with nan, simulate missing values
        "dist1": np.where(np.random.rand(n) > 0.4, np.random.exponential(15.0, size=n), np.nan),
        "C1": np.random.poisson(lam=1.5, size=n),
        "V1": np.random.normal(0, 1, size=n)
    })
    
    # 2. Identity Table (relational child - present for ~30% of transactions)
    id_mask = np.random.rand(n) < 0.35
    identity_df = pd.DataFrame({
        "TransactionID": tx_ids[id_mask],
        "DeviceType": np.random.choice(["desktop", "mobile"], size=sum(id_mask)),
        "DeviceInfo": np.random.choice(["Windows", "iOS Device", "MacOS", "SM-G960N"], size=sum(id_mask))
    })
    
    return transaction_df, identity_df