## app.py
## Yuan Wang

from flask import Flask
import explore
app = Flask(__name__)
import pandas

@app.route('/jumps')
def jumpsdata():
    df = pandas.read_csv('data/jump180905.csv', header=0, index_col=0)
    df.columns = ['load', 'center_of_pressure']
    
    df, jumps = explore.process(df = df, split = 100)

    return jumps.to_json(orient="records")

@app.route('/timeseries')
def tsdata():
    df = pandas.read_csv('data/jump180905.csv', header=0, index_col=0)
    df.columns = ['load', 'center_of_pressure']
    
    df, jumps = explore.process(df = df, split = 100)

    return df.to_json(orient="records")

if __name__ == "__main__":


    app.run()