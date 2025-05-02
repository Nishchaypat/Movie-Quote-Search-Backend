# Import Required Libraries
from flask import Flask, render_template, request
from flask import Markup
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from API_KEY import KEY

# Initialize OPEN AI Tool
openai.api_key = KEY
ENGINE = 'text-embedding-ada-002' # The embedding model
df = pd.read_csv('quote_embeds.csv')  
df['embedding'] = df['embedding'].apply(eval).apply(lambda x: np.array(x, dtype=np.float32))

# Create Flask App
app = Flask(__name__)

# Main Code
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_term = request.form.get('query')
        search_term_vector = get_embedding(search_term, engine=ENGINE)
        df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
        sorted_df = df.sort_values("similarities", ascending=False).head(5)
        result = f'<span style="font-weight: bold; color: red;">{sorted_df["type"].iloc[0]}:</span> {sorted_df["movie"].iloc[0]} <br> <span style="font-weight: bold; color: red;">Released in {sorted_df["year"].iloc[0]}</span> <br> <span style="font-weight: bold; color: red;"> The desired quote may be:</span> {sorted_df["quote"].iloc[0]}'
        result = Markup(result) 
        return render_template('index.html', result=result,query=search_term)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
