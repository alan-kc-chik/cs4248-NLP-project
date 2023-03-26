import pandas as pd 

df = pd.read_csv('FakeCovid_July2020.txt', encoding='utf-8')
df = df[df['lang']=='en']
df = df[['class', 'title', 'content_text']]
df.to_csv('FakeCovid_July2020_eng_only.txt', encoding='utf-8', index=False)