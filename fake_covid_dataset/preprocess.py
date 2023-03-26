import pandas as pd

propaganda = pd.read_csv('covid_propaganda_articles.csv', encoding='utf-8')
satire = pd.read_csv('covid_satire_articles.csv', encoding='utf-8')
real_or_fake = pd.read_csv('corona_fake_cleaned.csv', encoding='utf-8')
real = real_or_fake[real_or_fake['class'] == 'Reliable News']
fake = real_or_fake[real_or_fake['class'] == 'Hoax']

df_concat = pd.concat([real.sample(n=30,random_state=0), 
                       fake.sample(n=30, random_state=0), 
                       propaganda, 
                       satire
                       ],
                      axis=0)
df_concat.to_csv('covid_unreliable_news_cleaned.csv', index=False)