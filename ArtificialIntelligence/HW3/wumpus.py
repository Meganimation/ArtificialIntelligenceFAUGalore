import itertools
import pandas as pd

cells = ['P12', 'P22', 'P31']
models = list(itertools.product([False, True], repeat=len(cells)))

def satisfies_kb(m):
    p12, p22, p31 = m
    rule1 = (p12 == False)             # No breeze at [1,1]
    rule2 = (p22 == True or p31 == True) # Breeze at [2,1]
    return rule1 and rule2

alpha1 = lambda m: m[0] == False
alpha2 = lambda m: m[1] == False


kb_models = [m for m in models if satisfies_kb(m)]
entails_alpha1 = all(alpha1(m) for m in kb_models)
entails_alpha2 = all(alpha2(m) for m in kb_models)

#visualization of all models
df = pd.DataFrame(models, columns=cells)
df['Satisfies KB'] = df.apply(lambda row: satisfies_kb(tuple(row)), axis=1)
print(df)

print(f"\nKB b) ([1,2] is safe): {entails_alpha1}")
print(f"KB c) ([2,2] is safe): {entails_alpha2}")