

Create the new col with max value from all cols
```
cols = []
df['new_col'] = df[cols].max(axis=1)
```