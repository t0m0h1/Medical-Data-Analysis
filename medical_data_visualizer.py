import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add an overweight column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# Normalize cholesterol and glucose values
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

def draw_cat_plot():
    # Convert data into long format
    df_cat = df.melt(id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Group and reformat the data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Create categorical plot
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', 
                      data=df_cat, kind='bar', height=5, aspect=1.2)
    
    # Save the figure
    fig.savefig('catplot.png')
    plt.close(fig.fig)
    
    return fig

def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'].between(df['height'].quantile(0.025), df['height'].quantile(0.975))) &
        (df['weight'].between(df['weight'].quantile(0.025), df['weight'].quantile(0.975)))
    ]

    # Compute the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", linewidths=0.5, 
                square=True, cbar_kws={"shrink": 0.5}, cmap='coolwarm', annot_kws={"size": 8})
    
    ax.set_title("Correlation Matrix Heatmap")

    # Save the figure
    fig.savefig('heatmap.png')
    plt.close(fig)

    return fig
