import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data from medical_examination.csv and assign it to the df variable
df = pd.read_csv("D:/medical-data-visualizer/boilerplate-medical-data-visualizer/medical_examination.csv")

# 2. Create the overweight column in the df variable
df['overweight'] = ((df['weight'] / (df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalize data by making 0 always good and 1 always bad for cholesterol and gluc
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Draw the Categorical Plot in the draw_cat_plot function
def draw_cat_plot():
    # 5. Create a DataFrame for the cat plot using pd.melt
    df_cat = pd.melt(df, id_vars="cardio", value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])
    
    # 6. Group and reformat the data in df_cat to split it by cardio and show the counts of each feature
    df_cat = df_cat.groupby(["cardio", "variable", "value"]).size().reset_index(name="total")

    # 7. Create a chart that shows the value counts of the categorical features
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar")
    
    return fig.fig  # 8. Return the figure

# 10. Draw the Heat Map in the draw_heat_map function
def draw_heat_map():
    # 11. Clean the data in the df_heat variable by filtering out segments that represent incorrect data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()
    
    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # 15. Plot the correlation matrix using sns.heatmap()
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", square=True, center=0, cbar_kws={"shrink": .5})

    return fig  # 16. Return the figure

# Example usage
if __name__ == "__main__":
    # Draw the categorical plot
    draw_cat_plot()

    # Draw the heat map
    draw_heat_map()

    # Show the plots
    plt.show()  # This line will display the plots

















'''


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data from medical_examination.csv and assign it to the df variable
# 1. Import the data from medical_examination.csv and assign it to the df variable
df = pd.read_csv("D:/medical-data-visualizer/boilerplate-medical-data-visualizer/medical_examination.csv")

# Print the first few rows of the DataFrame
print(df.head())



# 2. Create the overweight column in the df variable
# Calculate BMI and set 'overweight' as 1 if BMI > 25, else 0


df = pd.read_csv("D:/medical-data-visualizer/boilerplate-medical-data-visualizer/medical_examination.csv")



df['overweight'] = ((df['weight'] / (df['height'] / 100) ** 2) > 25).astype(int)


print(df.head(7))




# 3. Normalize data by making 0 always good and 1 always bad for 'cholesterol' and 'gluc'
# Set cholesterol and gluc to 0 if normal (value 1), else set to 1
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)

print(df.head(6))

df['gluc'] = (df['gluc'] > 1).astype(int)

print(df.head(6))



# 4: Draw the Categorical Plot - Simple Categorical Count Plot
def draw_cat_plot():
    # Melt the DataFrame for categorical plotting
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])
    print(df_cat.head(11))  # Print to confirm the data format

    # Draw the categorical count plot
    fig = sns.catplot(data=df_cat, x="variable", hue="value", col="cardio", kind="count")
    fig.set_axis_labels("variable", "total")
    fig.set_titles("{col_name} - Cardio")
    plt.suptitle("Categorical Count Plot (Q#4)", y=1.05)  # Suptitle to distinguish this plot
    plt.show()

draw_cat_plot()
plt.show()

    # 5 Create a DataFrame for the cat plot using pd.melt with specified values
df_cat= pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])
print(df_cat.head(12))



# 6 Group and reformat the data in df_cat to split it by cardio. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
def draw_grouped_cat_plot():
    # Group and reformat the data, renaming the count column to 'total'
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # Draw the categorical bar plot with grouped counts
    fig = sns.catplot(data=df_cat, x="variable", y="total", hue="value", col="cardio", kind="bar")
    fig.set_axis_labels("variable", "total")
    fig.set_titles("{col_name} - Cardio Grouped Counts")
    plt.suptitle("Grouped Categorical Bar Plot (Q#6)", y=1.05)  # Suptitle to distinguish this plot
    plt.show()

draw_grouped_cat_plot()

   

# 7 Convert the data into long format and create a chart that shows the value counts of the categorical features using the following method provided by the seaborn library import : sns.catplot()
def draw_long_format_plot():
    # Re-melt and group the data to make sure 'total' column exists in df_cat
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # Directly plot the long-format data with sns.catplot
    fig = sns.catplot(data=df_cat, kind='bar', x='variable', y='total', hue='value', col='cardio')
    fig.set_axis_labels("variable", "total")
    fig.set_titles("{col_name} (Cardio)")
    plt.suptitle("Long Format Categorical Plot (Q#7)", y=1.05)  # Suptitle to distinguish this plot
    

# Call the function
draw_long_format_plot()
plt.show()


# 8  Get the figure for the output and store it in the fig variable



# Assuming df is already defined and contains the data
# Step to prepare df_cat
df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

# Group and reformat the data to get counts for each feature
df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

# Function to draw the categorical plot
def draw_cat_plot():
    # Draw the categorical plot
    fig = sns.catplot(data=df_cat, x="variable", y="total", hue="value", col="cardio", kind="bar")
    fig.set_axis_labels("variable", "total")
    fig.set_titles("{col_name} (Cardio)")
    fig.despine(left=True)
    plt.title("Categorical Plot of Features by Cardio")
    
    return fig  # Return the figure object

# Call the function and store the figure in the fig variable
fig = draw_cat_plot()

# Optionally display the figure
plt.show()



# 9  Save the Final Plot as 'catplot.png'


# Assuming df is already defined and contains the data
# Step to prepare df_cat
df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

# Group and reformat the data to get counts for each feature
df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

# Function to draw the categorical plot
def draw_cat_plot():
    # Draw the categorical plot
    fig = sns.catplot(data=df_cat, x="variable", y="total", hue="value", col="cardio", kind="bar")
    fig.set_axis_labels("variable", "total")
    fig.set_titles("{col_name} (Cardio)")
    fig.despine(left=True)
    plt.title("Categorical Plot of Features by Cardio")

    # Save the figure as a PNG file
    fig.savefig('catplot.png')  # Save the plot
    print("Saved plot as 'catplot.png'")  # Confirmation message
    return fig  # Return the figure object

# Call the function and store the figure in the fig variable
fig = draw_cat_plot()

# Optionally display the figure
plt.show()



# 10 Draw the Heat Map in the draw_heat_map function



def draw_heat_map():
    # Step 1: Clean the data as specified
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Step 2: Calculate the correlation matrix
    corr = df_heat.corr()

    # Step 3: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Step 4: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Step 5: Plot the heatmap
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}
    )

    # Save the figure and return it for output
    fig.savefig('heatmap.png')
    return fig


    
    
    
# 11 Clean the data in the df_heat variable by filtering out the following patient segments that represent incorrect data:
# diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
# height is less than the 2.5th percentile (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
# height is more than the 97.5th percentile
# weight is less than the 2.5th percentile
# weight is more than the 97.5th percentile
    
df = pd.read_csv("D:/medical-data-visualizer/boilerplate-medical-data-visualizer/medical_examination.csv")
# Clean the data in df_heat based on specified filtering criteria
df_heat = df[
    (df['ap_lo'] <= df['ap_hi']) &  # Diastolic pressure should be less than or equal to systolic pressure
    (df['height'] >= df['height'].quantile(0.025)) &  # Height should be at least the 2.5th percentile
    (df['height'] <= df['height'].quantile(0.975)) &  # Height should be at most the 97.5th percentile
    (df['weight'] >= df['weight'].quantile(0.025)) &  # Weight should be at least the 2.5th percentile
    (df['weight'] <= df['weight'].quantile(0.975))    # Weight should be at most the 97.5th percentile
]

# Display the first few rows of df_heat to verify the filtering
print(df_heat.head(8))




    
  
    
    

# 12  Calculate the correlation matrix and store it in the corr variable


def draw_heat_map():
    # Step 1: Clean the data as specified
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Step 2: Calculate the correlation matrix
    corr = df_heat.corr()

    # Step 3: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Step 4: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Step 5: Plot the heatmap
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}
    )

    # Save the figure and return it for output
    fig.savefig('heatmap.png')
    return fig


 # 13    Generate a mask for the upper triangle and store it in the mask variable   


def draw_heat_map():
    # Step 1: Clean the data as specified
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Step 2: Calculate the correlation matrix
    corr = df_heat.corr()

    # Step 3: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Step 4: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Step 5: Plot the heatmap
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}
    )

    # Save the figure and return it for output
    fig.savefig('heatmap.png')
    return fig



    # 14  Set up the matplotlib figure
def draw_heat_map():
    # Step 1: Clean the data by filtering out incorrect patient segments
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Step 2: Calculate the correlation matrix and store it in the corr variable
    corr = df_heat.corr()

    # Step 3: Generate a mask for the upper triangle of the heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Step 4: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted figsize for better visibility

    # Step 5: Plot the correlation matrix using sns.heatmap
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt=".1f", 
        cmap="coolwarm", 
        center=0, 
        square=True, 
        linewidths=0.5, 
        cbar_kws={"shrink": 0.5},
        ax=ax  # Specify the axis to draw on
    )

    # Step 6: Set the title of the heatmap
    ax.set_title("Correlation Heatmap", fontsize=16)

    # Step 7: Return the figure
    return fig

# Call the function to test
fig = draw_heat_map()
plt.show()  # Show the heatmap



 # 15  Plot the correlation matrix using the method provided by the seaborn library import: sns.heatmap()




def draw_heat_map():
    # Step 1: Clean the data as specified
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Step 2: Calculate the correlation matrix
    corr = df_heat.corr()

    # Step 3: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Step 4: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Step 5: Plot the heatmap
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}
    )

    # Save the figure and return it for output
    fig.savefig('heatmap.png')
    return fig
# Call the function to test
fig = draw_heat_map()
plt.show()  # Show the heatmap


 # 16 Do not modify the next two lines



def draw_heat_map():
    # Step 1: Clean the data as specified
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Step 2: Calculate the correlation matrix
    corr = df_heat.corr()

    # Step 3: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Step 4: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Step 5: Plot the heatmap
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}
    )

    # Save the figure and return it for output
    fig.savefig('heatmap.png')
    return fig
# Call the function to test
fig = draw_heat_map()
plt.show()  # Show the heatmap


'''