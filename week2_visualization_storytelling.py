# Week 2 – Advanced Data Visualization and Storytelling with Python
# Dataset: Plotly Gapminder sample dataset (publicly documented at Gapminder.org)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = px.data.gapminder()
print(df.head())
print(df.info())
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# Derived metric
df["gdp_total"] = df["pop"] * df["gdpPercap"]

# 1. Life expectancy trend
trend = df.groupby(["year", "continent"], as_index=False)["lifeExp"].mean()
sns.lineplot(data=trend, x="year", y="lifeExp", hue="continent", marker="o")
plt.title("Life Expectancy Improved Across Every Continent")
plt.show()

# 2. GDP vs life expectancy
latest = df[df["year"] == 2007]
sns.scatterplot(data=latest, x="gdpPercap", y="lifeExp",
                size="pop", hue="continent", sizes=(30, 900), alpha=.75)
plt.xscale("log")
plt.title("Higher Income Is Associated with Longer Life Expectancy (2007)")
plt.show()

# 3. Population growth
popc = df.groupby(["year", "continent"], as_index=False)["pop"].sum()
sns.lineplot(data=popc, x="year", y="pop", hue="continent", marker="o")
plt.title("Population Growth Has Been Uneven Across Continents")
plt.show()

# 4. Life expectancy distribution
sns.boxplot(data=latest, x="continent", y="lifeExp")
sns.stripplot(data=latest, x="continent", y="lifeExp", color="black", alpha=.35, size=3)
plt.title("Life Expectancy Still Varied Widely Within Continents (2007)")
plt.show()

# 5. Top GDP per capita
top10 = latest.nlargest(10, "gdpPercap").sort_values("gdpPercap")
sns.barplot(data=top10, y="country", x="gdpPercap", hue="continent", dodge=False, legend=False)
plt.title("Highest GDP per Capita Countries in 2007")
plt.show()

# 6. Interactive map
fig = px.choropleth(latest, locations="iso_alpha", color="lifeExp",
                    hover_name="country", title="Global Life Expectancy by Country – 2007")
fig.show()

# 7. Animated narrative
fig = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop",
                 color="continent", hover_name="country",
                 animation_frame="year", log_x=True,
                 size_max=55,
                 title="The Development Story: Income, Health and Population")
fig.show()
