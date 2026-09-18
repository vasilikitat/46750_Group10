from src.data_loader import load_question
from src.model import LinearDisutilityModel

data = load_question("Q2_linear")
print(data.summary(), "\n")

results = LinearDisutilityModel(data).build().solve()
print(results)
print("disutility:", results.meta["disutility"])
print(results.hourly[["load", "reference_load", "dev_pos", "dev_neg", "pv", "import", "export"]])