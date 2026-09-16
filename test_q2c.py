from src.data_loader import load_question
from src.model import QuadraticDisutilityModel

data = load_question("Q2_quadratic")
print(data.summary(), "\n")

results = QuadraticDisutilityModel(data).build().solve()
print(results)
print("disutility:", results.meta["disutility"])
