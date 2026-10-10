from src.data_loader import load_question
from src.model import QuadraticDisutilityModel
from src.plotting import plot_schedule
from main import sweep_quadratic_disutility, RESULTS_DIR

data = load_question("Q2_quadratic")
print(data.summary(), "\n")

out = RESULTS_DIR / "Q2_quadratic"
out.mkdir(parents=True, exist_ok=True)

results = QuadraticDisutilityModel(data).build().solve()
print(results)
print("disutility:", results.meta["disutility"])

hourly = results.hourly.copy()
hourly["deviation"] = hourly["load"] - hourly["reference_load"]
print(hourly[["load", "reference_load", "deviation", "pv", "import", "export"]].round(3))

plot_schedule(results, data, save_to=out / "schedule_q2c.png")

sweep_df = sweep_quadratic_disutility()
print(sweep_df.round(3).to_string())
sweep_df.to_csv(out / "cQ_sweep.csv", index=False)
