from src.data_loader import load_question
from src.model import LinearDisutilityModel
from src.plotting import plot_schedule, plot_sweep
from main import sweep_linear_disutility, RESULTS_DIR

data = load_question("Q2_linear")
print(data.summary(), "\n")

out = RESULTS_DIR / "Q2_linear"
out.mkdir(parents=True, exist_ok=True)

results = LinearDisutilityModel(data).build().solve()
print(results)
print("disutility:", results.meta["disutility"])
print(results.hourly[["load", "reference_load", "dev_pos", "dev_neg", "pv", "import", "export"]])

plot_schedule(results, data, save_to=out / "schedule_q2b.png")

sweep_df = sweep_linear_disutility()
sweep_df.to_csv(out / "cL_sweep.csv", index=False)
plot_sweep(sweep_df, data, save_to=out / "cL_sweep.png")
