import argparse
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Run an MPI program multiple times.")
    parser.add_argument(
        "--filename", required=True, help="Path to code file (e.g., first.c)"
    )
    parser.add_argument("--retries", default=10, help="Number of retries")
    parser.add_argument("--output", default="stats.csv", help="File for stats")

    return parser.parse_args()


def draw_graphs(output):
    df = pd.read_csv(output)

    df_merged = pd.merge(
        df[df["threads"] != 1].copy(),
        df[df["threads"] == 1][["points_number", "time"]],
        on="points_number",
        suffixes=("_parallel", "_serial")
    )

    df_merged["speedup"] = df_merged["time_serial"] / df_merged["time_parallel"]
    df_merged["efficiency"] = df_merged["speedup"] / df_merged["threads"]
    print(df_merged)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Задание 1 (расчет числа пи)", fontsize=14)

    axes[0].plot(df_merged["points_number"], df_merged["time_serial"], "o-", label="Последовательная версия")
    axes[0].plot(df_merged["points_number"], df_merged["time_parallel"], "o-", label="Параллельная версия (10 процессов)")
    axes[0].set_title("Сравнение по времени работы")
    axes[0].set_xlabel("Количество точек")
    axes[0].set_ylabel("Время, с")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df_merged["points_number"], df_merged["speedup"], "o-")
    axes[1].set_title("График ускорения")
    axes[1].set_xlabel("Количество точек")
    axes[1].set_ylabel("Ускорение")
    axes[1].grid(True)

    axes[2].plot(df_merged["points_number"], df_merged["efficiency"], "o-")
    axes[2].set_title("График эффективности")
    axes[2].set_xlabel("Количество точек")
    axes[2].set_ylabel("Эффективность")
    axes[2].grid(True)

    plt.tight_layout()
    output_file = output[:output.find('.')] + "_graph.png"
    plt.savefig(output_file, dpi=300)


def first_task(args):
    threads_all = [1, 10]
    points_numbers = [
        100,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        300000000,
        600000000,
        800000000,
        1000000000,
    ]
    # Build
    executable_filename = args.filename[: args.filename.rfind(".")]
    subprocess.run(
        ["mpicc", args.filename, "-o", executable_filename],
        capture_output=True,
        text=True,
    )

    with open(args.output, "w") as f:
        print("threads,pi,points_number,time", file=f)

        for threads in threads_all:
            for points_number in points_numbers:
                cur_string = ""
                times_sum = 0.0
                for i in range(args.retries):
                    # Execute + measure time
                    result = subprocess.run(
                        [
                            "mpiexec",
                            "-n",
                            str(threads),
                            executable_filename,
                            str(points_number),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    stdout = result.stdout
                    time_string = stdout[stdout.find("|") + 1 : stdout.rfind("|")]
                    print(time_string)
                    cur_string = time_string[: time_string.rfind(",")]
                    times_sum += float(time_string.split(",")[-1])

                    time.sleep(0.1)

                cur_string = f"{threads},{cur_string},{str(times_sum / args.retries)}"
                print("final: ", cur_string)
                print(cur_string, file=f)
    draw_graphs(args.output)


def main():
    args = parse_args()

    first_task(args)


if __name__ == "__main__":
    main()
