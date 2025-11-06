import argparse
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import time

threads_all = [1, 3, 5, 7, 10]
check_number = 1000000000

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
        df.copy(),
        df[df["threads"] == 1][["points_number", "time"]],
        on=("points_number"),
        suffixes=("_parallel", "_serial")
    )

    df_merged["speedup"] = df_merged["time_serial"] / df_merged["time_parallel"]
    df_merged["efficiency"] = df_merged["speedup"] / df_merged["threads"]
    print(df_merged)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    fig.suptitle("Задание 1 (Расчет числа пи)", fontsize=14)

    for threads in threads_all:
        label = f"{threads} процесса"
        if threads == 1:
            label = "Последовательная"
        elif threads == 3:
            label = "3 процесса"
        else:
            label = f"{threads} процессов"

        cur_df = df_merged[df_merged['threads'] == threads]
        axes[0].plot(cur_df["points_number"], cur_df["time_parallel"], "o-", label=label)

        axes[1].plot(cur_df["points_number"], cur_df["speedup"], "o-", label=label)
        axes[2].plot(cur_df["points_number"], cur_df["efficiency"], "o-", label=label)
    
    axes[0].set_title("Сравнение по времени работы")
    axes[0].set_xlabel("Количество точек")
    axes[0].set_ylabel("Время, с")
    axes[0].legend()
    axes[0].set_xscale("log")
    axes[0].grid(True)
    
    axes[1].set_title("Ускорение от времени")
    axes[1].set_xlabel("Количество точек")
    axes[1].set_ylabel("Ускорение")
    axes[1].legend()
    axes[1].set_xscale("log")
    axes[1].grid(True)

    axes[2].set_title("Эффективность от времени")
    axes[2].set_xlabel("Количество точек")
    axes[2].set_ylabel("Эффективность")
    axes[2].legend()
    axes[2].set_xscale("log")
    axes[2].grid(True)

    cur_df = df_merged[df_merged['points_number'] == check_number]
    axes[3].plot(cur_df["threads"], cur_df["speedup"], "o-", label=label)
    axes[3].set_title("Ускорения от количества процессов")
    axes[3].set_xlabel("Количество процессов")
    axes[3].set_ylabel("Ускорение")
    axes[3].grid(True)

    axes[4].plot(cur_df["threads"], cur_df["efficiency"], "o-", label=label)
    axes[4].set_title("Эффективности от количества процессов")
    axes[4].set_xlabel("Количество процессов")
    axes[4].set_ylabel("Эффективность")
    axes[4].grid(True)

    plt.tight_layout()
    fig.delaxes(axes[-1])
    output_file = output[:output.find('.')] + "_graph.png"
    plt.savefig(output_file, dpi=300)


def first_task(args):
    points_numbers = [
        100,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        300000000,
        1000000000
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
