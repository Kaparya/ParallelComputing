import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import subprocess
import time


threads_all = [1, 3, 5, 7, 10]


def parse_args():
    parser = argparse.ArgumentParser(description="Run an MPI program multiple times.")
    parser.add_argument(
        "--filename", required=True, help="Path to code file (e.g., first.c)"
    )
    parser.add_argument("--retries", default=10, help="Number of retries")
    parser.add_argument("--output", default="stats.csv", help="File for stats")

    return parser.parse_args()


def build(filename):
    executable_filename = filename[: filename.rfind(".")]
    subprocess.run(
        ["mpicc", filename, "-o", executable_filename],
        capture_output=True,
        text=True,
    )
    return executable_filename


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
            label = "Последовательный"
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

    check_number = 1000000000
    cur_df = df_merged[df_merged['points_number'] == check_number]
    axes[3].plot(cur_df["threads"], cur_df["speedup"], "o-", label=label)
    axes[3].set_title(f"Ускорения от количества процессов (на {check_number} точках)")
    axes[3].set_xlabel("Количество процессов")
    axes[3].set_ylabel("Ускорение")
    axes[3].grid(True)

    axes[4].plot(cur_df["threads"], cur_df["efficiency"], "o-", label=label)
    axes[4].set_title(f"Эффективности от количества процессов (на {check_number} точках)")
    axes[4].set_xlabel("Количество процессов")
    axes[4].set_ylabel("Эффективность")
    axes[4].grid(True)

    plt.tight_layout()
    fig.delaxes(axes[-1])
    output_file = output[:output.find('.')] + "_graph.png"
    plt.savefig(output_file, dpi=300)

def draw_graphs_second(output):
    df = pd.read_csv(output)
    df['size'] = df['row_size'] * df['column_size']

    df_merged = pd.merge(
        df.copy(),
        df[df["threads"] == 1][["algorithm", "row_size", "column_size", "time"]],
        on=("algorithm", "row_size", "column_size"),
        suffixes=("_parallel", "_serial")
    )

    df_merged["speedup"] = df_merged["time_serial"] / df_merged["time_parallel"]
    df_merged["efficiency"] = df_merged["speedup"] / df_merged["threads"]
    print(df_merged)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()
    fig.suptitle("Задание 2 (Умножение матрицы на вектор)", fontsize=14)

    for threads in threads_all:
        label = f"{threads} процесса"
        if threads == 1:
            label = "Последовательный"
        elif threads in (2, 3, 4):
            label = f"{threads} процесса"
        else:
            label = f"{threads} процессов"

        cur_df = df_merged[df_merged['threads'] == threads].sort_values('size')
        rows = cur_df[cur_df['algorithm'] == 'rows']
        columns = cur_df[cur_df['algorithm'] == 'columns']
        axes[0].plot(rows["size"], rows["time_parallel"], "o--", label=label + '(Строки)')
        axes[0].plot(columns["size"], columns["time_parallel"], "o-", label=label + '(Столбцы)')
    
    axes[0].set_title("Сравнение по времени работы")
    axes[0].set_xlabel("Всего элементов в матрице (row_size * column_size)")
    axes[0].set_ylabel("Время, с")
    axes[0].legend()
    axes[0].set_xscale("log")
    axes[0].grid(True)

    check_number = 100000000
    cur_df = df_merged[df_merged['size'] == check_number]
    rows = cur_df[cur_df['algorithm'] == 'rows']
    columns = cur_df[cur_df['algorithm'] == 'columns']

    axes[1].plot(rows["threads"], rows["speedup"], "o--", label='Строки')
    axes[1].plot(columns["threads"], columns["speedup"], "o-", label='Столбцы')
    axes[1].set_title("Ускорения от количества процессов (1000x100000)")
    axes[1].set_xlabel("Количество процессов")
    axes[1].set_ylabel("Ускорение")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(rows["threads"], rows["efficiency"], "o--", label='Строки')
    axes[2].plot(columns["threads"], columns["efficiency"], "o-", label='Столбцы')
    axes[2].set_title("Эффективности от количества процессов")
    axes[2].set_xlabel("Количество процессов")
    axes[2].set_ylabel("Эффективность")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    output_file = output[:output.find('.')] + "_graph.png"
    plt.savefig(output_file, dpi=300)

def draw_graphs_third(output):
    threads_all = [1, 4]
    data = []
    with open(output, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    data.append({
                        'threads': int(parts[0]),
                        'points_number': int(parts[1]),
                        'time': float(parts[3]) 
                    })
                except (ValueError, IndexError):
                    continue
    
    df = pd.DataFrame(data)

    if len(df) == 0:
        raise ValueError(f"Не удалось загрузить данные из {output}")

    df_merged = pd.merge(
        df.copy(),
        df[df["threads"] == 1][["points_number", "time"]],
        on=("points_number"),
        suffixes=("_parallel", "_serial")
    )
    df_merged["speedup"] = df_merged["time_serial"] / df_merged["time_parallel"]
    df_merged["efficiency"] = df_merged["speedup"] / df_merged["threads"]

    print("Данные для графиков:")
    print(df_merged)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Анализ производительности (third1.csv)", fontsize=16)

    for threads in threads_all:
        cur_df = df_merged[df_merged["threads"] == threads]
        if len(cur_df) > 0:
            axes[0].plot(cur_df["points_number"], cur_df["time_parallel"], "o-", label=f"{threads} потоков")

    axes[0].set_title("Время работы")
    axes[0].set_xlabel("Количество точек")
    axes[0].set_ylabel("Время (с)")
    axes[0].set_xscale("linear")
    axes[0].grid(True)
    axes[0].legend()

    for threads in threads_all:
        cur_df = df_merged[df_merged["threads"] == threads]
        if len(cur_df) > 0:
            axes[1].plot(cur_df["points_number"], cur_df["speedup"], "o-", label=f"{threads} потоков")

    axes[1].set_title("Ускорение")
    axes[1].set_xlabel("Количество точек")
    axes[1].set_ylabel("Ускорение")
    axes[1].set_xscale("linear")
    axes[1].grid(True)
    axes[1].legend()

    for threads in threads_all:
        cur_df = df_merged[df_merged["threads"] == threads]
        if len(cur_df) > 0:
            axes[2].plot(cur_df["points_number"], cur_df["efficiency"], "o-", label=f"{threads} потоков")

    axes[2].set_title("Эффективность")
    axes[2].set_xlabel("Количество точек")
    axes[2].set_ylabel("Эффективность")
    axes[2].set_xscale("linear")
    axes[2].grid(True)
    axes[2].legend()

    plt.savefig(output.replace(".csv", "_third_graphs.png"), dpi=300)
    plt.show()

def first_task(args):
    threads_all = [1, 3, 5, 7, 10]
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

    executable_filename = build(args.filename)

    with open(args.output, "w") as f:
        print("threads,pi,points_number,time", file=f)

        for threads in threads_all:
            for points_number in points_numbers:
                cur_string = ""
                times_sum = 0.0
                for _ in range(args.retries):
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


def second_task(args):
    row_sizes = [10, 100, 1000]
    column_sizes = [10, 100000]
    filenames = ['second_rows.c', 'second_columns.c']
    executable_filenames = []
    
    for filename in filenames:
        executable_filenames.append(build(filename))
    
    with open(args.output, "w") as f:
        print("algorithm,threads,total_sum,row_size,column_size,time", file=f)
        for executable_filename in executable_filenames:
            algorithm = executable_filename.split('_')[-1]
            for threads in threads_all:
                for row_size in row_sizes:
                    for column_size in column_sizes:
                        cur_string = ""
                        times_sum = 0.0
                        for _ in range(args.retries):
                            # Execute + measure time
                            result = subprocess.run(
                                [
                                    "mpiexec",
                                    "-n",
                                    str(threads),
                                    executable_filename,
                                    str(row_size),
                                    str(column_size)
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

                        cur_string = f"{algorithm},{threads},{cur_string},{str(times_sum / args.retries)}"
                        print("final: ", cur_string)
                        print(cur_string, file=f)
    draw_graphs_second(args.output)

def third_task(args):
    threads_all = [1, 4]
    points_numbers = [
        100,
        400,
        800,
        1200,

    ]
    # Build
    executable_filename = args.filename[: args.filename.rfind(".")]
    subprocess.run(
        ["mpicc", args.filename, "-o", executable_filename],
        capture_output=True,
        text=True,
    )

    with open(args.output, "w") as f:
        print("threads,points_number,time", file=f)

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
    draw_graphs_third(args.output)


def main():
    args = parse_args()

    if "first" in args.filename:
        first_task(args)
    elif "second" in args.filename:
        second_task(args)
    elif "third" in args.filename:
        third_task(args)


if __name__ == "__main__":
    main()
