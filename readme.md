# Лабораторная 1 - MPI

## Задание 1. Вычисление числа пи
Код: [first.c](first.c)

## Замер времени работы
Так как существует слишком много факторов, от которых зависит время работы. Мы попробуем подойти серьезно и замерять время не на одном запуске, а на N запусках (default: 10) и брать среднее. Результатом такого запуска является .csv файл с колонками `threads,pi,points_number,time`.

Для наглядности в этом же скрипте мы строим и графики.

Сам python скрипт: [measure_time.py](measure_time.py). 

**Пример запуска:**
```
python3 measure_time.py --filename first.c --output first.csv
```

**Параметры:**
- `--filename` - Path to code file (e.g., first.c) to run (required)
- `--retries` - Number of retries (default: 10)
- `--output` - File for stats (default: stats.csv)
