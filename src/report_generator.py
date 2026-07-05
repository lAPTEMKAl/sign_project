"""
report_generator.py
Генерация краткого отчёта (PDF и Excel) на основе:
  - runs/<experiment>/comparison_table.csv  (сравнение 5 архитектур)
  - runs/history.db                          (история инференс-запросов из демо)

Закрывает этап 7 задания: "Реализовать сохранение истории запусков ... а также
генерацию краткого отчёта в PDF или Excel."
"""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from fpdf import FPDF


def load_comparison(experiment_dir: Path) -> pd.DataFrame:
    csv_path = experiment_dir / "comparison_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Не найдена {csv_path}. Сначала запустите run_all.py")
    return pd.read_csv(csv_path)


def load_history(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()
    return df


def build_excel(comparison_df: pd.DataFrame, history_df: pd.DataFrame, out_path: Path):
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        comparison_df.to_excel(writer, sheet_name="Сравнение архитектур", index=False)
        if not history_df.empty:
            history_df.to_excel(writer, sheet_name="История запросов", index=False)
            summary = pd.DataFrame({
                "Метрика": ["Всего запросов", "Средняя уверенность", "Среднее время, мс", "Уникальных классов"],
                "Значение": [
                    len(history_df),
                    f"{history_df['confidence'].mean()*100:.1f}%",
                    f"{history_df['inference_ms'].mean():.1f}",
                    history_df["predicted_class"].nunique(),
                ],
            })
            summary.to_excel(writer, sheet_name="Сводка", index=False)
    print(f"Excel-отчёт сохранён: {out_path}")


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Otchet: sravnenie arhitektur (raspoznavanie dorozhnyh znakov)", ln=True, align="C")
        self.ln(2)


def build_pdf(comparison_df: pd.DataFrame, history_df: pd.DataFrame, out_path: Path):
    # ВАЖНО: fpdf2 со встроенными шрифтами не поддерживает кириллицу без подключения TTF.
    # Для финального отчёта на русском рекомендуется docx (см. report/ШАБЛОН.docx),
    # этот PDF — техническая выжимка для быстрого контроля результатов (на латинице/транслите).
    pdf = ReportPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Sravnenie arhitektur (test set)", ln=True)
    pdf.set_font("Helvetica", "", 9)

    col_names = ["arch", "test_acc", "avg_inference_ms", "model_size_mb", "epochs_run"]
    available_cols = [c for c in col_names if c in comparison_df.columns]
    col_w = 190 / len(available_cols)

    pdf.set_font("Helvetica", "B", 9)
    for c in available_cols:
        pdf.cell(col_w, 7, c, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for _, row in comparison_df.iterrows():
        for c in available_cols:
            val = row[c]
            text = f"{val:.4f}" if isinstance(val, float) else str(val)
            pdf.cell(col_w, 7, text, border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Statistika po istorii zaprosov demo-modulya", ln=True)
    pdf.set_font("Helvetica", "", 9)
    if history_df.empty:
        pdf.cell(0, 7, "Istoriya pusta.", ln=True)
    else:
        pdf.cell(0, 7, f"Vsego zaprosov: {len(history_df)}", ln=True)
        pdf.cell(0, 7, f"Srednyaya uverennost: {history_df['confidence'].mean()*100:.1f}%", ln=True)
        pdf.cell(0, 7, f"Srednee vremya, ms: {history_df['inference_ms'].mean():.1f}", ln=True)
        pdf.cell(0, 7, f"Unikalnyh klassov predskazano: {history_df['predicted_class'].nunique()}", ln=True)

    pdf.output(str(out_path))
    print(f"PDF-отчёт сохранён: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, default="runs/experiment_1",
                     help="Папка с результатами run_all.py (содержит comparison_table.csv)")
    ap.add_argument("--history-db", type=str, default="runs/history.db")
    ap.add_argument("--out", type=str, default="report")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = load_comparison(Path(args.experiment))
    history_df = load_history(Path(args.history_db))

    build_excel(comparison_df, history_df, out_dir / "report.xlsx")
    build_pdf(comparison_df, history_df, out_dir / "report_summary.pdf")


if __name__ == "__main__":
    main()
