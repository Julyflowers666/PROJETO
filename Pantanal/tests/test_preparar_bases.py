import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preparar_bases import (
    calcular_dias_sem_chuva,
    integrar_bases,
    preparar_base_oficial,
)


class TestPreparacaoBases(unittest.TestCase):
    def test_dias_sem_chuva_nao_confunde_ausencia_com_seca(self):
        chuva = pd.Series([0.0, 0.5, 2.0, np.nan, 0.0])
        resultado = calcular_dias_sem_chuva(chuva).tolist()
        self.assertEqual(resultado[:3], [1.0, 2.0, 0.0])
        self.assertTrue(np.isnan(resultado[3]))
        self.assertEqual(resultado[4], 1.0)

    def test_alvo_representa_o_foco_do_dia_seguinte(self):
        datas = pd.date_range("2026-01-01", periods=4)
        inmet = pd.DataFrame({
            "data": datas,
            "temperatura_maxima": [30, 31, 32, 33],
            "umidade_minima": [60, 55, 50, 45],
            "precipitacao_total": [0, 0, 2, 0],
            "dias_sem_chuva": [1, 2, 0, 1],
        })
        focos = pd.DataFrame({"data": [datas[1], datas[3]]})
        base = integrar_bases(inmet, focos)
        # Em 01/01, o alvo é 1 porque houve foco em 02/01.
        self.assertEqual(base["foco_dia_seguinte"].tolist(), [1, 0, 1])

    def test_preparacao_completa_em_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            pasta = Path(tmp)
            datas = pd.date_range("2026-01-01", periods=40)
            linhas = []
            for data in datas:
                linhas.append({
                    "Data": data.strftime("%Y/%m/%d"),
                    "TEMPERATURA MÁXIMA (°C)": "35,0",
                    "UMIDADE RELATIVA MÍNIMA (%)": "30,0",
                    "PRECIPITAÇÃO TOTAL (mm)": "0,0",
                })
            pd.DataFrame(linhas).to_csv(pasta / "inmet.csv", sep=";", index=False)
            pd.DataFrame({
                "data_hora_gmt": [f"{data:%Y-%m-%d} 12:00:00" for data in datas[::5]],
                "latitude": [-18.1] * 8,
                "longitude": [-57.5] * 8,
                "municipio": ["Corumbá"] * 8,
            }).to_csv(pasta / "inpe.csv", index=False)
            saida = pasta / "base.csv"
            base = preparar_base_oficial(
                pasta / "inmet.csv", pasta / "inpe.csv", saida, municipio="Corumbá"
            )
            self.assertTrue(saida.exists())
            self.assertTrue(saida.with_name("base_relatorio.json").exists())
            self.assertEqual(len(base), 39)
            self.assertEqual(int(base["foco_dia_seguinte"].sum()), 7)


if __name__ == "__main__":
    unittest.main()
