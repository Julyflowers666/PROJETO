import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pantanal_guardian import calcular_metricas, classificar_risco, dividir_cronologicamente, validar_base


class TestPantanalGuardian(unittest.TestCase):
    def criar_base(self):
        n = 100
        return pd.DataFrame({
            "data": pd.date_range("2024-01-01", periods=n),
            "temperatura_maxima": np.linspace(28, 40, n),
            "umidade_minima": np.linspace(80, 20, n),
            "dias_sem_chuva": np.arange(n) % 15,
            "foco_dia_seguinte": [0, 1] * 50,
        })

    def test_divisao_cronologica(self):
        base = validar_base(self.criar_base())
        treino, validacao, teste = dividir_cronologicamente(base)
        self.assertEqual((len(treino), len(validacao), len(teste)), (70, 15, 15))
        self.assertLess(treino.data.max(), validacao.data.min())
        self.assertLess(validacao.data.max(), teste.data.min())

    def test_metricas_perfeitas(self):
        m = calcular_metricas(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertEqual(m["f1_score"], 1.0)
        self.assertEqual(m["taxa_falsos_alertas"], 0.0)

    def test_classificacao_provisoria(self):
        self.assertEqual(classificar_risco(0.10), "baixo")
        self.assertEqual(classificar_risco(0.30), "moderado")
        self.assertEqual(classificar_risco(0.60), "alto")
        self.assertEqual(classificar_risco(0.90), "crítico")


if __name__ == "__main__":
    unittest.main()
