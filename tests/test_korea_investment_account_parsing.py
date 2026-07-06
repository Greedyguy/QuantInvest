import logging
import importlib.util
from pathlib import Path
import unittest


CONNECTOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "kiwoom_api"
    / "core"
    / "korea_investment_connector.py"
)
SPEC = importlib.util.spec_from_file_location(
    "korea_investment_connector_under_test",
    CONNECTOR_PATH,
)
korea_investment_connector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(korea_investment_connector)
KoreaInvestmentConnector = korea_investment_connector.KoreaInvestmentConnector


def make_connector() -> KoreaInvestmentConnector:
    connector = KoreaInvestmentConnector.__new__(KoreaInvestmentConnector)
    connector.account = "12345678"
    connector.logger = logging.getLogger("test_korea_investment_account_parsing")
    return connector


class KoreaInvestmentAccountParsingTest(unittest.TestCase):
    def test_unsettled_cash_gap_is_not_treated_as_stock_value(self):
        connector = make_connector()
        balance = {
            "output1": [],
            "output2": [
                {
                    "dnca_tot_amt": "900000",
                    "ord_psbl_cash": "1200000",
                    "tot_evlu_amt": "1200000",
                }
            ],
        }

        account = connector.parse_account_balance_data(balance)

        self.assertEqual(account["stock_value"], 0.0)
        self.assertEqual(account["stock_value_source"], "empty_holdings")

    def test_direct_stock_evaluation_field_is_used_when_present(self):
        connector = make_connector()
        balance = {
            "output1": [],
            "output2": [
                {
                    "dnca_tot_amt": "900000",
                    "ord_psbl_cash": "900000",
                    "tot_evlu_amt": "1200000",
                    "scts_evlu_amt": "300000",
                }
            ],
        }

        account = connector.parse_account_balance_data(balance)

        self.assertEqual(account["stock_value"], 300000.0)
        self.assertEqual(account["stock_value_source"], "scts_evlu_amt")

    def test_holdings_are_summed_when_summary_stock_field_is_absent(self):
        connector = make_connector()
        balance = {
            "output1": [
                {"pdno": "005930", "hldg_qty": "3", "evlu_amt": "210000"},
                {"pdno": "000660", "hldg_qty": "2", "prpr": "140000"},
                {"pdno": "035420", "hldg_qty": "0", "evlu_amt": "100000"},
            ],
            "output2": [
                {
                    "dnca_tot_amt": "900000",
                    "ord_psbl_cash": "900000",
                    "tot_evlu_amt": "1390000",
                }
            ],
        }

        account = connector.parse_account_balance_data(balance)

        self.assertEqual(account["stock_value"], 490000.0)
        self.assertEqual(account["stock_value_source"], "output1_sum")


if __name__ == "__main__":
    unittest.main()
